from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
import multiprocessing
import os
import queue
import time
import warnings

import numpy as np
import PIL.Image

from vetta.settings.data import ChunkTreeBufferSettings

from .trees import TreeChunk


@dataclass(frozen=True)
class BufferConfig:
    target_queue_length: int = 10
    poll_time: float = 0.001
    name: str | None = None
    daemon: bool = True


@dataclass
class BufferRuntime:
    source_queues: list
    backlog_queue: object
    output_queue: object
    lock: object
    stop_event: object
    target_queue_length: object
    process: multiprocessing.Process | None = None


class _FinishedFlag:
    def __init__(self, stop_event):
        self._stop_event = stop_event

    @property
    def value(self) -> int:
        return int(self._stop_event.is_set())

    @value.setter
    def value(self, new_value: int) -> None:
        if int(new_value):
            self._stop_event.set()
        else:
            self._stop_event.clear()


class Buffer(metaclass=abc.ABCMeta):


    @abc.abstractmethod
    def __init__(self, input_queues, config: BufferConfig | None = None):
        self.buffer_config = config or BufferConfig()
        self.runtime = BufferRuntime(
            source_queues=input_queues,
            backlog_queue=multiprocessing.Queue(),
            output_queue=multiprocessing.Queue(),
            lock=multiprocessing.Lock(),
            stop_event=multiprocessing.Event(),
            target_queue_length=multiprocessing.Value(
                'i', self.buffer_config.target_queue_length
            ),
        )
        self.finished = _FinishedFlag(self.runtime.stop_event)
        self.poll_time = self.buffer_config.poll_time
        self.name = self.buffer_config.name


    def start(self):
        if self.process is None:
            self.process = self._make_process()
        self.process.start()


    def signal_stop(self):
        self.runtime.stop_event.set()


    def terminate(self):
        if self.process is not None:
            self.process.terminate()


    def join(self, timeout=None, fallback="return"):
        if self.process is None:
            return None
        self.process.join(timeout)
        if fallback == "terminate" and self._process_is_alive():
            self.terminate()
            self.process.join(timeout)
        return None


    def stop(self, mode='terminate'):
        self.signal_stop()
        self.clear_output_queue()
        if self.process is None:
            return None
        if mode == 'terminate':
            if self.process._popen is not None:
                self.process.terminate()
        elif mode == 'join':
            if self.process.pid != os.getpid():
                self.process.join()
            elif self.process._popen is not None:
                self.process.terminate()
        else:
            raise Exception("Unknown mode: {}".format(mode))


    def clear_output_queue(self):
        while self.output_queue.qsize() > 0:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            time.sleep(self.poll_time)


    def process_loop(self):
        # Detach the queue feeder threads so this worker can exit promptly when
        # signalled to stop. A multiprocessing.Queue otherwise blocks process
        # exit until its feeder thread flushes the (potentially large) backlog
        # into the pipe; once the consumer stops draining, that flush never
        # completes, the worker never exits, the parent's join() times out, and
        # the worker is orphaned. The backlog is discarded on shutdown anyway.
        self._cancel_queue_join_thread(self.output_queue)
        self._cancel_queue_join_thread(self.input_queue)
        while not self.runtime.stop_event.is_set():
            try:
                if self.output_queue.qsize() < self.target_queue_length.value:
                    self.try_put_next()
                time.sleep(self.poll_time)
            except KeyboardInterrupt:
                self.stop()


    def try_pull_input_queues(self):
        for i in range(0, len(self.input_queues)):
            in_dict = None
            try:
                in_dict = self.input_queues[i].get_nowait()
            except queue.Empty:
                pass
            if in_dict is not None:
                self.input_queue.put(in_dict)


    @abc.abstractmethod
    def try_put_next(self):
        pass


    def _process_is_alive(self):
        if self.process is None:
            return False
        is_alive = getattr(self.process, "is_alive", None)
        if callable(is_alive):
            return is_alive()
        return getattr(self.process, "_popen", None) is not None


    def _make_process(self):
        process = multiprocessing.Process(target=self.process_loop, kwargs={})
        process.daemon = self.buffer_config.daemon
        return process


    @staticmethod
    def _cancel_queue_join_thread(queue_obj):
        cancel_join_thread = getattr(queue_obj, "cancel_join_thread", None)
        if callable(cancel_join_thread):
            cancel_join_thread()


    @property
    def input_queues(self):
        return self.runtime.source_queues


    @input_queues.setter
    def input_queues(self, value):
        self.runtime.source_queues = value


    @property
    def input_queue(self):
        return self.runtime.backlog_queue


    @input_queue.setter
    def input_queue(self, value):
        self.runtime.backlog_queue = value


    @property
    def output_queue(self):
        return self.runtime.output_queue


    @output_queue.setter
    def output_queue(self, value):
        self.runtime.output_queue = value


    @property
    def lock(self):
        return self.runtime.lock


    @lock.setter
    def lock(self, value):
        self.runtime.lock = value


    @property
    def target_queue_length(self):
        return self.runtime.target_queue_length


    @target_queue_length.setter
    def target_queue_length(self, value):
        self.runtime.target_queue_length = value


    @property
    def process(self):
        return self.runtime.process


    @process.setter
    def process(self, value):
        self.runtime.process = value


class ChunkPathBuffer(Buffer):

    def __init__(self, chunk_paths, seed=1, do_shuffle=True):
        super().__init__(input_queues=[])
        self.seed = seed
        self.do_shuffle = do_shuffle
        self.seed = self.seed % 2**32
        np.random.seed(self.seed)
        self.local_state = np.random.get_state()
        np.random.set_state(copy.deepcopy(self.local_state))
        self.chunk_paths = chunk_paths
        self.index = multiprocessing.Value('i', 0)
        self.epoch = multiprocessing.Value('i', 0)
        self.reset_sequence()


    def reset_sequence(self):
        self.index.value = 0
        if self.do_shuffle:
            inherited_state = np.random.get_state()
            np.random.set_state(self.local_state)
            self.sequence = np.random.permutation(
                len(self.chunk_paths)
            )
            np.random.set_state(inherited_state)
        else:
            self.sequence = np.arange(len(self.chunk_paths))


    def try_put_next(self):
        seq_idx = self.sequence[self.index.value]
        chunk_path = self.chunk_paths[seq_idx]
        out_dict = {
            'chunk_path': chunk_path,
            'index': seq_idx,
            'epoch': self.epoch.value
        }
        self.output_queue.put(out_dict)
        self.index.value += 1
        if self.index.value == len(self.sequence):
            self.reset_sequence()
            self.epoch.value += 1


class ChunkTreeBuffer(Buffer):


    def __init__(self, chunk_paths_queues, config=None):
        if config is None:
            config = ChunkTreeBufferSettings()
        elif not isinstance(config, ChunkTreeBufferSettings):
            raise TypeError(
                "ChunkTreeBuffer config must be a ChunkTreeBufferSettings or None, "
                f"got {type(config).__name__}"
            )
        self.settings = config
        super().__init__(
            input_queues=chunk_paths_queues,
            config=BufferConfig(
                target_queue_length=self.settings.target_queue_length,
                name=self.settings.name,
            ),
        )
        self.config = self.settings.to_dict()
        self.seed = self.settings.seed
        self.seed = self.seed % 2**32
        np.random.seed(self.seed)
        self.local_state = np.random.get_state()
        np.random.set_state(copy.deepcopy(self.local_state))


    @classmethod
    def default_config(cls):
        return ChunkTreeBufferSettings().to_dict()


    def try_put_next(self):
        PIL.Image.MAX_IMAGE_PIXELS = None
        warnings.simplefilter('ignore', PIL.Image.DecompressionBombWarning)
        if self.input_queue.qsize() < 1:
            self.try_pull_input_queues()
        inherited_state = np.random.get_state()
        if self.input_queue.qsize() >= 1:
            in_dict = self.input_queue.get()
            chunk_path = in_dict['chunk_path']
            chunk = TreeChunk.load(chunk_path, load_segmentation=self.config['load_segmentation'])
            np.random.set_state(self.local_state)
            sequence = np.arange(len(chunk.trees))
            if self.config['do_shuffle']:
                sequence = np.random.permutation(len(chunk.trees))
            for i in range(0, len(chunk.trees)):
                seq_idx = sequence[i]
                out_dict = {
                    'tree': chunk.trees[seq_idx],
                    'chunk_path': chunk_path,
                    'chunk_index': in_dict['index'],
                    'chunk_epoch': in_dict['epoch'],
                    'case_index': seq_idx,
                    'chunk_tree_buffer': self.config['name']
                }
                self.output_queue.put(out_dict)
        self.local_state = np.random.get_state()
        np.random.set_state(inherited_state)


__all__ = [
    "Buffer",
    "BufferConfig",
    "BufferRuntime",
    "ChunkPathBuffer",
    "ChunkTreeBuffer",
]
