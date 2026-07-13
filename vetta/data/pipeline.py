from __future__ import annotations

import contextlib
import multiprocessing
import time
import timeit

from .common import _require_optional_dependency
from .buffers import ChunkPathBuffer
from .buffers import ChunkTreeBuffer
import numpy as np

from vetta.settings.data import ChunkTreeBufferSettings
from vetta.settings.data import TreePipelineServerSettings
from vetta.settings.data import TreePipelineSettings
from vetta.utils import make_json_serializable
from vetta.utils import printf

try:
    import zmq
except ModuleNotFoundError:
    zmq = None

# --- begin public data: pipeline.py ---
def _pipeline_log(enabled, *parts):
    if enabled:
        printf(" ".join(str(part) for part in parts))


def _server_bind_endpoint(tcp_port):
    return "tcp://*:{}".format(tcp_port)


def _process_is_alive(process, assume_started=False):
    is_alive = getattr(process, "is_alive", None)
    if callable(is_alive):
        with contextlib.suppress(AssertionError, ValueError):
            return is_alive()
        return False
    return assume_started


def _join_process(process, timeout=None):
    try:
        if timeout is None:
            return process.join()
        return process.join(timeout=timeout)
    except TypeError:
        return process.join()
    except (AssertionError, ValueError):
        return None


def _terminate_process(process):
    with contextlib.suppress(AssertionError, ValueError):
        process.terminate()


def _stop_process(process, *, timeout=None, assume_started=False):
    _join_process(process, timeout=timeout)
    if _process_is_alive(process, assume_started=assume_started):
        _terminate_process(process)
        _join_process(process, timeout=timeout)


def _close_zmq_objects(socket=None, context=None):
    if socket is not None:
        with contextlib.suppress(Exception):
            socket.close()
    if context is not None:
        with contextlib.suppress(Exception):
            context.term()


class TreePipelineServer:

    def __init__(self, chunk_paths, config=None):
        if config is None:
            config = TreePipelineServerSettings()
        elif not isinstance(config, TreePipelineServerSettings):
            raise TypeError(
                "TreePipelineServer config must be a TreePipelineServerSettings or None, "
                f"got {type(config).__name__}"
            )
        self.settings = config
        self.config = self.settings.to_dict()

        self.chunk_paths = chunk_paths

        assert self.config['tcp_port'] is not None

        self.finished = multiprocessing.Value('i', 0)
        self.process = multiprocessing.Process(
            target=self.process_loop, kwargs={}
        )
        self.process.daemon = False
        self.started = False

        self.prefix = "[" + self.config['name'] + "]"

        self.buffers_filled = multiprocessing.Value('b', False)
        self.context = None
        self.server = None


    @classmethod
    def default_config(cls):
        return TreePipelineServerSettings().to_dict()


    def create_buffers(self):
        self.chunk_path_buffer = ChunkPathBuffer(
            self.chunk_paths, seed=self.config['seed'],
            do_shuffle=self.config['do_shuffle']
        )
        input_queues = [self.chunk_path_buffer.output_queue]
        self.chunk_tree_buffers = []
        for i in range(0, self.config['n_chunk_tree_buffers']):
            chunk_tree_buffer_config = ChunkTreeBuffer.default_config()
            chunk_tree_buffer_config['load_segmentation'] = self.config['load_segmentation']
            chunk_tree_buffer_config['do_shuffle'] = self.config['do_shuffle']
            chunk_tree_buffer_config['target_queue_length'] = self.config['target_queue_length']
            chunk_tree_buffer_config['name'] = 'chunk_tree_buffer_{}'.format(i)
            chunk_tree_buffer_config['seed'] = (
                self.config['seed'] * self.config['seed_prime'] ** i
            )
            self.chunk_tree_buffers.append(
                ChunkTreeBuffer(
                    input_queues,
                    ChunkTreeBufferSettings.from_mapping(chunk_tree_buffer_config),
                )
            )
        self.tree_queues = [tree_buf.output_queue for tree_buf in self.chunk_tree_buffers]


    def start_buffers(self):
        _pipeline_log(self.config['verbose'], self.prefix, "starting chunk_path_buffer")
        self.chunk_path_buffer.start()
        _pipeline_log(self.config['verbose'], self.prefix, "chunk_path_buffer started")

        for i in range(0, self.config['n_chunk_tree_buffers']):
            _pipeline_log(self.config['verbose'], self.prefix, "starting chunk_tree_buffer {}".format(i))
            self.chunk_tree_buffers[i].start()
            _pipeline_log(self.config['verbose'], self.prefix, "chunk_tree_buffer {} started".format(i))

        _pipeline_log(self.config['verbose'], self.prefix, "filling chunk tree buffer output queues")
        filled = False
        while not filled:
            all_filled = True
            queue_lens = []
            for buf in self.chunk_tree_buffers:
                if buf.output_queue.qsize() < buf.target_queue_length.value:
                    all_filled = False
                queue_lens.append(buf.output_queue.qsize())
            _pipeline_log(self.config['verbose'], self.prefix, "queue lengths", queue_lens)
            filled = all_filled
            time.sleep(0.5)
        self.buffers_filled.value = True
        _pipeline_log(self.config['verbose'], self.prefix, "chunk tree buffer output queues filled")


    def stop_buffers(self, mode='join'):
        start_time = timeit.default_timer()
        chunk_path_buffer = getattr(self, "chunk_path_buffer", None)
        chunk_tree_buffers = getattr(self, "chunk_tree_buffers", [])
        _pipeline_log(self.config['verbose'], self.prefix, "stopping buffers")
        if chunk_path_buffer is not None:
            _pipeline_log(self.config['verbose'], self.prefix, "calling chunk_path_buffer.signal_stop")
            chunk_path_buffer.signal_stop()
            _pipeline_log(self.config['verbose'], self.prefix, "finished calling chunk_path_buffer.signal_stop")
        for k in range(0, len(chunk_tree_buffers)):
            _pipeline_log(self.config['verbose'], self.prefix, "calling chunk_tree_buffers[{}].signal_stop".format(k))
            chunk_tree_buffers[k].signal_stop()
        if mode == 'join':
            if chunk_path_buffer is not None:
                _pipeline_log(self.config['verbose'], "calling chunk_path_buffer.join()")
                chunk_path_buffer.join(
                    timeout=self.config['join_timeout'],
                    fallback='terminate'
                )
            for k in range(0, len(chunk_tree_buffers)):
                _pipeline_log(self.config['verbose'], "calling chunk_tree_buffers[{}].join()".format(k))
                chunk_tree_buffers[k].join(
                    timeout=self.config['join_timeout'],
                    fallback='terminate'
                )
        elif mode == 'terminate':
            if chunk_path_buffer is not None:
                chunk_path_buffer.terminate()
            for k in range(0, len(chunk_tree_buffers)):
                chunk_tree_buffers[k].terminate()
        else:
            raise ValueError("invalid mode")
        end_time = timeit.default_timer()
        _pipeline_log(self.config['verbose'], self.prefix, "stopped buffers in {:.2f} seconds".format(end_time - start_time))


    def start(self):
        self.process.start()
        self.started = True


    def signal_stop(self):
        self.finished.value = 1


    def terminate(self):
        _terminate_process(self.process)


    def join(self, timeout=None, fallback='return'):
        _join_process(self.process, timeout=timeout)
        if fallback == 'terminate' and _process_is_alive(self.process):
            _terminate_process(self.process)
            _join_process(self.process, timeout=timeout)
        return None


    def stop(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        _pipeline_log(self.config['verbose'], self.prefix, "stop() called")
        self.finished.value = 1
        if not self.started and not _process_is_alive(process):
            return
        join_timeout = self.config.get('join_timeout', 5.0)
        _pipeline_log(self.config['verbose'], self.prefix, "joining process")
        _stop_process(
            process,
            timeout=join_timeout,
            assume_started=self.started,
        )
        self.started = False
        _pipeline_log(self.config['verbose'], self.prefix, "process joined")


    def process_loop(self):
        _require_optional_dependency(zmq, "pyzmq")
        self.create_buffers()
        self.start_buffers()

        _pipeline_log(self.config['verbose'], "creating zmq objects")
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        try:
            endpoint = _server_bind_endpoint(self.config['tcp_port'])
            _pipeline_log(self.config['verbose'], "binding server to endpoint", endpoint)
            self.server.bind(endpoint)
            poller = zmq.Poller()
            poller.register(self.server, zmq.POLLIN)
            _pipeline_log(self.config['verbose'], "zmq objects created")

            sleep_time = 0.0
            active_time = 0.0
            pulling_time = 0.0
            sending_time = 0.0
            n_sent = 0
            last_report = timeit.default_timer()

            while self.finished.value == 0:
                socks = dict(poller.poll(self.config['poll_timeout']))
                vget = socks.get(self.server)
                if vget == zmq.POLLIN:
                    _pipeline_log(self.config['verbose'], self.prefix, "received request")
                    start_active = timeit.default_timer()
                    self.server.recv()
                    queue_idx = np.random.randint(0, len(self.tree_queues))
                    start_pull = timeit.default_timer()
                    item_dict = self.tree_queues[queue_idx].get()
                    _pipeline_log(
                        self.config['verbose'],
                        self.prefix,
                        "sending tree from",
                        item_dict['chunk_path'],
                    )
                    pulling_time += timeit.default_timer() - start_pull
                    if self.config['send_mode'] == 'tree':
                        start_send = timeit.default_timer()
                        tree = item_dict['tree']
                        metadata = {
                            'chunk_path': item_dict['chunk_path'],
                            'chunk_index': item_dict['chunk_index'],
                            'chunk_epoch': item_dict['chunk_epoch'],
                            'case_index': item_dict['case_index'],
                            'chunk_tree_buffer': item_dict['chunk_tree_buffer']
                        }
                        tree.metadata = make_json_serializable(metadata)
                        tree.send_zmq(self.server, zmq.NOBLOCK)
                        sending_time += timeit.default_timer() - start_send
                        n_sent += 1
                    else:
                        raise Exception("send_mode must be 'tree'")
                    active_time += timeit.default_timer() - start_active
                else:
                    try:
                        start_sleep = timeit.default_timer()
                        time.sleep(self.config['sleep_time'])
                        sleep_time += timeit.default_timer() - start_sleep
                    except KeyboardInterrupt:
                        self.finished.value = 1

                if self.config['report_stats'] and timeit.default_timer() - last_report >= self.config['report_delay']:
                    f = "{:.3f}"
                    dt = timeit.default_timer() - last_report
                    bdwth = float(n_sent) / dt
                    activeratio = active_time / dt
                    pullratio = pulling_time / dt
                    sendratio = sending_time / dt
                    report = self.prefix
                    report += " dt=" + f.format(dt)
                    report += " bdwth=" + f.format(bdwth) + "t/s"
                    report += " sleep=" + f.format(sleep_time)
                    report += " active=" + f.format(active_time)
                    report += " activeratio=" + f.format(activeratio)
                    report += " pull=" + f.format(pulling_time)
                    report += " pullratio=" + f.format(pullratio)
                    report += " send=" + f.format(sending_time)
                    report += " sendratio=" + f.format(sendratio)
                    _pipeline_log(True, report)
                    sleep_time = 0.0
                    active_time = 0.0
                    n_sent = 0
                    pulling_time = 0.0
                    sending_time = 0.0
                    last_report = timeit.default_timer()
        finally:
            _pipeline_log(self.config['verbose'], "closing zmq objects")
            _close_zmq_objects(self.server, self.context)
            self.server = None
            self.context = None
            _pipeline_log(self.config['verbose'], "stopping buffers")
            self.stop_buffers()
            _pipeline_log(self.config['verbose'], "buffers stopped")



class TreePipeline:

    def __init__(self, chunk_paths, config=None):
        if config is None:
            config = TreePipelineSettings()
        elif not isinstance(config, TreePipelineSettings):
            raise TypeError(
                "TreePipeline config must be a TreePipelineSettings or None, "
                f"got {type(config).__name__}"
            )
        self.settings = config
        self.config = self.settings.to_dict()

        self.chunk_paths = chunk_paths

        self.finished = multiprocessing.Value('i', 0)
        self.process = multiprocessing.Process(
            target=self.process_loop, kwargs={}
        )
        self.process.daemon = False

        self.prefix = "tree_pipeline"
        self.started = False
        self.buffers_filled = multiprocessing.Value('b', False)

        self.servers = []


    def __del__(self):
        with contextlib.suppress(Exception):
            self.stop()


    @classmethod
    def default_config(cls):
        return TreePipelineSettings().to_dict()


    def create_servers(self):
        for i in range(0, self.config['n_servers']):
            server_config = TreePipelineServer.default_config()
            server_config['mode'] = self.config['mode']
            server_config['seed'] = (
                self.config['seed'] * self.config['seed_prime'] ** i
            )
            server_config['do_shuffle'] = self.config['do_shuffle']
            server_config['n_chunk_tree_buffers'] = self.config['n_chunk_tree_buffers']
            server_config['verbose'] = self.config['verbose']
            server_config['target_queue_length'] = self.config['target_queue_length']
            server_config['tcp_port'] = self.config['start_port'] + i
            server_config['name'] = 'tree_pipeline_server_{}'.format(i)
            server_config['send_mode'] = self.config['send_mode']
            server_config['report_stats'] = self.config['report_stats']
            server_config['load_segmentation'] = self.config['load_segmentation']
            server_config['join_timeout'] = self.config['join_timeout']
            self.servers.append(TreePipelineServer(
                self.chunk_paths,
                TreePipelineServerSettings.from_mapping(server_config),
            ))


    def start_servers(self):
        _pipeline_log(self.config['verbose'], self.prefix, "starting servers")
        for server in self.servers:
            server.start()
        _pipeline_log(self.config['verbose'], self.prefix, "started servers")


    def stop_servers(self, mode='join'):
        _pipeline_log(self.config['verbose'], self.prefix, "stopping", len(self.servers), "servers")
        for server in self.servers:
            server.signal_stop()
        if mode == 'join':
            join_timeout = self.config.get('join_timeout', 5.0)
            _pipeline_log(self.config['verbose'], "joining", len(self.servers), "servers")
            for server in self.servers:
                _pipeline_log(self.config['verbose'], "calling server.join()")
                server.join(timeout=join_timeout, fallback='terminate')
        elif mode == 'terminate':
            for server in self.servers:
                server.terminate()
        else:
            raise ValueError("invalid mode")
        _pipeline_log(self.config['verbose'], self.prefix, "stopped servers")


    def start(self):
        self.process.start()
        self.started = True


    def stop(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        _pipeline_log(self.config['verbose'], "tree_pipeline.stop()")
        self.finished.value = 1
        join_timeout = self.config.get('join_timeout', 5.0)
        _stop_process(
            process,
            timeout=join_timeout,
            assume_started=self.started,
        )
        self.started = False
        self.process = None


    def terminate(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        self.finished.value = 1
        if self.started:
            _terminate_process(process)
        self.started = False


    def check_buffers_filled(self):
        if len(self.servers) == 0:
            self.buffers_filled.value = False
        else:
            val = True
            for server in self.servers:
                if not server.buffers_filled.value:
                    val = False
            self.buffers_filled.value = val
        return self.buffers_filled.value


    def process_loop(self):
        self.create_servers()
        self.start_servers()

        try:
            while self.finished.value == 0:
                try:
                    if not self.buffers_filled.value:
                        self.check_buffers_filled()
                    time.sleep(self.config['poll_time'])
                except KeyboardInterrupt:
                    self.finished.value = 1
        finally:
            self.stop_servers()

# --- end public data: pipeline.py ---
