from __future__ import annotations

import os
from typing import Any

from vetta.utils import listdir_fullpath
from vetta.utils import load_json


class EnvironmentVariables:
    class _Instance:
        def __init__(self, envvars: dict[str, Any] | None):
            self.envvars = envvars
            self.path: str | None = None

    instance: _Instance | None = None

    def __init__(self, envvars: dict[str, Any] | None = None):
        if EnvironmentVariables.instance is None:
            EnvironmentVariables.instance = EnvironmentVariables._Instance(envvars)
        elif envvars is not None:
            EnvironmentVariables.instance.envvars = envvars

    @classmethod
    def load(cls, path: str) -> "EnvironmentVariables":
        env = EnvironmentVariables(load_json(path))
        if env.instance is None:
            raise RuntimeError("environment variable singleton is not initialised")
        env.instance.path = path
        return env

    @classmethod
    def get_path(cls) -> str:
        if cls.instance is None or cls.instance.path is None:
            raise RuntimeError("environment variable path is not available")
        return cls.instance.path

    def __getitem__(self, name: str) -> Any:
        if self.instance is None or self.instance.envvars is None:
            raise RuntimeError("environment variables are not loaded")
        return self.instance.envvars[name]

    def has_item(self, name: str) -> bool:
        if self.instance is None or self.instance.envvars is None:
            raise RuntimeError("environment variables are not loaded")
        return name in self.instance.envvars


def _get_dataset_dir(dataset: str) -> str | None:
    env = EnvironmentVariables()
    key_map = {
        "ssa": "SSA",
    }
    key = key_map.get(dataset)
    if key is None:
        raise ValueError(f"unknown dataset {dataset!r}")
    if not env.has_item(key):
        return None
    return env[key]


def get_chunk_paths(
    dataset: str = "ssa",
    mode: str = "train",
    chunks: str = "chunks_all",
) -> list[str] | None:
    dataset_dir = _get_dataset_dir(dataset)
    if dataset_dir is None:
        return None
    return listdir_fullpath(os.path.join(dataset_dir, chunks, mode))
