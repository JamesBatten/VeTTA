from __future__ import annotations

import abc
import os
import pathlib
import re

from vetta.training.environment import EnvironmentVariables
from vetta.utils import mkdir_custom


class BaseExperiment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def execute(self, output_dir: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    def get_next_experiment_name(cls, experiment_dir: str, n_pad: int = 6) -> str:
        experiment_names: list[str] = []
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        for element in os.listdir(experiment_dir):
            if re.match(r"experiment_*[0-9]", element):
                experiment_names.append(element)
        experiment_numbers: list[int] = []
        for exp_name in experiment_names:
            rhs = exp_name.split("_")[1]
            experiment_numbers.append(int(rhs.lstrip("0")))
            if len(rhs) != n_pad:
                raise ValueError(
                    f"padding scheme doesn't match: len({rhs}) != {n_pad}"
                )
        experiment_num = 1 if not experiment_numbers else max(experiment_numbers) + 1
        return "experiment_" + str(experiment_num).zfill(n_pad)

    def output_root(self) -> str:
        return os.path.join(
            pathlib.Path(__file__).resolve().parents[2],
            "experiments_data",
            "output_data",
        )

    def run(self) -> None:
        envvars_path = os.environ.get("ENVVARS")
        if envvars_path is None:
            raise RuntimeError("ENVVARS must be exported before running experiments")
        EnvironmentVariables.load(envvars_path)

        output_root = self.output_root()
        experiment_dir = mkdir_custom(
            os.path.join(output_root, self.name()),
            dir_exists_mode="return",
        )
        output_dir = mkdir_custom(
            os.path.join(
                experiment_dir,
                BaseExperiment.get_next_experiment_name(experiment_dir),
            )
        )
        self.execute(output_dir)
