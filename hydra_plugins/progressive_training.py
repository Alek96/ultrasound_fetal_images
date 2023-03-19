import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.core.utils import JobStatus
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, ListConfig, OmegaConf

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside sweep()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.

log = logging.getLogger(__name__)


@dataclass
class ProgressiveTrainingLauncherConfig:
    _target_: str = "hydra_plugins.progressive_training.ProgressiveTraining"
    # Sweeper arguments
    epochs: int = 100
    stages: int = 10
    ckpt: str = "last"
    params: dict | None = None


ConfigStore.instance().store(group="hydra/sweeper", name="progressive_training", node=ProgressiveTrainingLauncherConfig)


@dataclass
class ProgressiveTrainingSweeperLauncherConfig:
    _target_: str = "hydra_plugins.progressive_training.ProgressiveTrainingSweeper"
    # Sweeper arguments
    epochs: int = 100
    stages: int = 10
    runs_per_stage: int = 3
    params: dict | None = None


ConfigStore.instance().store(
    group="hydra/sweeper", name="progressive_training_sweeper", node=ProgressiveTrainingSweeperLauncherConfig
)


class ProgressiveTrainingSweeperBase(Sweeper):
    def __init__(
        self,
        epochs: int,
        stages: int,
        params: DictConfig,
    ):
        self.epochs = epochs
        self.stages = stages
        self.params = params or OmegaConf.create({})

        self.config: DictConfig | None = None
        self.launcher: Launcher | None = None
        self.hydra_context: HydraContext | None = None
        self.sweep_dir: Path | None = None

        self.job_results = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )
        self.hydra_context = hydra_context
        self.sweep_dir = Path(config.hydra.sweep.dir)

    def _parse_params_config(self) -> list[list[str]]:
        assert self.params is not None

        params_conf = [[] for _ in range(self.stages)]

        for k, value in self.params.items():
            if value.type == "linear":
                for i in range(self.stages):
                    if isinstance(value.min, ListConfig):
                        v = [
                            self._parse_value(i, v_min, v_max, value.dtype)
                            for v_min, v_max in zip(value.min, value.max)
                        ]
                    else:
                        v = self._parse_value(i, value.min, value.max, value.dtype)
                    params_conf[i].append(f"{k!s}={v}")
            elif value.type == "step":
                for i in range(len(value.steps)):
                    from_stage = value.steps[i].stage
                    to_stage = (value.steps[i + 1].stage - 1) if (i + 1 < len(value.steps)) else self.stages
                    for j in range(from_stage, to_stage):
                        params_conf[j].append(f"{k!s}={value.steps[i].value}")

        return params_conf

    def _parse_value(self, i, v_min, v_max, v_type):
        v = v_min + (v_max - v_min) * (i / (self.stages - 1))
        return self._parse_type(v, v_type)

    def _parse_type(self, v, v_type):
        if v_type == "float":
            return float(v)
        elif v_type == "int":
            return int(v)
        else:
            return v

    def sweep(self, arguments: list[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        log.info(f"{repr(self)}")
        params_conf = self._parse_params_config()
        return self._sweep(arguments, params_conf)

    def _sweep(self, arguments: list[str], params_conf: list[list[str]]) -> Any:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"epochs={self.epochs}, "
            f"stages={self.stages}, "
            f"params={self._parse_params_config()})"
        )


class ProgressiveTraining(ProgressiveTrainingSweeperBase):
    def __init__(
        self,
        epochs: int,
        stages: int,
        ckpt: Literal["best", "last"],
        params: DictConfig,
    ):
        super().__init__(epochs, stages, params)
        self.ckpt = ckpt

    def _sweep(self, arguments: list[str], params_conf: list[list[str]]) -> Any:
        returns = []
        for i in range(self.stages):
            conf = arguments + params_conf[i]
            if self.ckpt == "last":
                conf.append(f"trainer.min_epochs={int(self.epochs * (i / self.stages))}")
                conf.append(f"trainer.max_epochs={int(self.epochs * ((i + 1) / self.stages))}")
                if i > 0:
                    ckpt_path = self.sweep_dir / f"{i - 1}" / "checkpoints" / "last.ckpt"
                    conf.append(f"ckpt_path={ckpt_path}")
            elif self.ckpt == "best":
                conf.append(f"trainer.min_epochs={0}")
                conf.append(f"trainer.max_epochs={int(self.epochs / self.stages)}")
                if i > 0:
                    checkpoints = self.sweep_dir / f"{i - 1}" / "checkpoints"
                    ckpt_path = sorted(checkpoints.glob("epoch_*.ckpt"))[-1]
                    conf.append(f"model_path={ckpt_path}")

            conf = [conf]
            self.validate_batch_is_legal(conf)
            results = self.launcher.launch(conf, initial_job_idx=i)
            returns.append((i, results[0]))
        return returns


class ProgressiveTrainingSweeper(ProgressiveTrainingSweeperBase):
    def __init__(
        self,
        epochs: int,
        stages: int,
        runs_per_stage: int,
        params: DictConfig,
    ):
        super().__init__(epochs, stages, params)
        self.runs_per_stage = runs_per_stage

    def _sweep(self, arguments: list[str], params_conf: list[list[str]]) -> Any:
        best_run: dict = {"result": 0.0, "idx": -1}
        returns = []

        for i in range(self.stages):
            conf = arguments + params_conf[i]
            conf.append(f"trainer.min_epochs={0}")
            conf.append(f"trainer.max_epochs={int(self.epochs / self.stages)}")
            if i > 0:
                checkpoints = self.sweep_dir / f"{best_run['idx']}" / "checkpoints"
                ckpt_path = sorted(checkpoints.glob("epoch_*.ckpt"))[-1]
                conf.append(f"model_path={ckpt_path}")
                best_run["result"] = 0.0

            conf = [conf for _ in range(self.runs_per_stage)]
            self.validate_batch_is_legal(conf)
            results = self.launcher.launch(conf, initial_job_idx=i * self.runs_per_stage)
            for j, result in enumerate(results):
                if result.status == JobStatus.COMPLETED:
                    if best_run["result"] < result.return_value:
                        best_run["result"] = result.return_value
                        best_run["idx"] = i * self.runs_per_stage + j
                    returns.append((f"{i}.{j}", result.return_value))
                else:
                    returns.append((f"{i}.{j}", result.status))

        return returns
