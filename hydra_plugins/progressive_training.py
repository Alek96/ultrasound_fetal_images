import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
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
class LauncherConfig:
    _target_: str = "hydra_plugins.progressive_training.ProgressiveTrainingSweeper"
    # Sweeper arguments
    epochs: int = 100
    stages: int = 10
    params: dict | None = None


ConfigStore.instance().store(group="hydra/sweeper", name="progressive_training", node=LauncherConfig)


class ProgressiveTrainingSweeper(Sweeper):
    def __init__(self, epochs: int, stages: int, params: DictConfig):
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

        for k, (min_v, max_v, v_type) in self.params.items():
            for i in range(self.stages):
                if isinstance(min_v, ListConfig):
                    v = [self._parse_value(i, mi_v, ma_v, v_type) for mi_v, ma_v in zip(min_v, max_v)]
                else:
                    v = self._parse_value(i, min_v, max_v, v_type)
                params_conf[i].append(f"{k!s}={v}")
        return params_conf

    def _parse_value(self, i, min_v, max_v, v_type):
        v = min_v + (max_v - min_v) * (i / (self.stages - 1))
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

        # Save sweep run config in top level sweep working directory
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, self.sweep_dir / "multirun.yaml")

        params_conf = self._parse_params_config()

        returns = []
        for i in range(self.stages):
            conf = arguments + params_conf[i]
            conf.append(f"trainer.min_epochs={int(self.epochs * (i / self.stages))}")
            conf.append(f"trainer.max_epochs={int(self.epochs * ((i + 1) / self.stages))}")
            print(conf)
            if i > 0:
                ckpt_path = self.sweep_dir / f"{i-1}" / "checkpoints" / "last.ckpt"
                conf.append(f"ckpt_path={ckpt_path}")

            # conf = [conf]
            # self.validate_batch_is_legal(conf)
            # results = self.launcher.launch(conf, initial_job_idx=i)
            # returns.append((i, results[0]))
        return returns

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(epochs={self.epochs}, stages={self.stages}, params={self.params})"
