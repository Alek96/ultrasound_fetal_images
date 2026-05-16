import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train


@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_brain_planes_train: DictConfig, cfg_brain_planes_eval: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_brain_planes_train: A DictConfig containing a valid training configuration.
    :param cfg_brain_planes_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_brain_planes_train.paths.output_dir == cfg_brain_planes_eval.paths.output_dir

    with open_dict(cfg_brain_planes_train):
        cfg_brain_planes_train.trainer.max_epochs = 0
        cfg_brain_planes_train.test = True

    HydraConfig().set_config(cfg_brain_planes_train)
    train_metric_dict, _ = train(cfg_brain_planes_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_brain_planes_eval):
        cfg_brain_planes_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_brain_planes_eval)
    test_metric_dict, _ = evaluate(cfg_brain_planes_eval)

    assert test_metric_dict["test/loss"] >= 0.0
    assert abs(train_metric_dict["test/loss"] - test_metric_dict["test/loss"]) < 0.001
