import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.run_if import RunIf


def test_train_fast_dev_run(cfg_video_quality_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.trainer.fast_dev_run = True
        cfg_video_quality_train.trainer.accelerator = "cpu"
    train(cfg_video_quality_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_video_quality_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.trainer.fast_dev_run = True
        cfg_video_quality_train.trainer.accelerator = "gpu"
    train(cfg_video_quality_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_video_quality_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.trainer.max_epochs = 1
        cfg_video_quality_train.trainer.val_check_interval = 0.5
    train(cfg_video_quality_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_video_quality_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_video_quality_train)
    metric_dict_1, _ = train(cfg_video_quality_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_video_quality_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_video_quality_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/loss"] != metric_dict_2["train/loss"]
    assert metric_dict_1["val/loss"] != metric_dict_2["val/loss"]


@pytest.mark.slow
def test_train_find_lr(cfg_video_quality_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        # cfg_video_quality_train.trainer.fast_dev_run = True
        cfg_video_quality_train.find_lr = True
    train(cfg_video_quality_train)


@pytest.mark.slow
def test_train_seq_step(cfg_video_quality_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.data.seq_len = 64
        cfg_video_quality_train.data.seq_step = 20
    train(cfg_video_quality_train)


@pytest.mark.slow
def test_train_reverse(cfg_video_quality_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.data.reverse = True
    train(cfg_video_quality_train)


@pytest.mark.slow
def test_train_transform(cfg_video_quality_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.data.transform = True
    train(cfg_video_quality_train)


@pytest.mark.slow
def test_train_reverse_transform(cfg_video_quality_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_video_quality_train)
    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.data.reverse = True
        cfg_video_quality_train.data.transform = True
    train(cfg_video_quality_train)
