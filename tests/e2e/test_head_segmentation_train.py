"""Integration (e2e) tests for the head segmentation train cycle via Hydra config."""

from __future__ import annotations

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.run_if import RunIf


def test_train_fast_dev_run(cfg_head_segmentation_train: DictConfig) -> None:
    """Run one train, val, and test step on CPU.

    :param cfg_head_segmentation_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_head_segmentation_train)
    with open_dict(cfg_head_segmentation_train):
        cfg_head_segmentation_train.trainer.fast_dev_run = True
        cfg_head_segmentation_train.trainer.accelerator = "cpu"
    train(cfg_head_segmentation_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_head_segmentation_train: DictConfig) -> None:
    """Run one train, val, and test step on GPU.

    :param cfg_head_segmentation_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_head_segmentation_train)
    with open_dict(cfg_head_segmentation_train):
        cfg_head_segmentation_train.trainer.fast_dev_run = True
        cfg_head_segmentation_train.trainer.accelerator = "gpu"
    train(cfg_head_segmentation_train)


@pytest.mark.slow
def test_train_epoch(cfg_head_segmentation_train: DictConfig) -> None:
    """Train for a full single epoch.

    :param cfg_head_segmentation_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_head_segmentation_train)
    with open_dict(cfg_head_segmentation_train):
        cfg_head_segmentation_train.trainer.max_epochs = 1
    train(cfg_head_segmentation_train)
