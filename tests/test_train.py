import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.run_if import RunIf


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
@pytest.mark.skip(reason="To fix")
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
@pytest.mark.skip(reason="To fix")
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp"
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 0

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 1

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/loss"] != metric_dict_2["train/loss"]
    assert metric_dict_1["val/loss"] != metric_dict_2["val/loss"]


@pytest.mark.slow
def test_train_find_lr(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.find_lr = True
    train(cfg_train)


@pytest.mark.slow
def test_train_label_smoothing(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.model.criterion.reduction = "none"
        cfg_train.model.criterion.label_smoothing = 0.02
    train(cfg_train)


@pytest.mark.slow
def test_train_softmax_target(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.model.softmax_target = True
    train(cfg_train)


@pytest.mark.slow
def test_train_mix_up(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.callbacks.mix_up = {}
        cfg_train.callbacks.mix_up._target_ = "src.models.utils.callbacks.MixUpCallback"
        cfg_train.callbacks.mix_up.alpha = 0.4
        cfg_train.callbacks.mix_up.softmax_target = False
        cfg_train.callbacks.mix_up.labels = 5
    train(cfg_train)


@pytest.mark.slow
def test_train_mix_up_v2(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.callbacks.mix_up = {}
        cfg_train.callbacks.mix_up._target_ = "src.models.utils.callbacks.MixUpV2Callback"
        cfg_train.callbacks.mix_up.alpha = 0.4
        cfg_train.callbacks.mix_up.softmax_target = False
        cfg_train.callbacks.mix_up.labels = 5
    train(cfg_train)


@pytest.mark.slow
def test_train_vm_mix_up(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.callbacks.mix_up = {}
        cfg_train.callbacks.mix_up._target_ = "src.models.utils.callbacks.VHMixUpCallback"
        cfg_train.callbacks.mix_up.alpha = 0.4
        cfg_train.callbacks.mix_up.softmax_target = False
        cfg_train.callbacks.mix_up.labels = 5
    train(cfg_train)


@pytest.mark.slow
def test_train_mix_up_softmax_target(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.model.softmax_target = True
        cfg_train.callbacks.mix_up = {}
        cfg_train.callbacks.mix_up._target_ = "src.models.utils.callbacks.MixUpV2Callback"
        cfg_train.callbacks.mix_up.alpha = 0.4
        cfg_train.callbacks.mix_up.softmax_target = True
        cfg_train.callbacks.mix_up.labels = 5
    train(cfg_train)


@pytest.mark.slow
def test_train_mix_up_softmax_target_label_smoothing(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.model.criterion.reduction = "none"
        cfg_train.model.criterion.label_smoothing = 0.02
        cfg_train.model.softmax_target = True
        cfg_train.callbacks.mix_up = {}
        cfg_train.callbacks.mix_up._target_ = "src.models.utils.callbacks.MixUpV2Callback"
        cfg_train.callbacks.mix_up.alpha = 0.4
        cfg_train.callbacks.mix_up.softmax_target = True
        cfg_train.callbacks.mix_up.labels = 5
    train(cfg_train)


@pytest.mark.slow
def test_train_class_image_sampler(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # cfg_train.trainer.fast_dev_run = True
        cfg_train.callbacks.class_image_sampler = {}
        cfg_train.callbacks.class_image_sampler._target_ = "src.models.utils.callbacks.ClassImageSampler"
        cfg_train.callbacks.class_image_sampler.class_names = [
            "Trans-thalamic",
            "Trans-cerebellum",
            "Trans-ventricular",
            "Other",
            "Not A Brain",
        ]
    train(cfg_train)
