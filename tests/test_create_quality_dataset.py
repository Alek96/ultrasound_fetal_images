from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.create_quality_dataset import create_quality_dataset
from src.train import train


@pytest.mark.slow
def test_create_quality_dataset(
    tmp_path: Path,
    cfg_brain_planes_train: DictConfig,
    cfg_create_quality_dataset: DictConfig,
    cfg_video_quality_train: DictConfig,
) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_brain_planes_train: A DictConfig containing a valid training configuration.
    :param cfg_create_quality_dataset: A DictConfig containing a valid create_quality_dataset
        configuration.
    :param cfg_video_quality_train: A DictConfig containing a valid training configuration.
    """
    assert str(tmp_path) == cfg_brain_planes_train.paths.output_dir
    assert str(tmp_path) == cfg_create_quality_dataset.paths.output_dir
    assert str(tmp_path) == cfg_video_quality_train.paths.output_dir

    with open_dict(cfg_brain_planes_train):
        cfg_brain_planes_train.trainer.max_epochs = 0
        cfg_brain_planes_train.test = True

    HydraConfig().set_config(cfg_brain_planes_train)
    train_metric_dict, _ = train(cfg_brain_planes_train)

    with open_dict(cfg_create_quality_dataset):
        cfg_create_quality_dataset.model_path = str(tmp_path)

    HydraConfig().set_config(cfg_create_quality_dataset)
    create_quality_dataset(cfg_create_quality_dataset)

    with open_dict(cfg_video_quality_train):
        cfg_video_quality_train.trainer.max_epochs = 1
        cfg_video_quality_train.test = True

    HydraConfig().set_config(cfg_video_quality_train)
    train_metric_dict, _ = train(cfg_video_quality_train)
