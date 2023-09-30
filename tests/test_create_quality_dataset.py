from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.create_quality_dataset import create_quality_dataset
from src.train import train


@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_create_quality_dataset: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_create_quality_dataset: A DictConfig containing a valid create_quality_dataset
        configuration.
    """

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 0
        cfg_train.test = False

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    with open_dict(cfg_create_quality_dataset):
        cfg_create_quality_dataset.model_path = str(tmp_path)

    HydraConfig().set_config(cfg_create_quality_dataset)
    create_quality_dataset(cfg_create_quality_dataset)
