"""Unit tests for BrainPlanesDataModule (direct instantiation, sample=True)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from src.data.brain_planes import BrainPlanesDataModule
from src.data.components.dataset import (
    FetalBrainPlanesDataset,
    FetalBrainPlanesSamplesDataset,
)

BATCH_SIZE = 4


@pytest.fixture()
def dm(data_path: Path) -> BrainPlanesDataModule:
    return BrainPlanesDataModule(data_dir=str(data_path), sample=True, batch_size=BATCH_SIZE, num_workers=0)


@pytest.fixture()
def dm_setup(dm: BrainPlanesDataModule) -> BrainPlanesDataModule:
    dm.prepare_data()
    dm.setup()
    return dm


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    dm = BrainPlanesDataModule()
    assert dm.hparams.batch_size == 64
    assert dm.hparams.num_workers == 0
    assert tuple(dm.hparams.input_size) == (55, 80)
    assert dm.hparams.sample is False
    assert dm.hparams.sampler is None


def test_datasets_not_loaded_before_setup(dm: BrainPlanesDataModule) -> None:
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None


def test_sample_flag_selects_correct_dataset_class() -> None:
    dm_sample = BrainPlanesDataModule(sample=True)
    dm_full = BrainPlanesDataModule(sample=False)

    assert dm_sample.dataset is FetalBrainPlanesSamplesDataset
    assert dm_full.dataset is FetalBrainPlanesDataset


def test_num_classes(dm: BrainPlanesDataModule) -> None:
    assert dm.num_classes == 4


def test_num_classes_matches_labels(dm: BrainPlanesDataModule) -> None:
    assert dm.num_classes == len(dm.labels)


def test_state_dict_is_empty(dm: BrainPlanesDataModule) -> None:
    assert dm.state_dict() == {}


def test_load_state_dict_noop(dm: BrainPlanesDataModule) -> None:
    dm.load_state_dict({})


def test_default_transforms_are_compose_instances(dm: BrainPlanesDataModule) -> None:
    assert isinstance(dm.train_transforms, T.Compose)
    assert isinstance(dm.test_transforms, T.Compose)


# ---------------------------------------------------------------------------
# prepare_data / setup
# ---------------------------------------------------------------------------


def test_prepare_data_is_noop(dm: BrainPlanesDataModule) -> None:
    dm.prepare_data()
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None


def test_setup_populates_all_splits(dm_setup: BrainPlanesDataModule) -> None:
    assert dm_setup.data_train is not None
    assert dm_setup.data_val is not None
    assert dm_setup.data_test is not None


def test_setup_idempotent(dm_setup: BrainPlanesDataModule) -> None:
    train_before = dm_setup.data_train
    val_before = dm_setup.data_val
    test_before = dm_setup.data_test

    dm_setup.setup()

    assert dm_setup.data_train is train_before
    assert dm_setup.data_val is val_before
    assert dm_setup.data_test is test_before


def test_sample_dir_exists_after_setup(data_path: Path, dm_setup: BrainPlanesDataModule) -> None:
    assert Path(data_path, "FETAL_BRAIN_PLANES_SAMPLES").exists()


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def test_all_dataloaders_created(dm_setup: BrainPlanesDataModule) -> None:
    assert isinstance(dm_setup.train_dataloader(), DataLoader)
    assert isinstance(dm_setup.val_dataloader(), DataLoader)
    assert isinstance(dm_setup.test_dataloader(), DataLoader)


def test_dataloader_batch_sizes(dm_setup: BrainPlanesDataModule) -> None:
    """val uses batch_size*2 and test uses batch_size*3 per the implementation."""
    assert dm_setup.val_dataloader().batch_size == BATCH_SIZE * 2
    assert dm_setup.test_dataloader().batch_size == BATCH_SIZE * 3


def test_train_dataloader_shuffles_val_test_do_not(dm_setup: BrainPlanesDataModule) -> None:
    assert dm_setup.train_dataloader().sampler.__class__.__name__ == "RandomSampler"
    assert dm_setup.val_dataloader().sampler.__class__.__name__ == "SequentialSampler"
    assert dm_setup.test_dataloader().sampler.__class__.__name__ == "SequentialSampler"


@pytest.mark.parametrize("batch_size", [4, 8])
def test_batch_shapes_and_dtypes(data_path: Path, batch_size: int) -> None:
    dm = BrainPlanesDataModule(data_dir=str(data_path), sample=True, batch_size=batch_size, num_workers=0)
    dm.setup()

    image, label = next(iter(dm.train_dataloader()))

    assert image.shape == (batch_size, 1, 55, 80)
    assert image.dtype == torch.float32
    assert label.shape == (batch_size,)
    assert label.dtype == torch.int64


def test_image_values_normalized(dm_setup: BrainPlanesDataModule) -> None:
    image, _ = next(iter(dm_setup.train_dataloader()))
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_labels_are_valid_class_indices(dm_setup: BrainPlanesDataModule) -> None:
    _, label = next(iter(dm_setup.train_dataloader()))
    assert label.min() >= 0
    assert label.max() < dm_setup.num_classes
