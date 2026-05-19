"""Unit tests for HeadSegmentationDataModule (direct instantiation, sample=True)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from src.data.components.dataset import (
    HeadSegmentationDataset,
    HeadSegmentationSamplesDataset,
)
from src.data.head_segmentation import HeadSegmentationDataModule

BATCH_SIZE = 4


@pytest.fixture()
def dm(data_path: Path) -> HeadSegmentationDataModule:
    return HeadSegmentationDataModule(data_dir=str(data_path), sample=True, batch_size=BATCH_SIZE, num_workers=0)


@pytest.fixture()
def dm_setup(dm: HeadSegmentationDataModule) -> HeadSegmentationDataModule:
    dm.prepare_data()
    dm.setup()
    return dm


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    dm = HeadSegmentationDataModule()
    assert dm.hparams.batch_size == 64
    assert dm.hparams.num_workers == 0
    assert tuple(dm.hparams.input_size) == (55, 80)
    assert dm.hparams.sample is False


def test_datasets_not_loaded_before_setup(dm: HeadSegmentationDataModule) -> None:
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None


def test_sample_flag_selects_correct_dataset_class() -> None:
    dm_sample = HeadSegmentationDataModule(sample=True)
    dm_full = HeadSegmentationDataModule(sample=False)

    assert dm_sample.dataset is HeadSegmentationSamplesDataset
    assert dm_full.dataset is HeadSegmentationDataset


def test_num_classes(dm: HeadSegmentationDataModule) -> None:
    assert dm.num_classes == 2


def test_state_dict_is_empty(dm: HeadSegmentationDataModule) -> None:
    assert dm.state_dict() == {}


def test_load_state_dict_noop(dm: HeadSegmentationDataModule) -> None:
    dm.load_state_dict({})


def test_default_transforms_are_compose_instances(dm: HeadSegmentationDataModule) -> None:
    assert isinstance(dm.train_transforms, T.Compose)
    assert isinstance(dm.test_transforms, T.Compose)


# ---------------------------------------------------------------------------
# prepare_data / setup
# ---------------------------------------------------------------------------


def test_prepare_data_is_noop(dm: HeadSegmentationDataModule) -> None:
    dm.prepare_data()
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None


def test_setup_populates_all_splits(dm_setup: HeadSegmentationDataModule) -> None:
    assert dm_setup.data_train is not None
    assert dm_setup.data_val is not None
    assert dm_setup.data_test is not None


def test_setup_idempotent(dm_setup: HeadSegmentationDataModule) -> None:
    train_before = dm_setup.data_train
    val_before = dm_setup.data_val
    test_before = dm_setup.data_test

    dm_setup.setup()

    assert dm_setup.data_train is train_before
    assert dm_setup.data_val is val_before
    assert dm_setup.data_test is test_before


def test_sample_dir_exists_after_setup(data_path: Path, dm_setup: HeadSegmentationDataModule) -> None:
    assert Path(data_path, "FETAL_HEAD_SEGMENTATION_SAMPLES").exists()


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def test_all_dataloaders_created(dm_setup: HeadSegmentationDataModule) -> None:
    assert isinstance(dm_setup.train_dataloader(), DataLoader)
    assert isinstance(dm_setup.val_dataloader(), DataLoader)
    assert isinstance(dm_setup.test_dataloader(), DataLoader)


def test_train_dataloader_shuffles_val_test_do_not(dm_setup: HeadSegmentationDataModule) -> None:
    assert dm_setup.train_dataloader().sampler.__class__.__name__ == "RandomSampler"
    assert dm_setup.val_dataloader().sampler.__class__.__name__ == "SequentialSampler"
    assert dm_setup.test_dataloader().sampler.__class__.__name__ == "SequentialSampler"


@pytest.mark.parametrize("batch_size", [4, 8])
def test_batch_shapes_and_dtypes(data_path: Path, batch_size: int) -> None:
    dm = HeadSegmentationDataModule(data_dir=str(data_path), sample=True, batch_size=batch_size, num_workers=0)
    dm.setup()

    image, mask, label = next(iter(dm.train_dataloader()))

    assert image.shape == (batch_size, 1, 55, 80)
    assert mask.shape == (batch_size, 1, 55, 80)
    assert label.shape == (batch_size,)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert label.dtype == torch.int32


def test_image_values_normalized(dm_setup: HeadSegmentationDataModule) -> None:
    image, _, _ = next(iter(dm_setup.train_dataloader()))
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_mask_binary(dm_setup: HeadSegmentationDataModule) -> None:
    _, mask, _ = next(iter(dm_setup.train_dataloader()))
    unique = mask.unique()
    assert all(v.item() in {0.0, 1.0} for v in unique)
