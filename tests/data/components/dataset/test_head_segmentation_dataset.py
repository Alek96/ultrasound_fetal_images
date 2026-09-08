"""Unit tests for head segmentation sample data loading (offline, no gdown)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from src.data.components.dataset import HeadSegmentationSamplesDataset


@pytest.mark.parametrize(
    "subset, set_len",
    [
        ("train", 24),
        ("val", 20),
        ("test", 20),
        (None, 64),
    ],
)
def test_len(data_path: Path, subset: Literal["train", "val", "test"] | None, set_len: int) -> None:
    ds = HeadSegmentationSamplesDataset(data_dir=str(data_path), subset=subset)

    assert len(ds) == set_len


@pytest.mark.parametrize(
    "subset, image_shape, mask_shape, expected_label",
    [
        ("train", (3, 661, 959), (1, 661, 959), 1),
        ("val", (1, 381, 647), (1, 381, 647), 0),
        ("test", (1, 559, 745), (1, 559, 745), 0),
        (None, (3, 661, 959), (1, 661, 959), 1),
    ],
)
def test_getitem(
    data_path: Path, subset: Literal["train", "val", "test"] | None, image_shape, mask_shape, expected_label
) -> None:
    ds = HeadSegmentationSamplesDataset(data_dir=str(data_path), subset=subset)

    image, mask, label = ds[0]
    assert isinstance(image, tv_tensors.Image)
    assert isinstance(mask, tv_tensors.Mask)
    assert image.shape == image_shape
    assert mask.shape == mask_shape
    assert label.shape == ()
    assert int(label) == expected_label


def test_tuple_indexing(data_path: Path) -> None:
    ds = HeadSegmentationSamplesDataset(data_dir=str(data_path))

    img_only = ds[0, 0]
    mask_only = ds[0, 1]
    label_only = ds[0, 2]
    image, mask, label = ds[0]

    assert torch.equal(img_only, image)
    assert torch.equal(mask_only, mask)
    assert torch.equal(label_only, label)


def test_transform(data_path: Path) -> None:
    transform = T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST)
    ds = HeadSegmentationSamplesDataset(data_dir=str(data_path), transform=transform)

    image, mask, label = ds[0]
    assert isinstance(image, tv_tensors.Image)
    assert isinstance(mask, tv_tensors.Mask)
    assert image.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert label.shape == ()
    assert int(label) == 1

    img_only = ds[0, 0]
    mask_only = ds[0, 1]
    label_only = ds[0, 2]

    assert torch.equal(img_only, image)
    assert torch.equal(mask_only, mask)
    assert torch.equal(label_only, label)


def test_get_image_iterator(data_path: Path) -> None:
    ds = HeadSegmentationSamplesDataset(data_dir=str(data_path), subset="train")

    it = ds.get_image_iterator()
    images = list(it)

    assert len(images) == len(ds)
    assert isinstance(images[0], tv_tensors.Image)
    assert images[0].shape == (3, 661, 959)


class TestMissingSegmentationPath:
    """A row without a mask is only valid for negative (non-brain) samples."""

    @staticmethod
    def _dataset(data_path: Path, tmp_path: Path, brain_plane: int) -> HeadSegmentationSamplesDataset:
        """Copy the sample manifest into tmp_path and drop the first row's mask."""
        dataset_name = "FETAL_HEAD_SEGMENTATION_SAMPLES"
        source_dir = data_path / dataset_name
        target_dir = tmp_path / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)

        labels = pd.read_csv(source_dir / "data.csv", dtype={"Patient_num": str})
        labels = labels.head(1).copy()
        labels.loc[0, "Segmentation_path"] = pd.NA
        labels.loc[0, "Brain_plane"] = brain_plane
        # Point at the real image so only the mask lookup is exercised.
        labels.loc[0, "Ultrasound_path"] = str(source_dir / labels.loc[0, "Ultrasound_path"])
        labels.to_csv(target_dir / "data.csv", index=False)

        return HeadSegmentationSamplesDataset(data_dir=str(tmp_path))

    def test_positive_row_without_mask_raises(self, data_path: Path, tmp_path: Path) -> None:
        ds = self._dataset(data_path, tmp_path, brain_plane=1)

        with pytest.raises(ValueError, match="Missing 'Segmentation_path'"):
            ds.get_mask(0)

    def test_negative_row_without_mask_returns_empty_mask(self, data_path: Path, tmp_path: Path) -> None:
        ds = self._dataset(data_path, tmp_path, brain_plane=0)

        mask = ds.get_mask(0)

        assert isinstance(mask, tv_tensors.Mask)
        assert mask.shape == (1, 661, 959)
        assert int(mask.sum()) == 0
