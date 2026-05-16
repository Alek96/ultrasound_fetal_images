"""Unit tests for head segmentation sample data loading (offline, no gdown)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from src.data.components.dataset import HeadSegmentationSamplesDataset


@pytest.mark.parametrize("subset, set_len", [("train", 24), ("val", 20), ("test", 20), (None, 64)])
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
