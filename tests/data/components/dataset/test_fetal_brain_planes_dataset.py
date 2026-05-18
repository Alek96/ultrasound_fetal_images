"""Unit tests for fetal brain planes sample data loading (offline, no gdown)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
import torchvision.transforms.v2 as T

from src.data.components.dataset import FetalBrainPlanesSamplesDataset


@pytest.mark.parametrize(
    "subset, set_len",
    [
        ("train", 16),
        ("val", 16),
        ("test", 16),
        (None, 48),
    ],
)
def test_len(data_path: Path, subset: Literal["train", "val", "test"] | None, set_len: int) -> None:
    ds = FetalBrainPlanesSamplesDataset(data_dir=str(data_path), subset=subset)

    assert len(ds) == set_len


@pytest.mark.parametrize(
    "subset, image_shape, expected_label",
    [
        ("train", (3, 377, 648), "Not A Brain"),
        ("val", (1, 381, 647), "Not A Brain"),
        ("test", (1, 559, 745), "Not A Brain"),
        (None, (3, 377, 648), "Not A Brain"),
    ],
)
def test_getitem(
    data_path: Path, subset: Literal["train", "val", "test"] | None, image_shape, expected_label: str
) -> None:
    ds = FetalBrainPlanesSamplesDataset(data_dir=str(data_path), subset=subset)

    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == image_shape
    assert label == expected_label


def test_tuple_indexing(data_path: Path) -> None:
    ds = FetalBrainPlanesSamplesDataset(data_dir=str(data_path))

    img_only = ds[0, 0]
    label_only = ds[0, 1]
    image, label = ds[0]

    assert torch.equal(img_only, image)
    assert label_only == label


def test_transform(data_path: Path) -> None:
    transform = T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST)
    ds = FetalBrainPlanesSamplesDataset(data_dir=str(data_path), subset="train", transform=transform)

    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert label == "Not A Brain"

    img_only = ds[0, 0]
    label_only = ds[0, 1]

    assert torch.equal(img_only, image)
    assert label_only == label
