"""Unit tests for fetal planes sample data loading (offline, no gdown)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as T

from src.data.components.dataset import FetalPlanesSamplesDataset


@pytest.mark.parametrize(
    "train, set_len",
    [
        (True, 32),
        (False, 16),
        (None, 48),
    ],
)
def test_len(data_path: Path, train: bool | None, set_len: int) -> None:
    ds = FetalPlanesSamplesDataset(data_dir=str(data_path), train=train)

    assert len(ds) == set_len


@pytest.mark.parametrize(
    "train, image_shape, expected_label",
    [
        (True, (3, 377, 648), "Not A Brain"),
        (False, (1, 559, 745), "Not A Brain"),
        (None, (3, 377, 648), "Not A Brain"),
    ],
)
def test_getitem(data_path: Path, train: bool | None, image_shape, expected_label: str) -> None:
    ds = FetalPlanesSamplesDataset(data_dir=str(data_path), train=train)

    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == image_shape
    assert label == expected_label


def test_tuple_indexing(data_path: Path) -> None:
    ds = FetalPlanesSamplesDataset(data_dir=str(data_path))

    img_only = ds[0, 0]
    label_only = ds[0, 1]
    image, label = ds[0]

    assert torch.equal(img_only, image)
    assert label_only == label


def test_transform(data_path: Path) -> None:
    transform = T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST)
    ds = FetalPlanesSamplesDataset(data_dir=str(data_path), train=True, transform=transform)

    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert label == "Not A Brain"

    img_only = ds[0, 0]
    label_only = ds[0, 1]

    assert torch.equal(img_only, image)
    assert label_only == label
