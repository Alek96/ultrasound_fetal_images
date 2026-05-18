"""Unit tests for src.data.utils.utils."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import PIL.Image
import pytest
import torch
from matplotlib.figure import Figure
from torch.utils.data import TensorDataset

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from src.data.components.dataset import Subset
from src.data.utils.utils import (
    group_split,
    read_image_tensor,
    save_image_tensor,
    show_numpy_images,
    show_pytorch_images,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gray_png(tmp_path: Path) -> Path:
    arr = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
    path = tmp_path / "gray.png"
    PIL.Image.fromarray(arr, mode="L").save(path)
    return path


@pytest.fixture()
def rgb_png(tmp_path: Path) -> Path:
    arr = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    PIL.Image.fromarray(arr, mode="RGB").save(path)
    return path


@pytest.fixture()
def rgba_png(tmp_path: Path) -> Path:
    arr = np.random.randint(0, 256, (16, 16, 4), dtype=np.uint8)
    path = tmp_path / "rgba.png"
    PIL.Image.fromarray(arr, mode="RGBA").save(path)
    return path


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# read_image_tensor
# ---------------------------------------------------------------------------


class TestReadImageTensor:
    def test_grayscale_is_expanded_to_three_channels(self, gray_png: Path):
        """Single-channel images must be repeated to produce 3-channel output."""
        tensor = read_image_tensor(gray_png)
        assert tensor.shape[0] == 3
        # All three channels must contain the same data (repeat, not blend).
        assert torch.equal(tensor[0], tensor[1])
        assert torch.equal(tensor[1], tensor[2])

    def test_rgb_image_has_three_channels(self, rgb_png: Path):
        tensor = read_image_tensor(rgb_png)
        assert tensor.shape[0] == 3

    def test_rgba_is_trimmed_to_three_channels(self, rgba_png: Path):
        """Four-channel images must be trimmed to the first three channels."""
        tensor = read_image_tensor(rgba_png)
        assert tensor.shape[0] == 3

    def test_output_is_tensor(self, rgb_png: Path):
        assert isinstance(read_image_tensor(rgb_png), torch.Tensor)

    def test_output_spatial_dims_match_file(self, rgb_png: Path):
        tensor = read_image_tensor(rgb_png)
        assert tensor.shape[1] == 16
        assert tensor.shape[2] == 16


# ---------------------------------------------------------------------------
# save_image_tensor
# ---------------------------------------------------------------------------


class TestSaveImageTensor:
    def test_saves_file_to_disk(self, tmp_path: Path):
        img = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
        out = tmp_path / "out.png"
        save_image_tensor(img, out)
        assert out.exists()

    def test_round_trip_preserves_pixel_values(self, tmp_path: Path):
        img = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
        out = tmp_path / "round_trip.png"
        save_image_tensor(img, out)
        loaded = read_image_tensor(out)
        assert torch.equal(img, loaded)

    def test_returns_pil_image(self, tmp_path: Path):
        img = torch.zeros(3, 8, 8, dtype=torch.uint8)
        result = save_image_tensor(img, tmp_path / "pil.png")
        assert isinstance(result, PIL.Image.Image)


# ---------------------------------------------------------------------------
# group_split
# ---------------------------------------------------------------------------


class TestGroupSplit:
    N_GROUPS = 4
    N_PER_GROUP = 5
    N = N_GROUPS * N_PER_GROUP

    @pytest.fixture()
    def dataset_and_groups(self):
        dataset = TensorDataset(torch.arange(self.N))
        groups = pd.Series([f"g{g}" for g in range(self.N_GROUPS) for _ in range(self.N_PER_GROUP)])
        return dataset, groups

    def test_returns_two_subsets(self, dataset_and_groups):
        dataset, groups = dataset_and_groups
        train, test = group_split(dataset, test_size=0.25, groups=groups, random_state=0)
        assert isinstance(train, Subset)
        assert isinstance(test, Subset)

    def test_split_covers_all_samples(self, dataset_and_groups):
        dataset, groups = dataset_and_groups
        train, test = group_split(dataset, test_size=0.25, groups=groups, random_state=0)
        assert len(train) + len(test) == self.N

    def test_test_size_approximately_correct(self, dataset_and_groups):
        dataset, groups = dataset_and_groups
        _, test = group_split(dataset, test_size=0.25, groups=groups, random_state=0)
        # With 4 groups the split must land on whole-group boundaries: 1 or 2 groups in test.
        assert self.N_PER_GROUP <= len(test) <= 2 * self.N_PER_GROUP

    def test_no_group_appears_in_both_splits(self, dataset_and_groups):
        """GroupShuffleSplit must never put the same group in both train and test."""
        dataset, groups = dataset_and_groups
        train, test = group_split(dataset, test_size=0.25, groups=groups, random_state=0)
        train_groups = set(groups[list(train.indices)])
        test_groups = set(groups[list(test.indices)])
        assert len(train_groups & test_groups) == 0

    def test_random_state_gives_reproducible_splits(self, dataset_and_groups):
        dataset, groups = dataset_and_groups
        train_a, _ = group_split(dataset, test_size=0.5, groups=groups, random_state=99)
        train_b, _ = group_split(dataset, test_size=0.5, groups=groups, random_state=99)
        assert list(train_a.indices) == list(train_b.indices)


# ---------------------------------------------------------------------------
# show_pytorch_images
# ---------------------------------------------------------------------------


class TestShowPytorchImages:
    def _make_images(self, n: int = 4) -> list[tuple[torch.Tensor, str]]:
        return [(torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8), f"img{i}") for i in range(n)]

    def test_returns_figure(self):
        fig = show_pytorch_images(self._make_images())
        assert isinstance(fig, Figure)

    def test_gray_mode(self):
        fig = show_pytorch_images(self._make_images(), gray=True)
        assert isinstance(fig, Figure)

    def test_color_mode(self):
        fig = show_pytorch_images(self._make_images(), gray=False)
        assert isinstance(fig, Figure)

    def test_with_title_and_ylabel(self):
        fig = show_pytorch_images(self._make_images(), title="My title", ylabel="My ylabel")
        assert isinstance(fig, Figure)

    def test_with_col_and_row_names(self):
        images = self._make_images(n=4)
        fig = show_pytorch_images(
            images,
            cols_names=["c0", "c1"],
            rows_names=["r0", "r1"],
        )
        assert isinstance(fig, Figure)

    def test_with_tick_labels_enabled(self):
        fig = show_pytorch_images(self._make_images(), tick_labels=True)
        assert isinstance(fig, Figure)

    def test_single_image(self):
        fig = show_pytorch_images(self._make_images(n=1))
        assert isinstance(fig, Figure)

    def test_image_without_label(self):
        images = [torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)]
        fig = show_pytorch_images(images)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# show_numpy_images
# ---------------------------------------------------------------------------


class TestShowNumpyImages:
    def _make_images(self, n: int = 4, color: bool = True) -> list[tuple[np.ndarray, str]]:
        shape = (16, 16, 3) if color else (16, 16)
        return [(np.random.randint(0, 256, shape, dtype=np.uint8), f"img{i}") for i in range(n)]

    def test_returns_figure(self):
        """show_numpy_images must return a Figure, not call plt.show() (regression guard)."""
        fig = show_numpy_images(self._make_images())
        assert isinstance(fig, Figure)

    def test_bgr_images(self):
        fig = show_numpy_images(self._make_images(color=True))
        assert isinstance(fig, Figure)

    def test_grayscale_images(self):
        fig = show_numpy_images(self._make_images(color=False))
        assert isinstance(fig, Figure)

    def test_single_image(self):
        fig = show_numpy_images(self._make_images(n=1))
        assert isinstance(fig, Figure)
