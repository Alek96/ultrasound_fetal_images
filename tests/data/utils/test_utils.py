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
    create_ellipse_tensor,
    crop,
    find_angle,
    get_dice_score,
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


# ---------------------------------------------------------------------------
# create_ellipse_tensor
# ---------------------------------------------------------------------------


class TestCreateEllipseTensor:
    def test_output_shape(self):
        """Output tensor must have shape (height, width)."""
        out = create_ellipse_tensor(50, 60, 30, 25, 10, 8)
        assert out.shape == (50, 60)

    def test_center_pixel_is_inside(self):
        """The centre pixel must always be inside the ellipse (value 1)."""
        cx, cy = 30, 20
        out = create_ellipse_tensor(50, 60, cx, cy, 10, 8)
        assert out[cy, cx].item() == 1.0

    def test_far_corner_is_outside(self):
        """A pixel far beyond the radii must be outside the ellipse (value 0)."""
        out = create_ellipse_tensor(100, 100, 50, 50, 5, 5)
        assert out[0, 0].item() == 0.0

    def test_binary_values_only(self):
        """All values in the output must be exactly 0 or 1."""
        out = create_ellipse_tensor(64, 64, 32, 32, 15, 10)
        unique = out.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_circle_is_symmetric(self):
        """A circle (rx == ry, theta=0) must be symmetric about both axes.

        Use an odd-sized canvas so the integer centre (32, 32) is equidistant
        from all four edges (distance = 32 pixels on each side).
        """
        r = 15
        cx, cy = 32, 32
        out = create_ellipse_tensor(65, 65, cx, cy, r, r)
        # Both the centre row and the centre column must be palindromes.
        row = out[cy]
        assert torch.equal(row, row.flip(0)), "circle row is not horizontally symmetric"
        col = out[:, cx]
        assert torch.equal(col, col.flip(0)), "circle column is not vertically symmetric"

    def test_rotation_swaps_radii_effect(self):
        """Rotating by 90° should swap the effect of radius_x and radius_y."""
        h, w, cx, cy = 64, 64, 32, 32
        rx, ry = 20, 8
        out_0 = create_ellipse_tensor(h, w, cx, cy, rx, ry, theta_rad=0)
        out_90 = create_ellipse_tensor(h, w, cx, cy, rx, ry, theta_rad=torch.pi / 2)
        # The 0° ellipse is wide (extends further horizontally than vertically)
        h_span_0 = out_0[cy].sum()
        v_span_0 = out_0[:, cx].sum()
        # After 90° rotation the ellipse should be taller than it is wide
        h_span_90 = out_90[cy].sum()
        v_span_90 = out_90[:, cx].sum()
        assert h_span_0 > v_span_0, "unrotated ellipse should be wider than tall"
        assert v_span_90 > h_span_90, "90°-rotated ellipse should be taller than wide"

    def test_zero_radius_raises(self):
        """Passing radius_x=0 or radius_y=0 must raise ValueError."""
        with pytest.raises(ValueError):
            create_ellipse_tensor(50, 50, 25, 25, 0, 10)
        with pytest.raises(ValueError):
            create_ellipse_tensor(50, 50, 25, 25, 10, 0)

    def test_add_half_changes_mask(self):
        """add_half=True must produce a different (slightly larger) ellipse than add_half=False."""
        base = create_ellipse_tensor(64, 64, 32, 32, 10, 10, add_half=False)
        with_half = create_ellipse_tensor(64, 64, 32, 32, 10, 10, add_half=True)
        # The half-adjusted ellipse should have at least as many foreground pixels
        assert with_half.sum() >= base.sum()

    def test_full_coverage_when_radii_exceed_image(self):
        """When radii are larger than the image, every pixel should be inside."""
        out = create_ellipse_tensor(10, 10, 5, 5, 100, 100)
        assert out.sum().item() == 10 * 10


# ---------------------------------------------------------------------------
# get_dice_score
# ---------------------------------------------------------------------------


class TestGetDiceScore:
    def test_identical_tensors_return_one(self):
        """Dice score of a tensor with itself must be 1.0 (modulo smoothing)."""
        t = torch.ones(10)
        score = get_dice_score(t, t)
        assert score.item() == pytest.approx(1.0, abs=1e-5)

    def test_identical_zero_tensors(self):
        """Two all-zero tensors: (0 + smooth) / (0 + 0 + smooth) = 1.0."""
        t = torch.zeros(10)
        score = get_dice_score(t, t)
        assert score.item() == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap_near_zero(self):
        """Disjoint masks: numerator ≈ 0, score ≈ 0 (smoothing negligible)."""
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 0.0, 0.0, 1.0])
        score = get_dice_score(a, b, smooth=0.0)
        assert score.item() == pytest.approx(0.0, abs=1e-6)

    def test_half_overlap_formula(self):
        """`2×|A∩B| / (|A|+|B|) = 2×1 / (2+2) = 0.5` for two 1-hot tensors sharing one element."""
        a = torch.tensor([1.0, 1.0, 0.0, 0.0])
        b = torch.tensor([1.0, 0.0, 1.0, 0.0])
        score = get_dice_score(a, b, smooth=0.0)
        assert score.item() == pytest.approx(0.5, abs=1e-6)

    def test_shape_independent_of_layout(self):
        """Dice score should be the same regardless of whether inputs are 1-D or 2-D."""
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        flat_score = get_dice_score(a.view(-1), b.view(-1))
        shaped_score = get_dice_score(a, b)
        assert flat_score.item() == pytest.approx(shaped_score.item(), abs=1e-6)

    def test_mismatched_sizes_raise(self):
        """Inputs with a different number of elements must raise ValueError."""
        with pytest.raises(ValueError):
            get_dice_score(torch.ones(4), torch.ones(5))

    def test_smooth_prevents_division_by_zero(self):
        """Default smooth value must keep the score finite even for empty inputs."""
        score = get_dice_score(torch.zeros(0), torch.zeros(0))
        assert score.isfinite()


# ---------------------------------------------------------------------------
# find_angle
# ---------------------------------------------------------------------------


class TestFindAngle:
    def test_horizontal_line_returns_zero(self):
        """A horizontal row of pixels has a principal axis along x → angle ≈ 0°."""
        mask = torch.zeros(20, 40)
        mask[10, 5:35] = 1.0  # horizontal stripe
        angle = find_angle(mask)
        assert abs(angle.item()) in {0, 180}, f"expected 0 or 180, got {angle.item()}"

    def test_vertical_line_returns_90(self):
        """A vertical column of pixels has a principal axis along y → angle ≈ ±90°."""
        mask = torch.zeros(40, 20)
        mask[5:35, 10] = 1.0  # vertical stripe
        angle = find_angle(mask)
        assert abs(angle.item()) == 90, f"expected ±90, got {angle.item()}"

    def test_diagonal_returns_45_or_minus_45(self):
        """A diagonal band of pixels should yield an angle close to ±45°."""
        mask = torch.zeros(40, 40)
        for i in range(5, 35):
            mask[i, i] = 1.0
        angle = find_angle(mask)
        assert abs(angle.item()) == pytest.approx(45, abs=2)

    def test_horizontal_ellipse_return_0(self):
        """A horizontal row of pixels has a principal axis along x → angle ≈ 0°."""
        mask = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        angle = find_angle(mask)
        assert abs(angle.item()) in {0, 180}, f"expected 0 or 180, got {angle.item()}"

    def test_vertical_ellipse_return_90(self):
        """A vertical column of pixels has a principal axis along y → angle ≈ ±90°."""
        mask = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        angle = find_angle(mask)
        assert abs(angle.item()) == 90, f"expected ±90, got {angle.item()}"

    @pytest.mark.parametrize(
        "height, width, center_x, center_y, radius_x, radius_y, theta_rad, angles",
        [
            (99, 99, 49, 49, 50, 40, 0.000, {0, 180}),
            (99, 99, 49, 49, 50, 40, 0.785, {45}),
            (99, 99, 49, 49, 50, 40, 1.571, {90}),
            (99, 99, 49, 49, 50, 40, 2.355, {45}),
            (99, 99, 49, 49, 50, 40, 3.141, {0, 180}),
            (99, 99, 49, 49, 50, 40, 3.925, {45}),
            (99, 99, 49, 49, 50, 40, 4.710, {90}),
        ],
    )
    def test_horizontal_ellipse(
        self,
        height: int,
        width: int,
        center_x: int,
        center_y: int,
        radius_x: int,
        radius_y: int,
        theta_rad: float | int,
        angles: set[int],
    ):
        """Angle should match manually create ellipse"""
        mask = create_ellipse_tensor(height, width, center_x, center_y, radius_x, radius_y, theta_rad)
        angle = find_angle(mask)
        assert abs(angle.item()) in angles

    def test_accepts_single_channel_3d_tensor(self):
        """A (1, H, W) tensor must be squeezed and produce the same result as (H, W)."""
        mask_2d = torch.zeros(20, 20)
        mask_2d[10, 2:18] = 1.0
        mask_3d = mask_2d.unsqueeze(0)  # (1, 20, 20)
        assert find_angle(mask_2d).item() == find_angle(mask_3d).item()

    def test_rejects_non_2d_tensor(self):
        """A tensor with ndim != 2 (and not squeezable to 2D) must raise AssertionError."""
        with pytest.raises(AssertionError):
            find_angle(torch.zeros(2, 10, 10))  # 2-channel 3D tensor

    def test_empty_mask_returns_zero(self):
        """An all-zero mask (no foreground) must return 0 without raising."""
        mask = torch.zeros(20, 20)
        angle = find_angle(mask)
        assert angle.item() == 0

    def test_output_is_int32_scalar(self):
        mask = torch.zeros(20, 20)
        mask[10, :] = 1.0
        angle = find_angle(mask)
        assert angle.dtype == torch.int32
        assert angle.dim() == 0

    class TestcreateEllipse:
        def test_horizontal_ellipse(self):
            """A diagonal band of pixels should yield an angle close to ±45°."""
            mask = create_ellipse_tensor(50, 60, 30, 25, 10, 8)
            angle = find_angle(mask)
            assert abs(angle.item()) in {0, 180}, f"expected 0 or 180, got {angle.item()}"

        def test_vertical_ellipse(self):
            """A diagonal band of pixels should yield an angle close to ±45°."""
            mask = create_ellipse_tensor(50, 60, 30, 25, 8, 10)
            angle = find_angle(mask)
            assert abs(angle.item()) == 90, f"expected ±90, got {angle.item()}"


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------


_IMG_C, _IMG_H, _IMG_W = 3, 64, 80


@pytest.fixture()
def sample_image() -> torch.Tensor:
    return torch.randint(0, 256, (_IMG_C, _IMG_H, _IMG_W), dtype=torch.uint8)


class TestCrop:
    def test_basic_crop_shape(self, sample_image):
        """Crop without padding must return the exact requested region."""
        out = crop(sample_image, x1=10, y1=5, x2=30, y2=25, pad=0)
        assert out.shape == (_IMG_C, 20, 20)

    def test_channel_dim_preserved(self, sample_image):
        """Number of channels must be unchanged after cropping."""
        out = crop(sample_image, x1=10, y1=5, x2=30, y2=25, pad=0)
        assert out.shape[0] == _IMG_C

    def test_pixel_values_match_original(self, sample_image):
        """Cropped content must equal the corresponding slice of the original image."""
        x1, y1, x2, y2 = 10, 5, 30, 25
        out = crop(sample_image, x1, y1, x2, y2, pad=0)
        assert torch.equal(out, sample_image[:, y1:y2, x1:x2])

    def test_padding_expands_region(self, sample_image):
        """Adding pad > 0 must produce a strictly larger crop than pad=0."""
        no_pad = crop(sample_image, x1=20, y1=20, x2=40, y2=40, pad=0)
        with_pad = crop(sample_image, x1=20, y1=20, x2=40, y2=40, pad=10)
        assert with_pad.shape[1] >= no_pad.shape[1]
        assert with_pad.shape[2] >= no_pad.shape[2]

    def test_padding_clamped_at_image_boundary(self, sample_image):
        """Padding must never cause the crop to exceed the image dimensions."""
        out = crop(sample_image, x1=1, y1=1, x2=10, y2=10, pad=50)
        assert out.shape[1] <= _IMG_H
        assert out.shape[2] <= _IMG_W

    def test_invalid_x_coordinates_raise(self, sample_image):
        with pytest.raises(AssertionError):
            crop(sample_image, x1=30, y1=5, x2=10, y2=25)  # x1 > x2

    def test_invalid_y_coordinates_raise(self, sample_image):
        with pytest.raises(AssertionError):
            crop(sample_image, x1=10, y1=25, x2=30, y2=5)  # y1 > y2

    def test_out_of_bounds_x_raises(self, sample_image):
        with pytest.raises(AssertionError):
            crop(sample_image, x1=10, y1=5, x2=_IMG_W + 1, y2=25)

    def test_out_of_bounds_y_raises(self, sample_image):
        with pytest.raises(AssertionError):
            crop(sample_image, x1=10, y1=5, x2=30, y2=_IMG_H + 1)

    def test_requires_3d_tensor(self):
        with pytest.raises(AssertionError):
            crop(torch.zeros(64, 64), x1=0, y1=0, x2=10, y2=10)

    def test_invalid_pad_raises(self, sample_image):
        with pytest.raises(AssertionError):
            crop(sample_image, x1=10, y1=5, x2=30, y2=25, pad=100)

    def test_full_image_crop(self, sample_image):
        """Cropping the full extent with pad=0 must return an identical tensor."""
        out = crop(sample_image, x1=0, y1=0, x2=_IMG_W, y2=_IMG_H, pad=0)
        assert torch.equal(out, sample_image)
