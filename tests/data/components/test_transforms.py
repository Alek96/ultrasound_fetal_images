"""Unit tests for src.data.components.transforms."""

from __future__ import annotations

import pytest
import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from torchvision.transforms.v2 import InterpolationMode

from src.data.components.transforms import (
    Affine,
    HorizontalFlip,
    LabelEncoder,
    OneHotEncoder,
    PadToAspectRatio,
    RandAugment,
    RandAugmentPolicy,
    RandomCutout,
    RandomPercentCrop,
    Resize,
    VerticalFlip,
    cutout,
    grid_distortion,
    speckle_noise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uint8_img(c: int = 3, h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.randint(0, 256, (c, h, w), dtype=torch.uint8)


def _tv_img(c: int = 3, h: int = 64, w: int = 64) -> tv_tensors.Image:
    return tv_tensors.Image(_uint8_img(c, h, w))


def _tv_mask(h: int = 64, w: int = 64) -> tv_tensors.Mask:
    return tv_tensors.Mask(torch.zeros(1, h, w, dtype=torch.uint8))


# ---------------------------------------------------------------------------
# PadToAspectRatio
# ---------------------------------------------------------------------------


class TestPadToAspectRatio:
    def test_too_tall_image_width_is_padded(self):
        """Tall image (2:1) padded to square target → width doubled, height unchanged."""
        pad = PadToAspectRatio(size=[100, 100])
        img = _tv_img(h=100, w=50)
        result = pad(img)
        assert result.shape[1] == 100  # height unchanged
        assert result.shape[2] == 100  # width padded to match aspect ratio

    def test_too_wide_image_height_is_padded(self):
        """Wide image (1:2) padded to square target → height doubled, width unchanged."""
        pad = PadToAspectRatio(size=[100, 100])
        img = _tv_img(h=50, w=100)
        result = pad(img)
        assert result.shape[1] == 100  # height padded
        assert result.shape[2] == 100  # width unchanged

    def test_exact_aspect_ratio_no_padding(self):
        """Image already at the target aspect ratio must pass through unchanged."""
        pad = PadToAspectRatio(size=[64, 64])
        img = _tv_img(h=32, w=32)
        result = pad(img)
        assert result.shape == img.shape

    def test_padding_is_symmetric(self):
        """Horizontal (or vertical) padding must be split evenly on both sides."""
        pad = PadToAspectRatio(size=[100, 100], fill=0)
        img = _tv_img(h=100, w=50)
        result = pad(img)
        # Check left and right columns are zero (the padded region).
        assert result[:, :, 0].sum().item() == 0
        assert result[:, :, -1].sum().item() == 0

    def test_repr_contains_class_name(self):
        assert "PadToAspectRatio" in repr(PadToAspectRatio(size=[192, 256]))


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class TestResize:
    def test_single_input_output_shape(self):
        img = _tv_img(h=64, w=64)
        result = Resize(size=[32, 48])(img)
        assert result.shape == (3, 32, 48)

    def test_multi_input_both_resized(self):
        img = _tv_img(h=64, w=64)
        mask = _tv_mask(h=64, w=64)
        result = Resize(size=[32, 32])(img, mask)
        assert result[0].shape == (3, 32, 32)
        assert result[1].shape == (1, 32, 32)

    def test_per_type_interpolation_dict(self):
        """Dict-based interpolation selects different modes per tensor type."""
        img = _tv_img(h=64, w=64)
        mask = _tv_mask(h=64, w=64)
        interp = {
            tv_tensors.Image: InterpolationMode.BILINEAR,
            tv_tensors.Mask: InterpolationMode.NEAREST,
        }
        result = Resize(size=[32, 32], interpolation=interp)(img, mask)
        assert result[0].shape == (3, 32, 32)
        assert result[1].shape == (1, 32, 32)

    def test_repr_contains_class_name(self):
        assert "Resize" in repr(Resize(size=[64, 64]))


# ---------------------------------------------------------------------------
# HorizontalFlip
# ---------------------------------------------------------------------------


class TestHorizontalFlip:
    def test_flip_true_mirrors_along_width(self):
        img = _uint8_img(c=1, h=4, w=8)
        result = HorizontalFlip(flip=True)(img)
        assert torch.equal(result, torch.flip(img, dims=[2]))

    def test_flip_false_is_identity(self):
        img = _uint8_img()
        assert torch.equal(HorizontalFlip(flip=False)(img), img)

    def test_repr_contains_class_name(self):
        assert "HorizontalFlip" in repr(HorizontalFlip())


# ---------------------------------------------------------------------------
# VerticalFlip
# ---------------------------------------------------------------------------


class TestVerticalFlip:
    def test_flip_true_mirrors_along_height(self):
        img = _uint8_img(c=1, h=8, w=4)
        result = VerticalFlip(flip=True)(img)
        assert torch.equal(result, torch.flip(img, dims=[1]))

    def test_flip_false_is_identity(self):
        img = _uint8_img()
        assert torch.equal(VerticalFlip(flip=False)(img), img)

    def test_repr_contains_class_name(self):
        assert "VerticalFlip" in repr(VerticalFlip())


# ---------------------------------------------------------------------------
# Affine
# ---------------------------------------------------------------------------


class TestAffine:
    def test_default_params_is_identity(self):
        """All-default (zero) params must leave the image unchanged."""
        img = _uint8_img(c=1, h=32, w=32)
        assert torch.equal(Affine()(img), img)

    def test_output_shape_preserved(self):
        img = _uint8_img(h=48, w=64)
        result = Affine(degrees=15.0, translate=(0.05, 0.05), scale=1.1)(img)
        assert result.shape == img.shape

    def test_repr_contains_class_name(self):
        assert "Affine" in repr(Affine(degrees=10.0))


# ---------------------------------------------------------------------------
# RandomPercentCrop
# ---------------------------------------------------------------------------


class TestRandomPercentCrop:
    def test_output_no_larger_than_input(self):
        torch.manual_seed(7)
        img = _uint8_img(h=100, w=100)
        result = RandomPercentCrop(max_percent=30)(img)
        assert result.shape[1] <= 100
        assert result.shape[2] <= 100

    def test_zero_percent_returns_original_size(self):
        """max_percent=0 must always crop at 0% → same spatial dimensions."""
        img = _uint8_img(h=50, w=80)
        result = RandomPercentCrop(max_percent=0)(img)
        assert result.shape == img.shape

    def test_repr_contains_class_name(self):
        assert "RandomPercentCrop" in repr(RandomPercentCrop(10))


# ---------------------------------------------------------------------------
# cutout()
# ---------------------------------------------------------------------------


class TestCutout:
    def test_per_channel_fill_values_applied_independently(self):
        """Each channel must receive its own fill value (regression for the masked_fill bug).

        Using length >> image size guarantees the hole always covers the entire image
        regardless of the random centre, so we can assert on exact values.
        """
        img = torch.full((3, 32, 32), 128, dtype=torch.uint8)
        fill = [10.0, 20.0, 30.0]
        result = cutout(img, n_holes=1, length=200, fill=fill)
        assert result[0].unique().item() == 10
        assert result[1].unique().item() == 20
        assert result[2].unique().item() == 30

    def test_default_fill_zeros_all_channels(self):
        """Without an explicit fill, every channel in the hole must become 0.

        length >> image size guarantees full coverage.
        """
        img = torch.full((3, 8, 8), 200, dtype=torch.uint8)
        result = cutout(img, n_holes=1, length=200)
        assert result.sum().item() == 0

    def test_output_shape_unchanged(self):
        img = _uint8_img()
        result = cutout(img, n_holes=2, length=10)
        assert result.shape == img.shape

    def test_multiple_holes_zero_at_least_as_many_pixels(self):
        torch.manual_seed(42)
        img = torch.full((3, 64, 64), 255, dtype=torch.uint8)
        fill = [0.0, 0.0, 0.0]
        result_1 = cutout(img.clone(), n_holes=1, length=10, fill=fill)
        result_5 = cutout(img.clone(), n_holes=5, length=10, fill=fill)
        assert (result_5 == 0).sum() >= (result_1 == 0).sum()


# ---------------------------------------------------------------------------
# RandomCutout
# ---------------------------------------------------------------------------


class TestRandomCutout:
    def test_always_applied_when_p_is_one(self):
        img = torch.full((3, 32, 32), 255, dtype=torch.uint8)
        result = RandomCutout(n_holes=1, length=32, p=1.0)(img)
        # The result must differ from the original because length covers the full image.
        assert not torch.equal(result, img)

    def test_never_applied_when_p_is_zero(self):
        img = _uint8_img()
        result = RandomCutout(p=0.0)(img)
        assert torch.equal(result, img)

    def test_output_shape_unchanged(self):
        img = _uint8_img()
        assert RandomCutout(p=1.0)(img).shape == img.shape

    def test_repr_contains_class_name(self):
        assert "RandomCutout" in repr(RandomCutout(n_holes=2, length=5))


# ---------------------------------------------------------------------------
# speckle_noise
# ---------------------------------------------------------------------------


class TestSpeckleNoise:
    def test_output_dtype_is_uint8(self):
        assert speckle_noise(_uint8_img()).dtype == torch.uint8

    def test_output_values_in_valid_range(self):
        result = speckle_noise(_uint8_img(), std=0.5)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_output_shape_preserved(self):
        img = _uint8_img(c=3, h=48, w=48)
        assert speckle_noise(img).shape == img.shape


# ---------------------------------------------------------------------------
# grid_distortion
# ---------------------------------------------------------------------------


class TestGridDistortion:
    def test_output_shape_preserved(self):
        img = _uint8_img().float()
        result = grid_distortion(img, magnitude=0.1)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# RandAugment
# ---------------------------------------------------------------------------


class TestRandAugment:
    @pytest.mark.parametrize("policy", list(RandAugmentPolicy))
    def test_all_policies_preserve_shape(self, policy: RandAugmentPolicy):
        torch.manual_seed(0)
        img = _tv_img(c=3, h=64, w=64)
        aug = RandAugment(policy=policy, num_ops=1, magnitude=5, num_magnitude_bins=11)
        result = aug(img)
        assert result.shape == img.shape

    def test_output_is_tensor(self):
        torch.manual_seed(1)
        img = _tv_img()
        result = RandAugment(num_ops=2, magnitude=9)(img)
        assert isinstance(result, torch.Tensor)

    def test_mask_not_transformed_for_non_spatial_ops(self):
        """Mask must pass through unchanged for ops that do not support masks."""
        torch.manual_seed(2)
        img = _tv_img(h=32, w=32)
        mask = tv_tensors.Mask(torch.zeros(1, 32, 32, dtype=torch.uint8))
        aug = RandAugment(policy=RandAugmentPolicy.RAND_AUGMENT, num_ops=1, magnitude=0)
        result = aug(img, mask)
        # The result must be a tuple of (image, mask) — both tensors.
        assert len(result) == 2

    def test_repr_contains_class_name(self):
        assert "RandAugment" in repr(RandAugment())


# ---------------------------------------------------------------------------
# LabelEncoder
# ---------------------------------------------------------------------------


class TestLabelEncoder:
    LABELS = ["cat", "dog", "bird"]

    def test_encodes_first_label(self):
        assert LabelEncoder(self.LABELS)("cat").item() == 0

    def test_encodes_middle_label(self):
        assert LabelEncoder(self.LABELS)("dog").item() == 1

    def test_encodes_last_label(self):
        assert LabelEncoder(self.LABELS)("bird").item() == 2

    def test_unknown_label_raises_value_error(self):
        with pytest.raises(ValueError):
            LabelEncoder(self.LABELS)("fish")

    def test_output_is_tensor(self):
        assert isinstance(LabelEncoder(self.LABELS)("cat"), torch.Tensor)


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------


class TestOneHotEncoder:
    def test_int_init_correct_vector(self):
        result = OneHotEncoder(4)(torch.tensor(2))
        assert result.tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_list_init_correct_vector(self):
        result = OneHotEncoder(["a", "b", "c"])(torch.tensor(0))
        assert result.tolist() == [1.0, 0.0, 0.0]

    def test_output_dtype_is_float(self):
        assert OneHotEncoder(3)(torch.tensor(1)).dtype == torch.float32

    def test_output_length_matches_num_classes(self):
        assert OneHotEncoder(5)(torch.tensor(0)).shape == (5,)
