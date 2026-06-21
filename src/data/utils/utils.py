import math
from math import ceil, sqrt
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision.transforms.v2.functional as TF
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from src import utils
from src.data.components.dataset import Subset

log = utils.get_pylogger(__name__)


def group_split(
    dataset: Dataset,
    test_size: float,
    groups: pd.Series,
    random_state: int | None = None,
) -> tuple[Subset, Subset]:
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    split = splitter.split(dataset, groups=groups)
    train_idx, test_idx = next(split)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def read_image_tensor(image_path: str | Path):
    image = read_image(image_path)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3, :, :]
    return image


def save_image_tensor(image: Tensor, output_path: str | Path):
    """Save a CHW tensor as an image file (expects uint8 values in 0–255)."""
    image = image.cpu().permute(1, 2, 0).numpy()
    image = PIL.Image.fromarray(image)
    image.save(output_path)
    return image


def show_pytorch_images(
    images: list[tuple[Tensor, str]],
    gray: bool = True,
    tick_labels: bool = False,
    cols_names: list[str] | None = None,
    rows_names: list[str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
):
    cols_names = cols_names if cols_names else []
    rows_names = rows_names if rows_names else []

    ncols = ceil(sqrt(len(images)))
    nrows = ceil(len(images) / ncols)
    figsize = 16
    scale = 192 / 256
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(figsize, figsize * scale))

    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j >= len(images):
                continue

            img = images[i * ncols + j]
            label = None
            if isinstance(img, tuple):
                img, label = img

            if img is None:
                continue

            img = img.detach()
            img = TF.to_pil_image(img)

            if gray:
                img = TF.to_grayscale(img)
                img = np.asarray(img)
                axes[i, j].imshow(img, cmap="gray")
            else:
                img = np.asarray(img)
                axes[i, j].imshow(img)

            if label is not None:
                axes[i, j].set_xlabel(label)

    if title:
        fig.suptitle(title, fontsize=16)
    if ylabel:
        fig.supylabel(ylabel, fontsize=16)

    for ax, col in zip(axes[0], cols_names):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], rows_names):
        ax.set_ylabel(row, rotation=90, size="large")

    if not tick_labels:
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.tight_layout(h_pad=0.1, w_pad=0.1)
    return fig


def show_numpy_images(images: list[tuple[np.ndarray, str]]) -> plt.Figure:
    """Display numpy images; 3-channel inputs are assumed to be BGR (OpenCV convention)."""
    ncols = ceil(sqrt(len(images)))
    nrows = ceil(len(images) / ncols)

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(20, 15))

    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j >= len(images):
                continue

            img = images[i * ncols + j]
            label = None
            if isinstance(img, tuple):
                img, label = img

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            axes[i, j].imshow(np.asarray(img), cmap="gray")

            if label is not None:
                axes[i, j].set_xlabel(label)

            axes[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.tight_layout()
    return fig


def _add_half(center: float | int | torch.Tensor, radius: float | int | torch.Tensor) -> float | int:
    """Add 0.5 to radius when center and radius share the same integer-ness.

    Pixel-coordinate convention helper: if both coordinates land on integer grid
    points, or both land on half-pixel offsets, nudge the radius by 0.5 so that
    the ellipse boundary aligns correctly with the underlying pixel grid.
    """
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center)
    if not isinstance(radius, torch.Tensor):
        radius = torch.tensor(radius)
    if torch.isclose(center, torch.round(center)):
        if torch.isclose(radius, torch.round(radius)):
            radius = radius + 0.5
    else:
        if not torch.isclose(radius, torch.round(radius)):
            radius = radius + 0.5

    return radius.item()


def create_ellipse_tensor(
    height: int,
    width: int,
    center_x: float | int,
    center_y: float | int,
    radius_x: float | int,
    radius_y: float | int,
    theta_rad: float | int = 0,
    add_half: bool = False,
) -> torch.Tensor:
    """Create a 2D PyTorch tensor with a binary ellipse mask.

    Args:
        height: The height of the output tensor.
        width: The width of the output tensor.
        center_x: The x-coordinate (column) of the ellipse's centre.
        center_y: The y-coordinate (row) of the ellipse's centre.
        radius_x: The semi-axis of the ellipse along the x-axis (must be > 0).
        radius_y: The semi-axis of the ellipse along the y-axis (must be > 0).
        theta_rad: Counter-clockwise rotation of the ellipse in radians.
        add_half: When True, nudge each radius by 0.5 to align the ellipse
            boundary with the pixel grid (see ``_add_half``).

    Returns:
        A float tensor of shape ``(height, width)`` with 1s inside the ellipse
        and 0s outside.

    Raises:
        ValueError: If ``radius_x`` or ``radius_y`` is zero.
    """
    if radius_x == 0 or radius_y == 0:
        raise ValueError(f"Ellipse radii must be non-zero, got radius_x={radius_x}, radius_y={radius_y}")

    if add_half:
        radius_x = _add_half(center_x, radius_x)
        radius_y = _add_half(center_y, radius_y)

    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    # Translate the coordinates so the center of the ellipse is at (0, 0)
    x_translated = x - center_x
    y_translated = y - center_y

    # Pre-calculate the trigonometric values for the rotation
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    # Apply the rotation transformation to the coordinates
    x_rotated = x_translated * cos_theta + y_translated * sin_theta
    y_rotated = -x_translated * sin_theta + y_translated * cos_theta

    # Apply the ellipse equation to the rotated coordinates
    ellipse_mask = (x_rotated / radius_x) ** 2 + (y_rotated / radius_y) ** 2 <= 1

    # Convert the boolean mask to a float tensor
    ellipse_tensor = ellipse_mask.float()

    return ellipse_tensor


def get_dice_score(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute the Sørensen–Dice coefficient between two binary tensors.

    Both tensors are flattened before comparison, so shape does not need to
    match as long as the total number of elements is the same.

    Args:
        inputs: Predicted binary mask (float, values in [0, 1]).
        targets: Ground-truth binary mask (float, values 0 or 1).
        smooth: Additive smoothing term to avoid division by zero.

    Returns:
        A scalar tensor with the Dice score in [0, 1].
    """
    if inputs.numel() != targets.numel():
        raise ValueError(
            f"inputs and targets must have the same number of elements, got {inputs.numel()} and {targets.numel()}"
        )
    inputs = inputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (inputs * targets).sum()
    dice_score = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice_score


def find_angle(mask: Tensor) -> torch.Tensor:
    """Estimate the orientation angle of a binary mask via PCA.

    Uses eigenvalue decomposition of the 2-D coordinate covariance matrix to
    find the principal axis, then returns its angle in degrees (−180 to 180).

    Args:
        mask: A 2-D binary tensor ``(H, W)``, or a single-channel 3-D tensor
            ``(1, H, W)`` which is automatically squeezed.

    Returns:
        A scalar ``int32`` tensor with the rounded angle in degrees, or 0 if
        the mask contains no foreground pixels.
    """
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    assert mask.dim() == 2, f"Expected a 2D tensor, but got {mask.dim()} dimensions"

    coords = torch.nonzero(mask, as_tuple=False).float()

    if coords.shape[0] == 0:
        return torch.tensor(0, dtype=torch.int32)

    mean = torch.mean(coords, dim=0)
    centered_coords = coords - mean

    covariance_matrix = torch.matmul(centered_coords.T, centered_coords)
    _, eigenvectors = torch.linalg.eigh(covariance_matrix)
    # eigh returns eigenvalues in ascending order; last column is principal axis
    principal_axis = eigenvectors[:, 1]
    angle = torch.atan2(principal_axis[0], principal_axis[1]) * 180 / torch.pi

    return angle.round().int()


def crop(image: Tensor, x1: int, y1: int, x2: int, y2: int, pad: int = 10) -> Tensor:
    """Crop a ``(C, H, W)`` tensor to the bounding box ``[x1, y1, x2, y2]``.

    Args:
        image: Input tensor of shape ``(C, H, W)``.
        x1: Left column of the crop (inclusive, 0-indexed).
        y1: Top row of the crop (inclusive, 0-indexed).
        x2: Right column of the crop (exclusive).
        y2: Bottom row of the crop (exclusive).
        pad: Percentage of the bounding-box size to add as padding on each side.
            For example, ``pad=10`` adds 5 % of the box width to the left and
            right and 5 % of the box height to the top and bottom.
            Padding is clamped so the crop never exceeds the image boundaries.

    Returns:
        Cropped tensor of shape ``(C, y2−y1, x2−x1)`` (after padding adjustments).
    """
    assert image.dim() == 3, "Expected a 3D tensor (C, H, W)"
    assert x1 < x2, "Invalid crop coordinates: (x1) must be less than (x2)"
    assert y1 < y2, "Invalid crop coordinates: (y1) must be less than (y2)"
    assert 0 <= x1 < image.shape[2] and 0 < x2 <= image.shape[2], "Crop x-coordinates must be within image width"
    assert 0 <= y1 < image.shape[1] and 0 < y2 <= image.shape[1], "Crop y-coordinates must be within image height"
    assert 0 <= pad < 100, "Pad percentage must be between 0 and 100"

    pad_x = int((x2 - x1) * (pad / 100.0))
    left_pad = pad_x // 2
    right_pad = pad_x - left_pad

    pad_y = int((y2 - y1) * (pad / 100.0))
    top_pad = pad_y // 2
    bottom_pad = pad_y - top_pad

    x1 = max(x1 - left_pad, 0)
    y1 = max(y1 - top_pad, 0)
    x2 = min(x2 + right_pad, image.shape[2])
    y2 = min(y2 + bottom_pad, image.shape[1])

    return TF.crop(image, y1, x1, y2 - y1, x2 - x1)
