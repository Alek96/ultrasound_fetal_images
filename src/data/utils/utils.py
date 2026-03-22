from collections.abc import Sequence
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
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.io import read_image

from src import utils
from src.data.components.dataset import Subset
from src.data.components.samplers import UnderSampler

log = utils.get_pylogger(__name__)


def read_image_tensor(image_path: str | Path):
    image = read_image(image_path)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3, :, :]
    return image


def save_image_tensor(image: Tensor, output_path: str | Path):
    image = image.permute(1, 2, 0).numpy()
    image = PIL.Image.fromarray(image)
    image.save(output_path)
    return image


def group_split(
    dataset: Dataset,
    test_size: float,
    groups: pd.Series,
    random_state: int = None,
) -> tuple[Subset, Subset]:
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    split = splitter.split(dataset, groups=groups)
    train_idx, test_idx = next(split)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def get_under_sampler_config(
    dataset: Dataset,
    labels: torch.Tensor,
    max_sizes: Sequence[int],
) -> tuple[Sequence[torch.Tensor], Sequence[int]]:
    classes = torch.tensor([dataset[i, 1].item() for i in range(len(dataset))])
    classes_indices = [torch.nonzero(classes == class_id).flatten() for class_id in torch.unique(labels)]
    classes_num_samples = [len(indices) for indices in classes_indices]

    for i, max_size in enumerate(max_sizes):
        if max_size >= 0:
            classes_num_samples[i] = min(classes_num_samples[i], max_size)

    return classes_indices, classes_num_samples


def get_under_sampler(
    datasets: Sequence[Dataset], labels: torch.Tensor, max_sizes: Sequence[Sequence[int]]
) -> UnderSampler:
    log.info("Instantiating UnderSampler")
    classes_index = 0
    classes_indices = []
    classes_num_samples = []
    for dataset, max_size in zip(datasets, max_sizes):
        dataset_classes_indices, dataset_classes_num_samples = get_under_sampler_config(dataset, labels, max_size)
        dataset_classes_indices = [indices + classes_index for indices in dataset_classes_indices]

        classes_indices.extend(dataset_classes_indices)
        classes_num_samples.extend(dataset_classes_num_samples)
        classes_index += len(dataset)

    return UnderSampler(
        classes_indices=classes_indices,
        classes_num_samples=classes_num_samples,
    )


def get_over_sampler(dataset: Dataset) -> WeightedRandomSampler:
    log.info("Instantiating OverSampler")
    classes = np.array([dataset[i, 1].item() for i in range(len(dataset))])
    class_sample_count = np.array([len(np.where(classes == class_id)[0]) for class_id in np.unique(classes)])
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[label] for label in classes])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    return WeightedRandomSampler(
        weights=samples_weight,
        num_samples=1800,
        replacement=True,
    )


def show_pytorch_images(
    images: list[tuple[Tensor, str]],
    gray: bool = True,
    tick_labels: bool = False,
    cols_names=None,
    rows_names=None,
    title: str = None,
    ylabel: str = None,
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


def show_numpy_images(images: list[tuple[np.ndarray, str]]):
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
    plt.show()
