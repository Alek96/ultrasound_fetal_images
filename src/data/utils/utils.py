from math import ceil, sqrt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

from src import utils
from src.data.components.samplers import UnderSampler

log = utils.get_pylogger(__name__)


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


def get_under_sampler(dataset: Dataset) -> UnderSampler:
    log.info("Instantiating UnderSampler")
    classes = torch.tensor([dataset[i, 1].item() for i in range(len(dataset))])
    classes_indices = [torch.nonzero(classes == class_id).flatten() for class_id in torch.unique(classes)]
    # classes_indices[3] = torch.cat([classes_indices[3], classes_indices[3]])
    classes_num_samples = [len(indices) for indices in classes_indices]
    classes_num_samples[-1] = 500

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
    tick_labels: bool = False,
    cols_names=None,
    rows_names=None,
    title: str = None,
    ylabel: str = None,
):
    cols_names = cols_names if cols_names else []
    rows_names = rows_names if rows_names else []

    n = ceil(sqrt(len(images)))
    figsize = 16
    scale = 165 / 230
    fig, axes = plt.subplots(ncols=n, nrows=n, squeeze=False, figsize=(figsize, figsize * scale))

    for i in range(n):
        for j in range(n):
            if i * n + j >= len(images):
                continue

            img = images[i * n + j]
            label = None

            if img is None:
                continue

            if isinstance(img, tuple):
                img, label = img
            img = img.detach()
            img = TF.to_pil_image(img)
            img = TF.to_grayscale(img)
            axes[i, j].imshow(np.asarray(img), cmap="gray")
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
        for i in range(n):
            for j in range(n):
                axes[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.tight_layout(h_pad=0.1, w_pad=0.1)
    return fig


def show_numpy_images(images: list[tuple[np.ndarray, str]]):
    n = ceil(sqrt(len(images)))

    fig, axes = plt.subplots(ncols=n, nrows=n, squeeze=False, figsize=(20, 15))

    for i in range(n):
        for j in range(n):
            if i * n + j >= len(images):
                continue

            img, label = images[i * n + j]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            axes[i, j].imshow(np.asarray(img), cmap="gray")
            axes[i, j].set_xlabel(label)
            axes[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.tight_layout()
    plt.show()
