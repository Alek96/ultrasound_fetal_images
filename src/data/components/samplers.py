from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from src import utils

log = utils.get_pylogger(__name__)


class UnderSampler(Sampler[int]):
    """Undersample by drawing a fixed number of indices per class, then shuffling."""

    def __init__(
        self,
        classes_indices: Sequence[Sequence[int]],
        classes_num_samples: Sequence[int],
    ) -> None:
        self.classes_indices = classes_indices
        self.classes_num_samples = classes_num_samples
        self._num_samples = sum(self.classes_num_samples)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        indices = []
        for idx, num_samples in zip(self.classes_indices, self.classes_num_samples):
            if num_samples > 0 and len(idx) > 0:
                perm = torch.randperm(len(idx))[:num_samples]
                indices.extend([idx[i] for i in perm])

        for i in torch.randperm(len(indices)):
            index = indices[i]
            yield index.item() if isinstance(index, torch.Tensor) else index

    def __len__(self) -> int:
        return self.num_samples


def _build_class_sampling_plan(
    dataset: Dataset,
    labels: torch.Tensor,
    max_sizes: Sequence[int],
) -> tuple[Sequence[torch.Tensor], Sequence[int]]:
    """Per-class dataset indices and draw counts for one dataset.

    Args:
        dataset: Dataset whose items are indexed as ``dataset[i, 1]`` to retrieve the label.
        labels: 1-D tensor of *unique* class IDs expected in this dataset
                (e.g. ``torch.tensor([0, 1, 2])`` for a 3-class problem).
        max_sizes: Maximum number of samples to draw per class, aligned with ``labels``.
                   A value of ``-1`` means "use all available samples for that class".
    """
    if len(max_sizes) != len(labels):
        raise ValueError(f"max_sizes length ({len(max_sizes)}) must match the number of unique classes ({len(labels)})")

    classes = torch.tensor([dataset[i, 1].item() for i in range(len(dataset))])
    class_ids = torch.unique(labels, dim=None)
    classes_indices = [torch.nonzero(classes == class_id).flatten() for class_id in class_ids]
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
        dataset_classes_indices, dataset_classes_num_samples = _build_class_sampling_plan(dataset, labels, max_size)
        dataset_classes_indices = [indices + classes_index for indices in dataset_classes_indices]

        classes_indices.extend(dataset_classes_indices)
        classes_num_samples.extend(dataset_classes_num_samples)
        classes_index += len(dataset)

    return UnderSampler(
        classes_indices=classes_indices,
        classes_num_samples=classes_num_samples,
    )


def get_over_sampler(dataset: Dataset, num_samples: int | None = None) -> WeightedRandomSampler:
    log.info("Instantiating OverSampler")
    classes = np.array([dataset[i, 1].item() for i in range(len(dataset))])
    class_sample_count = np.array([len(np.nonzero(classes == class_id)[0]) for class_id in np.unique(classes)])
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[label] for label in classes])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    return WeightedRandomSampler(
        weights=samples_weight,
        num_samples=num_samples or len(dataset),
        replacement=True,
    )
