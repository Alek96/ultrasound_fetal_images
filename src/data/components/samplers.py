from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class UnderSampler(Sampler[int]):
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
            indices.extend([idx[i] for i in torch.randperm(len(idx))[:num_samples]])

        for i in torch.randperm(len(indices)):
            yield indices[i]

    def __len__(self) -> int:
        return self.num_samples
