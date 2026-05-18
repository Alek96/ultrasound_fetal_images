"""Unit tests for src.data.components.samplers."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from src.data.components.samplers import (
    UnderSampler,
    get_over_sampler,
    get_under_sampler,
)

# ---------------------------------------------------------------------------
# Shared fixture: a lightweight dataset with tuple-index support
# ---------------------------------------------------------------------------


class _MockDataset(Dataset):
    """Minimal dataset that stores (dummy_image, label) pairs.

    Supports the ``dataset[i, 1]`` tuple-index convention used by the samplers.
    """

    def __init__(self, labels: list[int]) -> None:
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, col = idx
            pair = (torch.zeros(1), torch.tensor(self._labels[i]))
            return pair[col]
        return torch.zeros(1), torch.tensor(self._labels[idx])


# ---------------------------------------------------------------------------
# UnderSampler
# ---------------------------------------------------------------------------


class TestUnderSampler:
    def test_len_equals_sum_of_num_samples(self):
        sampler = UnderSampler(
            classes_indices=[
                torch.arange(0, 10),
                torch.arange(10, 20),
                torch.arange(20, 30),
            ],
            classes_num_samples=[5, 3, 2],
        )

        assert len(sampler) == 10

    def test_iter_yields_correct_total_count(self):
        sampler = UnderSampler(
            classes_indices=[
                torch.arange(0, 10),
                torch.arange(10, 20),
            ],
            classes_num_samples=[4, 6],
        )

        assert len(list(sampler)) == 10

    def test_iter_yields_plain_python_ints(self):
        """All yielded indices must be Python ints, not 0-dim tensors (regression guard)."""
        sampler = UnderSampler(
            classes_indices=[
                [1, 2, 3],
                torch.arange(10, 13),
            ],
            classes_num_samples=[3, 3],
        )

        for idx in sampler:
            assert isinstance(idx, int), f"Expected int, got {type(idx)}"

    def test_all_yielded_indices_within_valid_range(self):
        """Every index must fall within the global range of the provided classes_indices."""
        sampler = UnderSampler(
            classes_indices=[
                torch.arange(0, 10),
                torch.arange(10, 20),
            ],
            classes_num_samples=[5, 6],
        )

        valid = set(range(20))  # two classes × 10 samples each
        for idx in sampler:
            assert idx in valid

    def test_empty_class_is_skipped_gracefully(self):
        """A class with zero num_samples must not crash and must not contribute indices."""
        sampler = UnderSampler(
            classes_indices=[
                torch.arange(0, 5),
                torch.arange(5, 10),
            ],
            classes_num_samples=[3, 0],
        )

        assert len(sampler) == 3
        indices = list(sampler)
        assert all(i < 5 for i in indices)

    def test_empty_class_indices_are_skipped_gracefully(self):
        """A class whose index list is empty must not crash."""
        sampler = UnderSampler(
            classes_indices=[
                torch.arange(5),
                torch.tensor([], dtype=torch.long),
            ],
            classes_num_samples=[3, 3],
        )
        assert len(list(sampler)) == 3

    def test_num_samples_property(self):
        sampler = UnderSampler(
            classes_indices=[
                torch.arange(0, 10),
                torch.arange(10, 20),
            ],
            classes_num_samples=[2, 8],
        )

        assert sampler.num_samples == 10


# ---------------------------------------------------------------------------
# get_under_sampler
# ---------------------------------------------------------------------------


class TestGetUnderSampler:
    def test_single_dataset_correct_total(self):
        """With two balanced classes and no cap, all samples should be drawn."""
        labels = [0] * 6 + [1] * 6
        ds = _MockDataset(labels)
        sampler = get_under_sampler(
            datasets=[ds],
            labels=torch.tensor([0, 1]),
            max_sizes=[[-1, -1]],
        )
        assert len(sampler) == 12

    def test_single_dataset_capped_per_class(self):
        labels = [0] * 10 + [1] * 10
        ds = _MockDataset(labels)
        sampler = get_under_sampler(
            datasets=[ds],
            labels=torch.tensor([0, 1]),
            max_sizes=[[4, 4]],
        )
        assert len(sampler) == 8

    def test_multiple_datasets_indices_are_offset(self):
        """Indices from the second dataset must be shifted by the length of the first."""
        ds_a = _MockDataset([0] * 5 + [1] * 5)  # indices 0–9
        ds_b = _MockDataset([0] * 5 + [1] * 5)  # indices 10–19 after offset
        sampler = get_under_sampler(
            datasets=[ds_a, ds_b],
            labels=torch.tensor([0, 1]),
            max_sizes=[[-1, -1], [-1, -1]],
        )
        dataset = ConcatDataset([ds_a, ds_b])

        assert len(sampler) == 20
        assert len(list(sampler)) == 20
        assert sorted(list(sampler)) == list(range(20))

        for idx in sampler:
            assert dataset[idx] is not None

    def test_max_sizes_length_mismatch_raises(self):
        ds = _MockDataset([0, 1, 2])
        with pytest.raises(ValueError, match="must match the number of unique classes"):
            get_under_sampler(
                datasets=[ds],
                labels=torch.tensor([0, 1, 2]),
                max_sizes=[[10]],  # only 1 entry for 3 classes
            )


# ---------------------------------------------------------------------------
# get_over_sampler
# ---------------------------------------------------------------------------


class TestGetOverSampler:
    def _imbalanced_dataset(self) -> _MockDataset:
        """4× more class-0 samples than class-1."""
        return _MockDataset([0] * 80 + [1] * 20)

    def test_returns_weighted_random_sampler(self):
        sampler = get_over_sampler(self._imbalanced_dataset())
        assert isinstance(sampler, WeightedRandomSampler)

    def test_default_num_samples_equals_dataset_length(self):
        ds = self._imbalanced_dataset()
        sampler = get_over_sampler(ds)
        assert sampler.num_samples == len(ds)

    def test_custom_num_samples_respected(self):
        sampler = get_over_sampler(self._imbalanced_dataset(), num_samples=50)
        assert sampler.num_samples == 50

    def test_minority_class_gets_higher_weight(self):
        """Minority class (class-1) must have a larger sampling weight than majority class."""
        ds = self._imbalanced_dataset()
        sampler = get_over_sampler(ds)
        weights = sampler.weights.tolist()
        # Class-0 weight < class-1 weight in an imbalanced dataset.
        w_majority = weights[0]  # class 0
        w_minority = weights[-1]  # class 1
        assert w_minority > w_majority
