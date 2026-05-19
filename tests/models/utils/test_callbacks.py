"""Unit tests for src/models/utils/callbacks.

Tests drive on_train_batch_start — the public Lightning entrypoint — rather than
the internal mix() method.  The trainer and pl_module arguments are not used
inside on_train_batch_start, so None is passed for both.

Batch is always a plain list [x, y] because on_train_batch_start mutates it
in-place (batch[0] = x_new; batch[1] = y_new).

Integer labels are used as the primary input to exercise the one-hot encoding
path (not softmax_target and len(y.shape) == 1).  For softmax_target=True tests,
pre-computed one-hot tensors are supplied directly.
"""

import pytest
import torch
import torch.nn.functional as F
import torch.testing as tt

from src.models.utils.callbacks import MixUpCallback, MixUpV2Callback, VHMixUpCallback

BATCH = 8
CLASSES = 5
H, W = 32, 32


# # Deterministic inputs so tests are reproducible.
# torch.manual_seed(0)
# _IMG = torch.rand(BATCH, 1, H, W)
# _LABELS_INT = torch.randint(0, CLASSES, (BATCH,))
#
#
# def _batch_int() -> list:
#     """Fresh list[x, y_int] so each test gets its own mutable batch."""
#     return [_IMG.clone(), _LABELS_INT.clone()]


@pytest.fixture()
def images() -> torch.Tensor:
    return torch.rand(BATCH, 1, H, W)


@pytest.fixture()
def labels() -> torch.Tensor:
    return torch.randint(0, CLASSES, (BATCH,))


@pytest.fixture()
def batch(images: torch.Tensor, labels: torch.Tensor):
    return [images.clone(), labels.clone()]


# ---------------------------------------------------------------------------
# MixUpCallback
# ---------------------------------------------------------------------------


class TestMixUpCallback:
    def test_batch_mutated_in_place(self, batch):
        """The original list object must be modified, not replaced."""
        original_list_id = id(batch)
        mix_up = MixUpCallback(alpha=0.4, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert id(batch) == original_list_id

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_x_shape_preserved(self, softmax_target, batch, images):
        mix_up = MixUpCallback(alpha=0.4, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert batch[0].shape == images.shape

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_alpha_zero_x_unchanged(self, softmax_target, batch, images):
        """alpha=0 => lam=1 => x must be identical to the original input."""
        mix_up = MixUpCallback(alpha=0.0, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert torch.allclose(batch[0], images)

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_mixed_x_values_in_range(self, softmax_target, batch, images):
        """Pixel values must stay within [0.0, 1.0] after mixing."""
        mix_up = MixUpCallback(alpha=0.5, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert batch[0].min() >= 0.0
        assert batch[0].max() <= 1.0

    class TestSoftmaxTargetFalse:
        def test_y_shape_is_batch_x_classes(self, batch):
            """Output must be a (batch, classes) tensor — not raw integer labels."""
            mix_up = MixUpCallback(alpha=0.4, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert y.shape == (BATCH, CLASSES)

        def test_y_rows_sum_to_one(self, batch):
            """Mixed one-hot rows must conserve probability mass (sum == 1 per sample)."""
            mix_up = MixUpCallback(alpha=0.4, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y.sum(dim=1), torch.ones(BATCH))

        def test_alpha_zero_y_preserved_as_one_hot(self, batch, labels):
            """alpha=0 => lam=1 => y_new must equal the one-hot encoding of the original labels."""
            mix_up = MixUpCallback(alpha=0.0, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            expected = F.one_hot(labels, num_classes=CLASSES).float()
            tt.assert_close(y, expected)

    class TestSoftmaxTargetTrue:
        def test_y_shape_is_batch_x_4(self, batch):
            """Output must be a (batch, 4) tensor: [y, y_perm, lam, 1-lam]."""
            mix_up = MixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert y.shape == (BATCH, 4)

        def test_y_original_labels_preserved(self, batch, labels):
            """Column 0 must contain the original (un-permuted) integer labels."""
            mix_up = MixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert torch.allclose(y[:, 0].long(), labels)

        def test_y_permuted_labels_are_a_permutation(self, batch, labels):
            """Column 1 must contain a permutation of the original labels (same multiset)."""
            mix_up = MixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert torch.allclose(torch.sort(y[:, 1].long())[0], torch.sort(labels)[0])

        def test_lambda_sums_to_one(self, batch):
            """Columns 2 and 3 (lam and 1-lam) must sum to 1 for every sample."""
            mix_up = MixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y[:, 2] + y[:, 3], torch.ones(BATCH))

        def test_single_lambda_shared_across_batch(self, batch):
            """MixUpCallback draws one λ for the whole batch — all rows must share the same value."""
            mix_up = MixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            lam_values = y[:, 2]
            tt.assert_close(lam_values, lam_values[0].expand_as(lam_values))

        def test_alpha_zero_lambda_is_one(self, batch, labels):
            """alpha=0 => lam=1 for all samples: lambda column must be all 1s, complement all 0s."""
            mix_up = MixUpCallback(alpha=0.0, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y[:, 0].long(), labels)
            tt.assert_close(y[:, 2], torch.ones(BATCH))
            tt.assert_close(y[:, 3], torch.zeros(BATCH))


# ---------------------------------------------------------------------------
# MixUpV2Callback
# ---------------------------------------------------------------------------


class TestMixUpV2Callback:
    def test_batch_mutated_in_place(self, batch):
        """The original list object must be modified, not replaced."""
        original_list_id = id(batch)
        mix_up = MixUpV2Callback(alpha=0.4, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert id(batch) == original_list_id

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_x_shape_preserved(self, softmax_target, batch, images):
        mix_up = MixUpV2Callback(alpha=0.4, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert batch[0].shape == images.shape

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_alpha_zero_x_unchanged(self, softmax_target, batch, images):
        """alpha=0 => lam=1 => x must be identical to the original input."""
        mix_up = MixUpV2Callback(alpha=0.0, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert torch.allclose(batch[0], images)

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_mixed_x_values_in_range(self, softmax_target, batch, images):
        """Pixel values must stay within [0.0, 1.0] after mixing."""
        mix_up = MixUpV2Callback(alpha=0.5, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert batch[0].min() >= 0.0
        assert batch[0].max() <= 1.0

    class TestSoftmaxTargetFalse:
        def test_y_shape_is_batch_x_classes(self, batch):
            """Output must be a (batch, classes) tensor — not raw integer labels."""
            mix_up = MixUpV2Callback(alpha=0.4, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert y.shape == (BATCH, CLASSES)

        def test_y_rows_sum_to_one(self, batch):
            """Mixed one-hot rows must conserve probability mass (sum == 1 per sample)."""
            mix_up = MixUpV2Callback(alpha=0.4, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y.sum(dim=1), torch.ones(BATCH))

        def test_alpha_zero_y_preserved_as_one_hot(self, batch, labels):
            """alpha=0 => lam=1 => y_new must equal the one-hot encoding of the original labels."""
            mix_up = MixUpV2Callback(alpha=0.0, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            expected = F.one_hot(labels, num_classes=CLASSES).float()
            tt.assert_close(y, expected)

    class TestSoftmaxTargetTrue:
        def test_y_shape_is_batch_x_4(self, batch):
            """Output must be a (batch, 4) tensor: [y, y_perm, lam, 1-lam]."""
            mix_up = MixUpV2Callback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert y.shape == (BATCH, 4)

        def test_y_original_labels_preserved(self, batch, labels):
            """Column 0 must contain the original (un-permuted) integer labels."""
            mix_up = MixUpV2Callback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert torch.allclose(y[:, 0].long(), labels)

        def test_y_permuted_labels_are_a_permutation(self, batch, labels):
            """Column 1 must contain a permutation of the original labels (same multiset)."""
            mix_up = MixUpV2Callback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert torch.allclose(torch.sort(y[:, 1].long())[0], torch.sort(labels)[0])

        def test_lambda_sums_to_one(self, batch):
            """Columns 2 and 3 (lam and 1-lam) must sum to 1 for every sample."""
            mix_up = MixUpV2Callback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y[:, 2] + y[:, 3], torch.ones(BATCH))

        def test_alpha_zero_lambda_is_one(self, batch, labels):
            """alpha=0 => lam=1 for all samples: lambda column must be all 1s, complement all 0s."""
            mix_up = MixUpV2Callback(alpha=0.0, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y[:, 0].long(), labels)
            tt.assert_close(y[:, 2], torch.ones(BATCH))
            tt.assert_close(y[:, 3], torch.zeros(BATCH))


# ---------------------------------------------------------------------------
# VHMixUpCallback
# ---------------------------------------------------------------------------


class TestVHMixUpCallback:
    def test_batch_mutated_in_place(self, batch):
        """The original list object must be modified, not replaced."""
        original_list_id = id(batch)
        mix_up = VHMixUpCallback(alpha=0.4, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert id(batch) == original_list_id

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_x_shape_preserved(self, softmax_target, batch, images):
        mix_up = VHMixUpCallback(alpha=0.4, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert batch[0].shape == images.shape

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_alpha_zero_x_unchanged(self, softmax_target, batch, images):
        """alpha=0 => lam=1 => x must be identical to the original input."""
        mix_up = VHMixUpCallback(alpha=0.0, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert torch.allclose(batch[0], images)

    @pytest.mark.parametrize("softmax_target", [False, True])
    def test_mixed_x_values_in_range(self, softmax_target, batch, images):
        """Pixel values must stay within [0.0, 1.0] after mixing."""
        mix_up = VHMixUpCallback(alpha=0.5, softmax_target=softmax_target, labels=CLASSES)
        mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

        assert batch[0].min() >= 0.0
        assert batch[0].max() <= 1.0

    class TestSoftmaxTargetFalse:
        def test_y_shape_is_batch_x_classes(self, batch):
            """Output must be a (batch, classes) tensor — not raw integer labels."""
            mix_up = VHMixUpCallback(alpha=0.4, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert y.shape == (BATCH, CLASSES)

        def test_y_rows_sum_to_one(self, batch):
            """Mixed one-hot rows must conserve probability mass (sum == 1 per sample)."""
            mix_up = VHMixUpCallback(alpha=0.4, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y.sum(dim=1), torch.ones(BATCH))

        def test_alpha_zero_y_preserved_as_one_hot(self, batch, labels):
            """alpha=0 => lam=1 => y_new must equal the one-hot encoding of the original labels."""
            mix_up = VHMixUpCallback(alpha=0.0, softmax_target=False, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            expected = F.one_hot(labels, num_classes=CLASSES).float()
            tt.assert_close(y, expected)

    class TestSoftmaxTargetTrue:
        def test_y_shape_is_batch_x_4(self, batch):
            """Output must be a (batch, 4) tensor: [y, y_perm, lam, 1-lam]."""
            mix_up = VHMixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert y.shape == (BATCH, 4)

        def test_y_original_labels_preserved(self, batch, labels):
            """Column 0 must contain the original (un-permuted) integer labels."""
            mix_up = VHMixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert torch.allclose(y[:, 0].long(), labels)

        def test_y_permuted_labels_are_a_permutation(self, batch, labels):
            """Column 1 must contain a permutation of the original labels (same multiset)."""
            mix_up = VHMixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            assert torch.allclose(torch.sort(y[:, 1].long())[0], torch.sort(labels)[0])

        def test_lambda_sums_to_one(self, batch):
            """Columns 2 and 3 (y_lam and y_perm_lam) must sum to 1 for every sample."""
            mix_up = VHMixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y[:, 2] + y[:, 3], torch.ones(BATCH))

        def test_alpha_zero_lambda_is_one(self, batch, labels):
            """alpha=0 => lam=1 for all samples: lambda column must be all 1s, complement all 0s."""
            mix_up = VHMixUpCallback(alpha=0.0, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            tt.assert_close(y[:, 0].long(), labels)
            tt.assert_close(y[:, 2], torch.ones(BATCH))
            tt.assert_close(y[:, 3], torch.zeros(BATCH))

        def test_per_sample_lambdas_in_range(self, batch):
            """VHMixUp draws independent lambdas per sample; each must lie in [0, 1]."""
            mix_up = VHMixUpCallback(alpha=0.4, softmax_target=True, labels=CLASSES)
            mix_up.on_train_batch_start(None, None, batch, batch_idx=0)

            _, y = batch
            lam_a, lam_b = y[:, 2], y[:, 3]
            assert lam_a.min() >= 0.0 and lam_a.max() <= 1.0
            assert lam_b.min() >= 0.0 and lam_b.max() <= 1.0
