"""Unit tests for src/models/components/loss."""

import pytest
import torch
from torchmetrics import F1Score

from src.models.components.loss import (
    BinaryDiceCrossEntropyLoss,
    BinaryDiceFocalLoss,
    BinaryDiceLoss,
    BinaryDiceScore,
    BinaryFocalLoss,
    BinaryFocalTverskyLoss,
    BinaryTverskyLoss,
    WeightedMSELoss,
)

# ---------------------------------------------------------------------------
# WeightedMSELoss
# ---------------------------------------------------------------------------


class TestWeightedMSELoss:
    def test_zero_residual(self):
        loss_fn = WeightedMSELoss(weight=2.0)
        t = torch.ones(4)
        assert loss_fn(t, t).item() == pytest.approx(0.0)

    def test_over_prediction_is_upweighted(self):
        """Positive residuals (over-predictions) must receive the custom weight."""
        loss_fn = WeightedMSELoss(weight=3.0)
        y_hat = torch.tensor([2.0])
        y = torch.tensor([1.0])
        unweighted = WeightedMSELoss(weight=1.0)(y_hat, y)
        weighted = loss_fn(y_hat, y)
        assert weighted.item() > unweighted.item()

    def test_under_prediction_uses_weight_one(self):
        """Negative residuals (under-predictions) must use weight 1 regardless of setting."""
        loss_fn = WeightedMSELoss(weight=5.0)
        y_hat = torch.tensor([0.0])
        y = torch.tensor([1.0])
        assert loss_fn(y_hat, y).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BinaryDiceScore
# ---------------------------------------------------------------------------


class TestBinaryDiceScore:
    def test_perfect_prediction(self):
        score_fn = BinaryDiceScore()
        t = torch.ones(1, 1, 4, 4)
        score_fn.update(t, t)
        assert score_fn.compute().item() == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        score_fn = BinaryDiceScore()
        inputs = torch.zeros(1, 1, 4, 4)
        targets = torch.ones(1, 1, 4, 4)
        score_fn.update(inputs, targets)
        assert score_fn.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_output_in_range(self):
        score_fn = BinaryDiceScore()
        inputs = torch.rand(2, 1, 8, 8)
        targets = (torch.rand(2, 1, 8, 8) > 0.5).float()
        score_fn.update(inputs, targets)
        score = score_fn.compute().item()
        assert 0.0 <= score <= 1.0

    def test_accepts_channelless_input(self):
        """The metric must accept both [B, 1, H, W] and [B, H, W] shapes."""
        score_fn = BinaryDiceScore()
        t = torch.ones(2, 4, 4)
        score_fn.update(t, t)
        assert score_fn.compute().item() == pytest.approx(1.0, abs=1e-5)

    def test_equals_pixel_f1(self):
        """On binary masks Dice == foreground F1; assert exact match incl. a
        partial final batch and an empty-mask sample (regression for the old
        batch-averaging bug)."""
        dice = BinaryDiceScore()
        f1 = F1Score(task="binary")
        for n in [8, 8, 3]:
            preds = torch.rand(n, 1, 16, 16)
            targets = (torch.rand(n, 1, 16, 16) > 0.5).float()
            if n == 3:
                targets[0] = 0.0  # empty-mask sample
            dice.update(preds, targets)
            f1.update(preds, targets)
        assert dice.compute().item() == pytest.approx(f1.compute().item(), abs=1e-6)

    def test_global_not_batch_average(self):
        """Global Dice must differ from a naive per-batch mean of Dice ratios.

        Batch A is easy (perfect); batch B has a tiny prediction over a large
        target. The per-batch mean would be ~0.5+, whereas the correct global
        Dice pools counts and is much lower.
        """
        dice = BinaryDiceScore()
        # Batch A: perfect overlap, small region.
        a_pred = torch.zeros(1, 1, 10, 10)
        a_target = torch.zeros(1, 1, 10, 10)
        a_pred[..., 0, 0] = 1.0
        a_target[..., 0, 0] = 1.0
        # Batch B: predict nothing over a fully-positive target.
        b_pred = torch.zeros(1, 1, 10, 10)
        b_target = torch.ones(1, 1, 10, 10)
        dice.update(a_pred, a_target)
        dice.update(b_pred, b_target)
        # Global: numerator=2*1, denominator=(1)+(1+100)=102 -> 2/102.
        expected = 2.0 / 102.0
        naive_batch_mean = (1.0 + 0.0) / 2.0
        score = dice.compute().item()
        assert score == pytest.approx(expected, abs=1e-4)
        assert abs(score - naive_batch_mean) > 0.4


# ---------------------------------------------------------------------------
# BinaryDiceLoss
# ---------------------------------------------------------------------------


class TestBinaryDiceLoss:
    def test_perfect_prediction(self):
        loss_fn = BinaryDiceLoss()
        logits = torch.full((1, 4, 4), 100.0)
        targets = torch.ones(1, 4, 4)
        assert loss_fn(logits, targets).item() == pytest.approx(0.0, abs=1e-3)

    def test_is_non_negative(self):
        loss_fn = BinaryDiceLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert loss_fn(logits, targets).item() >= 0.0


# ---------------------------------------------------------------------------
# BinaryFocalLoss
# ---------------------------------------------------------------------------


class TestBinaryFocalLoss:
    def test_is_non_negative(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert loss_fn(logits, targets).item() >= 0.0

    def test_perfect_prediction_is_near_zero(self):
        loss_fn = BinaryFocalLoss(alpha=None)
        logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert loss_fn(logits, targets).item() == pytest.approx(0.0, abs=1e-4)

    def test_gamma_zero_no_alpha_equals_bce(self):
        """With gamma=0 and alpha disabled, focal loss reduces to plain BCE."""
        loss_fn = BinaryFocalLoss(alpha=None, gamma=0.0)
        bce_fn = torch.nn.BCEWithLogitsLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert loss_fn(logits, targets).item() == pytest.approx(bce_fn(logits, targets).item(), rel=1e-5)

    def test_gamma_downweights_easy_examples(self):
        """Higher gamma must reduce loss on confidently-correct predictions."""
        logits = torch.tensor([[3.0]])
        targets = torch.tensor([[1.0]])
        low_gamma = BinaryFocalLoss(alpha=None, gamma=0.0)(logits, targets)
        high_gamma = BinaryFocalLoss(alpha=None, gamma=2.0)(logits, targets)
        assert high_gamma.item() < low_gamma.item()

    def test_reduction_none_preserves_shape(self):
        loss_fn = BinaryFocalLoss(reduction="none")
        logits = torch.randn(2, 1, 4, 4)
        targets = (torch.rand(2, 1, 4, 4) > 0.5).float()
        assert loss_fn(logits, targets).shape == logits.shape


# ---------------------------------------------------------------------------
# BinaryTverskyLoss
# ---------------------------------------------------------------------------


class TestBinaryTverskyLoss:
    def test_is_non_negative(self):
        loss_fn = BinaryTverskyLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert loss_fn(logits, targets).item() >= 0.0

    def test_perfect_prediction_is_near_zero(self):
        loss_fn = BinaryTverskyLoss(smooth=0.0)
        logits = torch.full((1, 4, 4), 100.0)
        targets = torch.ones(1, 4, 4)
        assert loss_fn(logits, targets).item() == pytest.approx(0.0, abs=1e-3)

    def test_equals_dice_when_alpha_beta_half(self):
        """alpha == beta == 0.5 must be equivalent to soft Dice loss.

        The equivalence is exact only without smoothing (the smoothing constant
        enters the Dice and Tversky denominators at different scales).
        """
        tversky_fn = BinaryTverskyLoss(alpha=0.5, beta=0.5, smooth=0.0)
        dice_fn = BinaryDiceLoss(smooth=0.0)
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert tversky_fn(logits, targets).item() == pytest.approx(dice_fn(logits, targets).item(), rel=1e-5)

    def test_higher_alpha_penalises_false_positives(self):
        """With only false positives present, larger alpha must increase the loss."""
        # Prediction is all-positive, target is all-negative => pure false positives.
        logits = torch.full((1, 4, 4), 100.0)
        targets = torch.zeros(1, 4, 4)
        low_alpha = BinaryTverskyLoss(alpha=0.3, beta=0.7, smooth=1e-6)(logits, targets)
        high_alpha = BinaryTverskyLoss(alpha=0.7, beta=0.3, smooth=1e-6)(logits, targets)
        assert high_alpha.item() > low_alpha.item()


# ---------------------------------------------------------------------------
# BinaryDiceCrossEntropyLoss
# ---------------------------------------------------------------------------


class TestBinaryDiceCrossEntropyLoss:
    def test_combined_loss_is_non_negative(self):
        loss_fn = BinaryDiceCrossEntropyLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert loss_fn(logits, targets).item() >= 0.0

    def test_combined_loss_equals_dice_plus_bce(self):
        """Combined loss must equal individual dice + bce losses."""
        loss_fn = BinaryDiceCrossEntropyLoss(smooth=1.0)
        dice_fn = BinaryDiceLoss(smooth=1.0)
        bce_fn = torch.nn.BCEWithLogitsLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        expected = dice_fn(logits, targets) + bce_fn(logits, targets)
        assert loss_fn(logits, targets).item() == pytest.approx(expected.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# BinaryDiceFocalLoss
# ---------------------------------------------------------------------------


class TestBinaryDiceFocalLoss:
    def test_is_non_negative(self):
        loss_fn = BinaryDiceFocalLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert loss_fn(logits, targets).item() >= 0.0

    def test_equals_dice_plus_focal(self):
        """Combined loss must equal individual dice + focal losses."""
        loss_fn = BinaryDiceFocalLoss(smooth=1.0, alpha=0.25, gamma=2.0)
        dice_fn = BinaryDiceLoss(smooth=1.0)
        focal_fn = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        expected = dice_fn(logits, targets) + focal_fn(logits, targets)
        assert loss_fn(logits, targets).item() == pytest.approx(expected.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# BinaryFocalTverskyLoss
# ---------------------------------------------------------------------------


class TestBinaryFocalTverskyLoss:
    def test_is_non_negative(self):
        loss_fn = BinaryFocalTverskyLoss()
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert loss_fn(logits, targets).item() >= 0.0

    def test_perfect_prediction_is_near_zero(self):
        loss_fn = BinaryFocalTverskyLoss(smooth=0.0)
        logits = torch.full((1, 4, 4), 100.0)
        targets = torch.ones(1, 4, 4)
        assert loss_fn(logits, targets).item() == pytest.approx(0.0, abs=1e-3)

    def test_equals_dice_when_alpha_beta_half_and_gamma_one(self):
        """alpha == beta == 0.5 and gamma == 1 must reduce to soft Dice loss.

        Exact only without smoothing (see TestBinaryTverskyLoss equivalence note).
        """
        ft_fn = BinaryFocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0, smooth=0.0)
        dice_fn = BinaryDiceLoss(smooth=0.0)
        logits = torch.randn(2, 8, 8)
        targets = (torch.rand(2, 8, 8) > 0.5).float()
        assert ft_fn(logits, targets).item() == pytest.approx(dice_fn(logits, targets).item(), rel=1e-5)
