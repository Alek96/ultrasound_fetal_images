"""Unit tests for src/models/components/loss."""

import pytest
import torch

from src.models.components.loss import (
    BinaryDiceCrossEntropyLoss,
    BinaryDiceLoss,
    BinaryDiceScore,
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
        t = torch.ones(1, 4, 4)
        assert score_fn(t, t).item() == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        score_fn = BinaryDiceScore(smooth=0.0)
        inputs = torch.zeros(1, 4, 4)
        targets = torch.ones(1, 4, 4)
        assert score_fn(inputs, targets).item() == pytest.approx(0.0, abs=1e-5)

    def test_output_in_range(self):
        score_fn = BinaryDiceScore()
        inputs = torch.rand(1, 8, 8)
        targets = (torch.rand(1, 8, 8) > 0.5).float()
        score = score_fn(inputs, targets).item()
        assert 0.0 <= score <= 1.0


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
