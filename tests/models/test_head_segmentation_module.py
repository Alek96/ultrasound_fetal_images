"""Unit tests for HeadSegmentationLitModule (direct instantiation, no Trainer)."""

from __future__ import annotations

import functools
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from src.models.components.loss import BinaryDiceCrossEntropyLoss
from src.models.head_segmentation_module import HeadSegmentationLitModule

_B = 2
_H, _W = 55, 80


@pytest.fixture()
def lit_module() -> HeadSegmentationLitModule:
    lit_module = HeadSegmentationLitModule(
        model=functools.partial(torch.nn.Conv2d, in_channels=1, out_channels=1, kernel_size=1),  # type: ignore
        criterion=BinaryDiceCrossEntropyLoss,  # type: ignore
        lr=1e-3,
        optimizer=functools.partial(torch.optim.Adam),  # type: ignore
        scheduler=functools.partial(  # type: ignore
            torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=5
        ),
    )
    lit_module.log = MagicMock()
    return lit_module


@pytest.fixture()
def lit_module_no_scheduler() -> HeadSegmentationLitModule:
    return HeadSegmentationLitModule(
        model=functools.partial(torch.nn.Conv2d, in_channels=1, out_channels=1, kernel_size=1),  # type: ignore
        criterion=BinaryDiceCrossEntropyLoss,  # type: ignore
        lr=1e-3,
        optimizer=functools.partial(torch.optim.Adam),  # type: ignore
        scheduler=None,
    )


def _make_batch(batch_size: int = _B, h: int = _H, w: int = _W) -> tuple[Tensor, Tensor, Tensor]:
    images = torch.rand(batch_size, 1, h, w)
    masks = torch.randint(0, 2, (batch_size, 1, h, w)).float()
    labels = torch.randint(0, 2, (batch_size,)).int()
    return images, masks, labels


def _set_model_bias(module: HeadSegmentationLitModule, bias: float) -> None:
    """Force the tiny Conv2d to output a constant value regardless of input."""
    with torch.no_grad():
        module.model.weight.fill_(0.0)
        module.model.bias.fill_(bias)


def _reset_val_metrics(module: HeadSegmentationLitModule) -> None:
    """Reset per-epoch val metrics, simulating what Lightning does at epoch end."""
    for m in [
        module.val_loss,
        module.val_dice,
        module.val_label_f1,
        module.val_label_acc,
        module.val_pixel_f1,
        module.val_pixel_acc,
    ]:
        m.reset()


# ---------------------------------------------------------------------------
# calculate_prediction — static method, no fixture needed
# ---------------------------------------------------------------------------


class TestCalculatePrediction:
    def test_all_negative(self) -> None:
        logits = torch.full((2, 1, 10, 10), -10.0)  # [B, 1, H, W]
        prediction_mask, prediction_label = HeadSegmentationLitModule.calculate_prediction(logits)

        assert (prediction_label == 0).all()
        assert prediction_mask.max().item() < 0.1

    def test_all_positive(self) -> None:
        logits = torch.full((2, 1, 10, 10), 10.0)
        prediction_mask, prediction_label = HeadSegmentationLitModule.calculate_prediction(logits)

        assert (prediction_label == 1).all()
        assert prediction_mask.min().item() > 0.9

    def test_below_pixel_threshold(self) -> None:
        """99 of 10000 pixels positive (1%) is below the 5% threshold → label 0."""
        logits = torch.full((1, 1, 100, 100), -10.0)
        logits[0, 0, :99, 0] = 10.0
        print((logits / 10).sum())

        _, prediction_label = HeadSegmentationLitModule.calculate_prediction(logits)

        assert prediction_label.item() == 0

    def test_at_pixel_threshold(self) -> None:
        """100 of 10000 pixels positive (5%) is exactly at the threshold → label 1."""
        logits = torch.full((1, 1, 100, 100), -10.0)
        logits[0, 0, :100, 0] = 10.0

        _, prediction_label = HeadSegmentationLitModule.calculate_prediction(logits)

        assert prediction_label.item() == 1

    def test_below_confidence_threshold(self) -> None:
        """100 pixels are positive (clears 1% area gate), but their mean confidence

        is ~0.73 (below 75% threshold) → label 0.
        """
        # A logit of 1.0 gives a sigmoid value of ~0.73
        logits = torch.full((1, 1, 100, 100), -10.0)
        logits[0, 0, :100, 0] = 1.0

        _, prediction_label = HeadSegmentationLitModule.calculate_prediction(logits)

        # 1 (percentage) * 0 (confidence) = 0
        assert prediction_label.item() == 0

    def test_at_confidence_threshold(self) -> None:
        """100 pixels are positive (clears 1% area gate), and their mean confidence

        is ~0.75 (exactly at/above threshold) → label 1.
        """
        # A logit of 1.10 match a sigmoid value of ~0.75
        logits = torch.full((1, 1, 100, 100), -10.0)
        logits[0, 0, :100, 0] = 1.10

        _, prediction_label = HeadSegmentationLitModule.calculate_prediction(logits)

        # 1 (percentage) * 1 (confidence) = 1
        assert prediction_label.item() == 1

    def test_output_shapes(self) -> None:
        logits = torch.zeros(_B, 1, _H, _W)
        prediction_mask, prediction_label = HeadSegmentationLitModule.calculate_prediction(logits)

        assert prediction_mask.shape == (_B, 1, _H, _W)
        assert prediction_label.shape == (_B,)


# ---------------------------------------------------------------------------
# forward and model_step
# ---------------------------------------------------------------------------


def test_forward_output_shape(lit_module: HeadSegmentationLitModule) -> None:
    x = torch.randn(_B, 1, _H, _W)
    out = lit_module(x)
    assert out.shape == (_B, 1, _H, _W)


def test_model_step_returns_six_items(lit_module: HeadSegmentationLitModule) -> None:
    batch = _make_batch()
    result = lit_module.model_step(batch)

    assert len(result) == 6

    loss, logits, prediction_mask, masks, prediction_label, labels = result
    assert loss.shape == ()
    assert logits.shape == (_B, 1, _H, _W)
    assert prediction_mask.shape == (_B, 1, _H, _W)
    assert prediction_label.shape == (_B,)
    assert masks.shape == (_B, 1, _H, _W)
    assert labels.shape == (_B,)


def test_model_step_loss_is_greater_or_equal_0(lit_module: HeadSegmentationLitModule) -> None:
    batch = _make_batch()
    loss, *_ = lit_module.model_step(batch)

    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# training / validation / test steps
# ---------------------------------------------------------------------------


def test_training_step_returns_loss(lit_module: HeadSegmentationLitModule) -> None:
    batch = _make_batch()
    loss = lit_module.training_step(batch, 0)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()


def test_validation_step_returns_loss(lit_module: HeadSegmentationLitModule) -> None:
    batch = _make_batch()
    loss = lit_module.validation_step(batch, 0)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()


def test_test_step_returns_loss(lit_module: HeadSegmentationLitModule) -> None:
    batch = _make_batch()
    loss = lit_module.test_step(batch, 0)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


def test_configure_optimizers_with_scheduler(lit_module: HeadSegmentationLitModule) -> None:
    result = lit_module.configure_optimizers()

    assert "optimizer" in result
    assert isinstance(result["optimizer"], torch.optim.Adam)
    assert "lr_scheduler" in result
    assert "scheduler" in result["lr_scheduler"]
    assert isinstance(result["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_configure_optimizers_without_scheduler(lit_module_no_scheduler: HeadSegmentationLitModule) -> None:
    result = lit_module_no_scheduler.configure_optimizers()

    assert "optimizer" in result
    assert isinstance(result["optimizer"], torch.optim.Adam)
    assert "lr_scheduler" not in result


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Metric accumulation, correctness, best-value tracking, and confusion matrix.

    Naming convention for fixtures
    ──────────────────────────────
    lit_module_one  – model that always predicts the positive class  (bias = +10 → sigmoid ≈ 1)
    lit_module_zero – model that always predicts the negative class  (bias = -10 → sigmoid ≈ 0)
    batch_one       – all-positive masks  / all-positive labels
    batch_zero      – all-zero    masks  / all-zero    labels
    batch_mixed     – first sample: mask=all-one, label=1
                      second sample: mask=all-zero, label=0
                      (with lit_module_one this creates 1 TP + 1 FP, making
                       label_acc = 0.5 while label_f1 = 2/3 — two clearly
                       distinct values that verify the metrics are distinct)
    """

    @pytest.fixture()
    def lit_module_one(self, lit_module: HeadSegmentationLitModule) -> HeadSegmentationLitModule:
        """A model that always predicts the positive class (sigmoid ≈ 1 → label 1)."""
        _set_model_bias(lit_module, 10.0)
        return lit_module

    @pytest.fixture()
    def lit_module_zero(self, lit_module: HeadSegmentationLitModule) -> HeadSegmentationLitModule:
        """A model that always predicts the negative class (sigmoid ≈ 0 → label 0)."""
        _set_model_bias(lit_module, -10.0)
        return lit_module

    @pytest.fixture()
    def batch_one(self):
        """All-one masks and all-one labels (every sample is positive)."""
        return (
            torch.rand(_B, 1, _H, _W),
            torch.ones(_B, 1, _H, _W),
            torch.ones(_B, dtype=torch.int32),
        )

    @pytest.fixture()
    def batch_zero(self):
        """All-zero masks and all-zero labels (every sample is negative)."""
        return (
            torch.rand(_B, 1, _H, _W),
            torch.zeros(_B, 1, _H, _W),
            torch.zeros(_B, dtype=torch.int32),
        )

    @pytest.fixture()
    def batch_mixed(self):
        """First sample: mask=all-one, label=1. Second sample: mask=all-zero, label=0."""
        masks = torch.zeros(_B, 1, _H, _W)
        masks[0] = 1.0
        labels = torch.zeros(_B, dtype=torch.int32)
        labels[0] = 1
        return torch.rand(_B, 1, _H, _W), masks, labels

    # ------------------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------------------

    class TestTrainMetrics:

        # ------------------------------------------------------------------
        # train_loss
        # ------------------------------------------------------------------

        class TestTrainLoss:
            def test_matches_model_step(
                self,
                lit_module: HeadSegmentationLitModule,
            ) -> None:
                """MeanMetric-backed train_loss should equal the loss returned by model_step."""
                batch = _make_batch()
                reference_loss, *_ = lit_module.model_step(batch)
                lit_module.training_step(batch, 0)

                assert lit_module.train_loss.compute().item() == pytest.approx(reference_loss.item(), rel=1e-4)

        # ------------------------------------------------------------------
        # train_dice
        # ------------------------------------------------------------------

        class TestTrainDice:
            def test_is_one_when_perfect_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → train_dice ≈ 1.0."""
                lit_module_one.training_step(batch_one, 0)

                assert lit_module_one.train_dice.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_no_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → train_dice ≈ 0.0."""
                lit_module_one.training_step(batch_zero, 0)

                assert lit_module_one.train_dice.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_measures_spatial_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, first mask=1 and second mask=0.

                `dice = 2*|A∩B| / (|A|+|B|) = 2*(H*W) / (2*(H*W) + H*W) = 2/3`.
                """
                lit_module_one.training_step(batch_mixed, 0)

                expected = 2 * _H * _W / (2 * _H * _W + _H * _W)  # 2/3
                assert lit_module_one.train_dice.compute().item() == pytest.approx(expected, abs=1e-4)

        # ------------------------------------------------------------------
        # train_label_acc
        # ------------------------------------------------------------------

        class TestTrainLabelAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive labels → train_label_acc = 1.0."""
                lit_module_one.training_step(batch_one, 0)

                assert lit_module_one.train_label_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Model predicts 1 for both samples; only label[0]=1 is correct → acc = 0.5."""
                lit_module_one.training_step(batch_mixed, 0)

                assert lit_module_one.train_label_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Model predicts 1 for all; all labels are 0 → train_label_acc = 0.0."""
                lit_module_one.training_step(batch_zero, 0)

                assert lit_module_one.train_label_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

        # ------------------------------------------------------------------
        # train_label_f1
        # ------------------------------------------------------------------

        class TestTrainLabelF1:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive labels → train_label_f1 = 1.0."""
                lit_module_one.training_step(batch_one, 0)

                assert lit_module_one.train_label_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_only_true_negatives(
                self,
                lit_module_zero: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All predictions=0 and all labels=0 → binary F1=0.0 (no positive-class activity).

                This is the clearest demonstration that accuracy and F1 are distinct: the same
                outcome (all-TN) gives acc=1.0 but F1=0.0.
                """
                lit_module_zero.training_step(batch_zero, 0)

                assert lit_module_zero.train_label_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_weighs_precision_and_recall(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Same mixed batch: TP=1 FP=1 FN=0 → F1=2/3, which differs from acc=0.5."""
                lit_module_one.training_step(batch_mixed, 0)

                # precision = 1/(1+1) = 0.5, recall = 1/(1+0) = 1.0 → F1 = 2*(0.5*1)/(0.5+1) = 2/3
                assert lit_module_one.train_label_f1.compute().item() == pytest.approx(2 / 3, abs=1e-4)

        # ------------------------------------------------------------------
        # train_pixel_acc
        # ------------------------------------------------------------------

        class TestTrainPixelAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → train_pixel_acc = 1.0."""
                lit_module_one.training_step(batch_one, 0)

                assert lit_module_one.train_pixel_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → train_pixel_acc = 0.0."""
                lit_module_one.training_step(batch_zero, 0)

                assert lit_module_one.train_pixel_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, half-positive mask → pixel_acc = 0.5."""
                lit_module_one.training_step(batch_mixed, 0)

                assert lit_module_one.train_pixel_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

        # ------------------------------------------------------------------
        # train_pixel_f1
        # ------------------------------------------------------------------

        class TestTrainPixelF1:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → train_pixel_f1 = 1.0."""
                lit_module_one.training_step(batch_one, 0)

                assert lit_module_one.train_pixel_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → train_pixel_f1 = 0.0."""
                lit_module_one.training_step(batch_zero, 0)

                assert lit_module_one.train_pixel_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_weighs_precision_and_recall(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, half-positive mask: pixel F1 (2/3) differs from acc (0.5)."""
                lit_module_one.training_step(batch_mixed, 0)

                # TP=H*W, FP=H*W, FN=0 → precision=0.5, recall=1.0 → F1=2/3
                assert lit_module_one.train_pixel_f1.compute().item() == pytest.approx(2 / 3, abs=1e-4)

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------

    class TestValMetrics:

        # ------------------------------------------------------------------
        # val_loss
        # ------------------------------------------------------------------

        class TestValLoss:
            def test_matches_model_step(
                self,
                lit_module: HeadSegmentationLitModule,
            ) -> None:
                """MeanMetric-backed val_loss should equal the loss returned by model_step."""
                batch = _make_batch()
                reference_loss, *_ = lit_module.model_step(batch)
                lit_module.validation_step(batch, 0)

                assert lit_module.val_loss.compute().item() == pytest.approx(reference_loss.item(), rel=1e-4)

        # ------------------------------------------------------------------
        # val_dice / val_dice_best
        # ------------------------------------------------------------------

        class TestValDice:
            def test_is_one_when_perfect_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → val_dice ≈ 1.0."""
                lit_module_one.validation_step(batch_one, 0)

                assert lit_module_one.val_dice.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_no_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → val_dice ≈ 0.0."""
                lit_module_one.validation_step(batch_zero, 0)

                assert lit_module_one.val_dice.compute().item() == pytest.approx(0.0, abs=1e-3)

            def test_measures_spatial_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, first mask=1 and second mask=0 → val_dice = 2/3."""
                lit_module_one.validation_step(batch_mixed, 0)

                expected = 2 * _H * _W / (2 * _H * _W + _H * _W)  # 2/3
                assert lit_module_one.val_dice.compute().item() == pytest.approx(expected, abs=1e-4)

            def test_best_increases_with_better_epoch(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_dice_best should increase when a better epoch follows a worse one."""
                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_dice_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_dice_best.compute().item() > best_after_epoch1

            def test_best_does_not_decrease(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_dice_best should not decrease when a worse epoch follows a better one."""
                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_dice_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_dice_best.compute().item() == pytest.approx(best_after_epoch1)

        # ------------------------------------------------------------------
        # val_label_acc / val_label_acc_best
        # ------------------------------------------------------------------

        class TestValLabelAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive labels → val_label_acc = 1.0."""
                lit_module_one.validation_step(batch_one, 0)

                assert lit_module_one.val_label_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Model predicts 1 for all; all labels are 0 → val_label_acc = 0.0."""
                lit_module_one.validation_step(batch_zero, 0)

                assert lit_module_one.val_label_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Model predicts 1 for both samples; only label[0]=1 is correct → val_label_acc = 0.5."""
                lit_module_one.validation_step(batch_mixed, 0)

                assert lit_module_one.val_label_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

            def test_best_increases_with_better_epoch(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_label_acc_best should increase when a better epoch follows a worse one."""
                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_label_acc_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_label_acc_best.compute().item() > best_after_epoch1

            def test_best_tracks_maximum(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_label_acc_best retains the epoch-high accuracy and does not decrease."""
                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_label_acc_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_label_acc_best.compute().item() == pytest.approx(best_after_epoch1)

        # ------------------------------------------------------------------
        # val_label_f1 / val_label_f1_best
        # ------------------------------------------------------------------

        class TestValLabelF1:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive labels → val_label_f1 = 1.0."""
                lit_module_one.validation_step(batch_one, 0)

                assert lit_module_one.val_label_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_only_true_negatives(
                self,
                lit_module_zero: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All predictions=0 and all labels=0 → binary val_label_f1=0.0."""
                lit_module_zero.validation_step(batch_zero, 0)

                assert lit_module_zero.val_label_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_weighs_precision_and_recall(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Same mixed batch: TP=1 FP=1 FN=0 → val_label_f1=2/3, differs from acc=0.5."""
                lit_module_one.validation_step(batch_mixed, 0)

                assert lit_module_one.val_label_f1.compute().item() == pytest.approx(2 / 3, abs=1e-4)

            def test_best_increases_with_better_epoch(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_label_f1_best should increase when a better epoch follows a worse one."""
                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_label_f1_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_label_f1_best.compute().item() > best_after_epoch1

            def test_best_tracks_maximum(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_label_f1_best retains the epoch-high F1 and does not decrease."""
                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_label_f1_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_label_f1_best.compute().item() == pytest.approx(best_after_epoch1)

        # ------------------------------------------------------------------
        # val_pixel_acc / val_pixel_acc_best
        # ------------------------------------------------------------------

        class TestValPixelAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → val_pixel_acc = 1.0."""
                lit_module_one.validation_step(batch_one, 0)

                assert lit_module_one.val_pixel_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → val_pixel_acc = 0.0."""
                lit_module_one.validation_step(batch_zero, 0)

                assert lit_module_one.val_pixel_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, half-positive mask → val_pixel_acc = 0.5."""
                lit_module_one.validation_step(batch_mixed, 0)

                assert lit_module_one.val_pixel_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

            def test_best_increases_with_better_epoch(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_pixel_acc_best should increase when a better epoch follows a worse one."""
                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_pixel_acc_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_pixel_acc_best.compute().item() > best_after_epoch1

            def test_best_tracks_maximum(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_pixel_acc_best retains the epoch-high pixel accuracy and does not decrease."""
                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_pixel_acc_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_pixel_acc_best.compute().item() == pytest.approx(best_after_epoch1)

        # ------------------------------------------------------------------
        # val_pixel_f1 / val_pixel_f1_best
        # ------------------------------------------------------------------

        class TestValPixelF1:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → val_pixel_f1 = 1.0."""
                lit_module_one.validation_step(batch_one, 0)

                assert lit_module_one.val_pixel_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → val_pixel_f1 = 0.0."""
                lit_module_one.validation_step(batch_zero, 0)

                assert lit_module_one.val_pixel_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_weighs_precision_and_recall(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, half-positive mask: val_pixel_f1 (2/3) differs from acc (0.5)."""
                lit_module_one.validation_step(batch_mixed, 0)

                assert lit_module_one.val_pixel_f1.compute().item() == pytest.approx(2 / 3, abs=1e-4)

            def test_best_increases_with_better_epoch(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_pixel_f1_best should increase when a better epoch follows a worse one."""
                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_pixel_f1_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_pixel_f1_best.compute().item() > best_after_epoch1

            def test_best_tracks_maximum(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """val_pixel_f1_best retains the epoch-high pixel F1 and does not decrease."""
                lit_module_one.validation_step(batch_one, 0)
                lit_module_one.on_validation_epoch_end()
                best_after_epoch1 = lit_module_one.val_pixel_f1_best.compute().item()

                _reset_val_metrics(lit_module_one)

                lit_module_one.validation_step(batch_zero, 0)
                lit_module_one.on_validation_epoch_end()

                assert lit_module_one.val_pixel_f1_best.compute().item() == pytest.approx(best_after_epoch1)

    # ------------------------------------------------------------------
    # Test metrics
    # ------------------------------------------------------------------

    class TestTestMetrics:

        # ------------------------------------------------------------------
        # test_loss
        # ------------------------------------------------------------------

        class TestTestLoss:
            def test_matches_model_step(
                self,
                lit_module: HeadSegmentationLitModule,
            ) -> None:
                """MeanMetric-backed test_loss should equal the loss returned by model_step."""
                batch = _make_batch()
                reference_loss, *_ = lit_module.model_step(batch)
                lit_module.test_step(batch, 0)

                assert lit_module.test_loss.compute().item() == pytest.approx(reference_loss.item(), rel=1e-4)

        # ------------------------------------------------------------------
        # test_dice
        # ------------------------------------------------------------------

        class TestTestDice:
            def test_is_one_when_perfect_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → test_dice ≈ 1.0."""
                lit_module_one.test_step(batch_one, 0)

                assert lit_module_one.test_dice.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_no_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → test_dice ≈ 0.0."""
                lit_module_one.test_step(batch_zero, 0)

                assert lit_module_one.test_dice.compute().item() == pytest.approx(0.0, abs=1e-3)

            def test_measures_spatial_overlap(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, first mask=1 and second mask=0 → test_dice = 2/3."""
                lit_module_one.test_step(batch_mixed, 0)

                expected = 2 * _H * _W / (2 * _H * _W + _H * _W)  # 2/3
                assert lit_module_one.test_dice.compute().item() == pytest.approx(expected, abs=1e-4)

        # ------------------------------------------------------------------
        # test_label_acc
        # ------------------------------------------------------------------

        class TestTestLabelAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive labels → test_label_acc = 1.0."""
                lit_module_one.test_step(batch_one, 0)

                assert lit_module_one.test_label_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Model predicts 1 for all; all labels are 0 → test_label_acc = 0.0."""
                lit_module_one.test_step(batch_zero, 0)

                assert lit_module_one.test_label_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Model predicts 1 for both samples; only label[0]=1 is correct → test_label_acc = 0.5."""
                lit_module_one.test_step(batch_mixed, 0)

                assert lit_module_one.test_label_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

        # ------------------------------------------------------------------
        # test_label_f1
        # ------------------------------------------------------------------

        class TestTestLabelF1:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive labels → test_label_f1 = 1.0."""
                lit_module_one.test_step(batch_one, 0)

                assert lit_module_one.test_label_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_only_true_negatives(
                self,
                lit_module_zero: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All predictions=0 and all labels=0 → binary test_label_f1=0.0.

                Paired with test_is_one_when_all_true_negatives this confirms that
                accuracy and F1 diverge on all-negative outcomes.
                """
                lit_module_zero.test_step(batch_zero, 0)

                assert lit_module_zero.test_label_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_weighs_precision_and_recall(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """Same mixed batch: TP=1 FP=1 FN=0 → test_label_f1=2/3, differs from acc=0.5."""
                lit_module_one.test_step(batch_mixed, 0)

                assert lit_module_one.test_label_f1.compute().item() == pytest.approx(2 / 3, abs=1e-4)

        # ------------------------------------------------------------------
        # test_pixel_acc
        # ------------------------------------------------------------------

        class TestTestPixelAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → test_pixel_acc = 1.0."""
                lit_module_one.test_step(batch_one, 0)

                assert lit_module_one.test_pixel_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → test_pixel_acc = 0.0."""
                lit_module_one.test_step(batch_zero, 0)

                assert lit_module_one.test_pixel_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, half-positive mask → test_pixel_acc = 0.5."""
                lit_module_one.test_step(batch_mixed, 0)

                assert lit_module_one.test_pixel_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

        # ------------------------------------------------------------------
        # test_pixel_f1
        # ------------------------------------------------------------------

        class TestTestPixelF1:
            def test_is_one_when_perfect(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive masks → test_pixel_f1 = 1.0."""
                lit_module_one.test_step(batch_one, 0)

                assert lit_module_one.test_pixel_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-zero masks → test_pixel_f1 = 0.0."""
                lit_module_one.test_step(batch_zero, 0)

                assert lit_module_one.test_pixel_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_weighs_precision_and_recall(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_mixed: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions, half-positive mask: test_pixel_f1 (2/3) differs from acc (0.5)."""
                lit_module_one.test_step(batch_mixed, 0)

                # TP=H*W, FP=H*W, FN=0 → precision=0.5, recall=1.0 → F1=2/3
                assert lit_module_one.test_pixel_f1.compute().item() == pytest.approx(2 / 3, abs=1e-4)

        # ------------------------------------------------------------------
        # test_label_cm
        # ------------------------------------------------------------------

        class TestTestLabelCm:
            def test_all_true_positive(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-positive labels → only TP cell populated."""
                lit_module_one.on_test_start()
                lit_module_one.test_step(batch_one, 0)

                cm = lit_module_one.test_label_cm.compute()  # layout: [[TN, FP], [FN, TP]]
                assert cm[0, 0].item() == 0  # TN
                assert cm[0, 1].item() == 0  # FP
                assert cm[1, 0].item() == 0  # FN
                assert cm[1, 1].item() == _B  # TP

            def test_all_true_negative(
                self,
                lit_module_zero: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-negative predictions on all-negative labels → only TN cell populated."""
                lit_module_zero.on_test_start()
                lit_module_zero.test_step(batch_zero, 0)

                cm = lit_module_zero.test_label_cm.compute()  # layout: [[TN, FP], [FN, TP]]
                assert cm[0, 0].item() == _B  # TN
                assert cm[0, 1].item() == 0  # FP
                assert cm[1, 0].item() == 0  # FN
                assert cm[1, 1].item() == 0  # TP

            def test_all_false_positive(
                self,
                lit_module_one: HeadSegmentationLitModule,
                batch_zero: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-positive predictions on all-negative labels → only FP cell populated."""
                lit_module_one.on_test_start()
                lit_module_one.test_step(batch_zero, 0)

                cm = lit_module_one.test_label_cm.compute()  # layout: [[TN, FP], [FN, TP]]
                assert cm[0, 0].item() == 0  # TN
                assert cm[0, 1].item() == _B  # FP
                assert cm[1, 0].item() == 0  # FN
                assert cm[1, 1].item() == 0  # TP

            def test_all_false_negative(
                self,
                lit_module_zero: HeadSegmentationLitModule,
                batch_one: tuple[Tensor, Tensor, Tensor],
            ) -> None:
                """All-negative predictions on all-positive labels → only FN cell populated."""
                lit_module_zero.on_test_start()
                lit_module_zero.test_step(batch_one, 0)

                cm = lit_module_zero.test_label_cm.compute()  # layout: [[TN, FP], [FN, TP]]
                assert cm[0, 0].item() == 0  # TN
                assert cm[0, 1].item() == 0  # FP
                assert cm[1, 0].item() == _B  # FN
                assert cm[1, 1].item() == 0  # TP


# ---------------------------------------------------------------------------
# epoch hooks
# ---------------------------------------------------------------------------


def test_on_train_start_resets_best_metrics(lit_module: HeadSegmentationLitModule) -> None:
    # Run a validation step to accumulate some non-zero best values
    batch = _make_batch()
    lit_module.validation_step(batch, 0)
    lit_module.on_validation_epoch_end()

    # Now simulate a new train start — best metrics should be reset.
    # MaxMetric.compute() returns -inf when no value has been accumulated yet.
    lit_module.on_train_start()

    assert lit_module.val_dice_best.compute().item() == float("-inf")
    assert lit_module.val_label_f1_best.compute().item() == float("-inf")
    assert lit_module.val_label_acc_best.compute().item() == float("-inf")
    assert lit_module.val_pixel_f1_best.compute().item() == float("-inf")
    assert lit_module.val_pixel_acc_best.compute().item() == float("-inf")
