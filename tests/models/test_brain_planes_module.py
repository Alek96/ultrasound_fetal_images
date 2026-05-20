"""Unit and integration tests for BrainPlanesLitModule (direct instantiation, no Trainer)."""

from __future__ import annotations

import functools
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import ConfusionMatrix, Metric

from src.models.brain_planes import BrainPlanesLitModule

_NUM_CLASSES = 4
_B = 2
_H, _W = 32, 32
_C = 3


# ---------------------------------------------------------------------------
# Minimal two-head network fixture
# ---------------------------------------------------------------------------


class _TwoOutputNet(torch.nn.Module):
    """Returns (x, fc(x_flat)) — same (features, logits) interface as real backbones."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(_C * _H * _W, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.fc(x.flatten(1))
        return x, logits


def _make_module(
    softmax_target: bool = False,
    vta_transforms: dict = {"horizontal_flips": [False]},
    tta_transforms: dict = {"horizontal_flips": [False]},
    reduction: str = "mean",
    with_scheduler: bool = True,
) -> BrainPlanesLitModule:
    with patch("src.models.brain_planes.get_model", return_value=_TwoOutputNet(_NUM_CLASSES)):
        module = BrainPlanesLitModule(
            net_spec={},
            num_classes=_NUM_CLASSES,
            softmax_target=softmax_target,
            vta_transforms=vta_transforms,
            tta_transforms=tta_transforms,
            criterion=functools.partial(torch.nn.CrossEntropyLoss, reduction=reduction),  # type: ignore
            lr=1e-3,
            optimizer=functools.partial(torch.optim.Adam),  # type: ignore
            scheduler=(  # type: ignore
                functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=5)
                if with_scheduler
                else None
            ),
        )
    module.log = MagicMock()
    return module


@pytest.fixture()
def lit_module() -> BrainPlanesLitModule:
    return _make_module()


@pytest.fixture()
def lit_module_no_scheduler() -> BrainPlanesLitModule:
    return _make_module(with_scheduler=False)


def _make_batch(batch_size: int = _B) -> tuple[Tensor, Tensor]:
    images = torch.rand(batch_size, _C, _H, _W)
    labels = torch.randint(0, _NUM_CLASSES, (batch_size,))
    return images, labels


def _make_batch_2d_one_hot_label(batch_size: int = _B) -> tuple[Tensor, Tensor]:
    images = torch.rand(batch_size, _C, _H, _W)
    labels = torch.randint(0, _NUM_CLASSES, (batch_size,))
    labels = F.one_hot(labels, num_classes=_NUM_CLASSES).float()
    return images, labels


def _make_batch_softmax_target_label(batch_size: int = _B) -> tuple[Tensor, Tensor]:
    images = torch.rand(batch_size, _C, _H, _W)
    labels = torch.empty((batch_size, 4), dtype=torch.float32)
    # y columns: [y_a, y_b, lam_a, lam_b]
    labels[:, 0] = torch.randint(0, _NUM_CLASSES, (batch_size,)).float()  # y_a
    labels[:, 1] = torch.randint(0, _NUM_CLASSES, (batch_size,)).float()  # y_b
    labels[:, 2] = 0.6  # lam_a
    labels[:, 3] = 0.4  # lam_b
    return images, labels


def _set_model_output(module: BrainPlanesLitModule, class_idx: int) -> None:
    """Force the net to always predict *class_idx* with high confidence."""
    with torch.no_grad():
        module.net.fc.weight.fill_(0.0)
        module.net.fc.bias.fill_(-100.0)
        module.net.fc.bias[class_idx] = 100.0


def _reset_val_metrics(module: BrainPlanesLitModule) -> None:
    """Reset per-epoch val metrics, simulating what Lightning does at epoch end."""
    for m in [
        module.val_loss,
        module.val_base_acc,
        module.val_base_acc_cm,
        module.val_base_f1,
        module.val_tta_acc,
        module.val_tta_acc_cm,
        module.val_tta_f1,
    ]:
        m.reset()


# ---------------------------------------------------------------------------
# create_transforms
# ---------------------------------------------------------------------------


class TestCreateTransforms:
    def test_empty_dict_yields_one_identity_transform(self):
        """An empty config dict must fall back to one identity transform."""
        transforms = BrainPlanesLitModule.create_transforms({})
        assert len(transforms) == 1

    def test_identity_transform_produces_one_entry(self):
        """A single False/no-op per axis must yield exactly one transform."""
        transforms = BrainPlanesLitModule.create_transforms({"horizontal_flips": [False]})
        assert len(transforms) == 1

    def test_product_of_two_h_flip_values_gives_two_transforms(self):
        """[True, False] on one axis should double the transform count."""
        transforms = BrainPlanesLitModule.create_transforms({"horizontal_flips": [True, False]})
        assert len(transforms) == 2

    def test_product_is_computed_across_all_axes(self):
        """2 h-flips × 3 rotations = 6 transforms."""
        transforms = BrainPlanesLitModule.create_transforms(
            {
                "horizontal_flips": [True, False],
                "rotate_degrees": [0.0, 10.0, -10.0],
            }
        )
        assert len(transforms) == 6


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_returns_tuple_of_two_tensors(self, lit_module: BrainPlanesLitModule) -> None:
        x = torch.rand(_B, _C, _H, _W)
        features, logits = lit_module(x)
        assert isinstance(features, Tensor)
        assert isinstance(logits, Tensor)

    def test_logits_shape(self, lit_module: BrainPlanesLitModule) -> None:
        x = torch.rand(_B, _C, _H, _W)
        _, logits = lit_module(x)
        assert logits.shape == (_B, _NUM_CLASSES)


# ---------------------------------------------------------------------------
# forward_tta
# ---------------------------------------------------------------------------


class TestForwardTta:
    @pytest.fixture()
    def transforms(self) -> list[Callable]:
        return BrainPlanesLitModule.create_transforms({"horizontal_flips": [True, False]})

    def test_returns_tuple_of_two_tensors(self, lit_module: BrainPlanesLitModule, transforms) -> None:
        x = torch.rand(_B, _C, _H, _W)
        features, logits = lit_module.forward_tta(x, transforms)
        assert isinstance(features, Tensor)
        assert isinstance(logits, Tensor)

    def test_logits_shape(self, lit_module: BrainPlanesLitModule, transforms) -> None:
        x = torch.rand(_B, _C, _H, _W)
        _, logits = lit_module.forward_tta(x, transforms)
        assert logits.shape == (_B, _NUM_CLASSES)


# ---------------------------------------------------------------------------
# model_step
# ---------------------------------------------------------------------------


class TestModelStep:
    @pytest.mark.parametrize(
        "module, batch",
        [
            (_make_module(), _make_batch()),
            (_make_module(softmax_target=False, reduction="none"), _make_batch_2d_one_hot_label()),
            (_make_module(softmax_target=True, reduction="none"), _make_batch_softmax_target_label()),
        ],
    )
    def test_returns_three_items(self, module: BrainPlanesLitModule, batch) -> None:
        result = module.model_step(batch)

        assert len(result) == 3

        loss, preds, targets = result
        assert loss.shape == ()
        assert preds.shape == (_B,)
        assert targets.shape == (_B,)


# ---------------------------------------------------------------------------
# model_tta_step
# ---------------------------------------------------------------------------


class TestModelTtaStep:
    @pytest.fixture()
    def transforms(self) -> list[Callable]:
        return BrainPlanesLitModule.create_transforms({"horizontal_flips": [True, False]})

    @pytest.mark.parametrize(
        "module, batch",
        [
            (_make_module(), _make_batch()),
            (_make_module(softmax_target=False, reduction="none"), _make_batch_2d_one_hot_label()),
            (_make_module(softmax_target=True, reduction="none"), _make_batch_softmax_target_label()),
        ],
    )
    def test_returns_three_items(self, module: BrainPlanesLitModule, batch, transforms) -> None:
        result = module.model_tta_step(batch, transforms)

        assert len(result) == 4

        loss, preds, tta_preds, targets = result
        assert loss.shape == ()
        assert preds.shape == (_B,)
        assert tta_preds.shape == (_B,)
        assert targets.shape == (_B,)


# ---------------------------------------------------------------------------
# training / validation / test steps
# ---------------------------------------------------------------------------


def test_training_step_returns_dict_with_loss(lit_module: BrainPlanesLitModule) -> None:
    batch = _make_batch()
    result = lit_module.training_step(batch, 0)

    assert "loss" in result
    assert result["loss"].shape == ()


def test_validation_step_returns_dict_with_loss(lit_module: BrainPlanesLitModule) -> None:
    batch = _make_batch()
    result = lit_module.validation_step(batch, 0)

    assert "loss" in result
    assert result["loss"].shape == ()


def test_test_step_returns_dict_with_loss(lit_module: BrainPlanesLitModule) -> None:
    batch = _make_batch()
    result = lit_module.test_step(batch, 0)

    assert "loss" in result
    assert result["loss"].shape == ()


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


def test_configure_optimizers_with_scheduler(lit_module: BrainPlanesLitModule) -> None:
    result = lit_module.configure_optimizers()

    assert "optimizer" in result
    assert isinstance(result["optimizer"], torch.optim.Adam)
    assert "lr_scheduler" in result
    assert "scheduler" in result["lr_scheduler"]
    assert isinstance(result["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_configure_optimizers_without_scheduler(lit_module_no_scheduler: BrainPlanesLitModule) -> None:
    result = lit_module_no_scheduler.configure_optimizers()

    assert "optimizer" in result
    assert isinstance(result["optimizer"], torch.optim.Adam)
    assert "lr_scheduler" not in result


# ---------------------------------------------------------------------------
# confusion_matrix_acc and brain_acc (pure math)
# ---------------------------------------------------------------------------


class TestConfusionMatrixAcc:
    @pytest.fixture()
    def cm_metric(self) -> Metric:
        return ConfusionMatrix(task="multiclass", num_classes=_NUM_CLASSES, normalize="true")

    def test_perfect_acc_returns_one(self, cm_metric) -> None:
        """Sum of diagonal / n for a perfect normalised CM must be 1.0."""
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        cm_metric.update(preds, targets)

        cm = cm_metric.compute()
        result = BrainPlanesLitModule.confusion_matrix_acc(cm, list(range(_NUM_CLASSES)))
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_zero_acc_returns_zero(self, cm_metric) -> None:
        preds = torch.tensor([1, 2, 3, 0])
        targets = torch.tensor([0, 1, 2, 3])
        cm_metric.update(preds, targets)

        cm = cm_metric.compute()
        result = BrainPlanesLitModule.confusion_matrix_acc(cm, list(range(_NUM_CLASSES)))
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_halh_acc_returns_zero(self, cm_metric) -> None:
        """Only the requested class indices should contribute to the average."""
        preds = torch.tensor([0, 0, 2, 2])
        targets = torch.tensor([0, 1, 2, 3])
        cm_metric.update(preds, targets)

        cm = cm_metric.compute()
        result = BrainPlanesLitModule.confusion_matrix_acc(cm, list(range(_NUM_CLASSES)))
        assert result.item() == pytest.approx(0.5, abs=1e-6)


class TestBrainAcc:
    @pytest.fixture()
    def cm_metric(self) -> Metric:
        return ConfusionMatrix(task="multiclass", num_classes=_NUM_CLASSES, normalize="true")

    def test_perfect_acc_returns_one(self, lit_module: BrainPlanesLitModule, cm_metric) -> None:
        """Sum of diagonal / n for a perfect normalised CM must be 1.0."""
        preds = torch.tensor([0, 1, 2, 0])
        targets = torch.tensor([0, 1, 2, 3])
        cm_metric.update(preds, targets)

        result = lit_module.brain_acc(cm_metric)
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_zero_acc_returns_zero(self, lit_module: BrainPlanesLitModule, cm_metric) -> None:
        preds = torch.tensor([1, 2, 3, 3])
        targets = torch.tensor([0, 1, 2, 3])
        cm_metric.update(preds, targets)

        result = lit_module.brain_acc(cm_metric)
        assert result.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Metric accumulation and correctness tests.

    Fixture convention
    ──────────────────
    lit_module_cls0 – model that always predicts class 0 (bias trick)
    batch_cls0      – all images labelled as class 0
    batch_cls1      – all images labelled as class 1
    batch_mixed     – first sample class 0, second sample class 1
    """

    @pytest.fixture()
    def lit_module_cls0(self, lit_module: BrainPlanesLitModule) -> BrainPlanesLitModule:
        """A model that always outputs class 0 with high confidence."""
        _set_model_output(lit_module, 0)
        return lit_module

    @pytest.fixture()
    def batch_cls0(self) -> tuple[Tensor, Tensor]:
        return torch.rand(_B, _C, _H, _W), torch.zeros(_B, dtype=torch.long)

    @pytest.fixture()
    def batch_cls1(self) -> tuple[Tensor, Tensor]:
        return torch.rand(_B, _C, _H, _W), torch.ones(_B, dtype=torch.long)

    @pytest.fixture()
    def batch_mixed(self) -> tuple[Tensor, Tensor]:
        """First sample: class 0, second sample: class 1."""
        images = torch.rand(_B, _C, _H, _W)
        labels = torch.tensor([0, 1], dtype=torch.long)
        return images, labels

    # ------------------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------------------

    class TestTrainMetrics:

        # ------------------------------------------------------------------
        # train_loss
        # ------------------------------------------------------------------

        class TestTrainLoss:
            def test_matches_model_step(self, lit_module: BrainPlanesLitModule) -> None:
                """MeanMetric-backed train_loss should equal the loss from model_step."""
                batch = _make_batch()
                ref_loss, _, _ = lit_module.model_step(batch)
                lit_module.training_step(batch, 0)
                assert lit_module.train_loss.compute().item() == pytest.approx(ref_loss.item(), rel=1e-4)

        # ------------------------------------------------------------------
        # train_acc
        # ------------------------------------------------------------------

        class TestTrainAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → train_acc = 1.0."""
                lit_module_cls0.training_step(batch_cls0, 0)
                assert lit_module_cls0.train_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → train_acc = 0.0."""
                lit_module_cls0.training_step(batch_cls1, 0)
                assert lit_module_cls0.train_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for all; only the first sample is correct → train_acc = 0.5."""
                lit_module_cls0.training_step(batch_mixed, 0)
                assert lit_module_cls0.train_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

        # ------------------------------------------------------------------
        # train_f1
        # ------------------------------------------------------------------

        class TestTrainF1:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → macro F1 = 1.0."""
                lit_module_cls0.training_step(batch_cls0, 0)
                assert lit_module_cls0.train_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for all; all labels are class 1 → macro F1 = 0.0."""
                lit_module_cls0.training_step(batch_cls1, 0)
                assert lit_module_cls0.train_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_partial_when_mixed(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for both samples; labels are [0, 1].

                Class 0: TP=1, FP=1, FN=0 → precision=0.5, recall=1.0, F1=2/3.
                Class 1: TP=0, FP=0, FN=1 → F1=0.0.
                Classes 2 and 3: no support → excluded from macro denominator.
                Macro F1 = (2/3 + 0.0) / 2 = 1/3.
                """
                lit_module_cls0.training_step(batch_mixed, 0)
                assert lit_module_cls0.train_f1.compute().item() == pytest.approx(1 / 3, abs=1e-4)

        # ------------------------------------------------------------------
        # train_cm
        # ------------------------------------------------------------------

        class TestTrainCm:
            def test_has_shape_n_by_n(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """train_cm is a _NUM_CLASSES×_NUM_CLASSES matrix."""
                lit_module_cls0.training_step(batch_cls0, 0)
                cm = lit_module_cls0.train_cm.compute()
                assert cm.shape == (_NUM_CLASSES, _NUM_CLASSES)

            def test_has_dtype_long(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """normalize='none' → raw integer counts → dtype torch.long."""
                lit_module_cls0.training_step(batch_cls0, 0)
                cm = lit_module_cls0.train_cm.compute()
                assert cm.dtype == torch.long

            def test_all_correct_populates_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → only cm[0, 0] is populated."""
                lit_module_cls0.training_step(batch_cls0, 0)
                cm = lit_module_cls0.train_cm.compute()
                assert cm[0, 0].item() == _B
                assert cm.sum().item() == _B

            def test_all_wrong_populates_off_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong (pred=0, target=1) → only cm[1, 0] is populated."""
                lit_module_cls0.training_step(batch_cls1, 0)
                cm = lit_module_cls0.train_cm.compute()
                assert cm[1, 0].item() == _B
                assert cm.sum().item() == _B

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------

    class TestValMetrics:

        # ------------------------------------------------------------------
        # val_loss
        # ------------------------------------------------------------------

        class TestValLoss:
            def test_matches_model_step(self, lit_module: BrainPlanesLitModule) -> None:
                batch = _make_batch()
                ref_loss, _, _ = lit_module.model_step(batch)
                lit_module.validation_step(batch, 0)
                assert lit_module.val_loss.compute().item() == pytest.approx(ref_loss.item(), rel=1e-4)

        # ------------------------------------------------------------------
        # val_base_acc
        # ------------------------------------------------------------------

        class TestValBaseAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → val_base_acc = 1.0."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                assert lit_module_cls0.val_base_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → val_base_acc = 0.0."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                assert lit_module_cls0.val_base_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for all; labels [0, 1] → val_base_acc = 0.5."""
                lit_module_cls0.validation_step(batch_mixed, 0)
                assert lit_module_cls0.val_base_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

        # ------------------------------------------------------------------
        # val_base_f1
        # ------------------------------------------------------------------

        class TestValBaseF1:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → val_base_f1 = 1.0."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                assert lit_module_cls0.val_base_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → val_base_f1 = 0.0."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                assert lit_module_cls0.val_base_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_partial_when_mixed(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for both; labels [0, 1] → val_base_f1 = 1/3."""
                lit_module_cls0.validation_step(batch_mixed, 0)
                assert lit_module_cls0.val_base_f1.compute().item() == pytest.approx(1 / 3, abs=1e-4)

        # ------------------------------------------------------------------
        # val_tta_acc / val_tta_acc_best
        # ------------------------------------------------------------------

        class TestValTtaAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → val_tta_acc = 1.0."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                assert lit_module_cls0.val_tta_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → val_tta_acc = 0.0."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                assert lit_module_cls0.val_tta_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for all; labels [0, 1] → val_tta_acc = 0.5."""
                lit_module_cls0.validation_step(batch_mixed, 0)
                assert lit_module_cls0.val_tta_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

            def test_best_increases_with_better_epoch(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """val_tta_acc_best must increase when a better epoch follows a worse one."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                lit_module_cls0.on_validation_epoch_end()
                best_after_epoch1 = lit_module_cls0.val_tta_acc_best.compute().item()

                _reset_val_metrics(lit_module_cls0)

                lit_module_cls0.validation_step(batch_cls0, 0)
                lit_module_cls0.on_validation_epoch_end()
                assert lit_module_cls0.val_tta_acc_best.compute().item() > best_after_epoch1

            def test_best_does_not_decrease(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """val_tta_acc_best must not decrease when a worse epoch follows a better one."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                lit_module_cls0.on_validation_epoch_end()
                best_after_epoch1 = lit_module_cls0.val_tta_acc_best.compute().item()

                _reset_val_metrics(lit_module_cls0)

                lit_module_cls0.validation_step(batch_cls1, 0)
                lit_module_cls0.on_validation_epoch_end()
                assert lit_module_cls0.val_tta_acc_best.compute().item() == pytest.approx(best_after_epoch1)

        # ------------------------------------------------------------------
        # val_tta_f1 / val_tta_f1_best
        # ------------------------------------------------------------------

        class TestValTtaF1:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → val_tta_f1 = 1.0."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                assert lit_module_cls0.val_tta_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → val_tta_f1 = 0.0."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                assert lit_module_cls0.val_tta_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_partial_when_mixed(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for both; labels [0, 1] → val_tta_f1 = 1/3."""
                lit_module_cls0.validation_step(batch_mixed, 0)
                assert lit_module_cls0.val_tta_f1.compute().item() == pytest.approx(1 / 3, abs=1e-4)

            def test_best_increases_with_better_epoch(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """val_tta_f1_best must increase when a better epoch follows a worse one."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                lit_module_cls0.on_validation_epoch_end()
                best_after_epoch1 = lit_module_cls0.val_tta_f1_best.compute().item()

                _reset_val_metrics(lit_module_cls0)

                lit_module_cls0.validation_step(batch_cls0, 0)
                lit_module_cls0.on_validation_epoch_end()
                assert lit_module_cls0.val_tta_f1_best.compute().item() > best_after_epoch1

            def test_best_tracks_maximum(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """val_tta_f1_best must not decrease when a worse epoch follows a better one."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                lit_module_cls0.on_validation_epoch_end()
                best_f1 = lit_module_cls0.val_tta_f1_best.compute().item()

                _reset_val_metrics(lit_module_cls0)

                lit_module_cls0.validation_step(batch_cls1, 0)
                lit_module_cls0.on_validation_epoch_end()
                assert lit_module_cls0.val_tta_f1_best.compute().item() == pytest.approx(best_f1)

        # ------------------------------------------------------------------
        # val_base_acc_cm
        # ------------------------------------------------------------------

        class TestValBaseAccCm:
            def test_has_shape_n_by_n(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """val_base_acc_cm is a _NUM_CLASSES×_NUM_CLASSES matrix."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                cm = lit_module_cls0.val_base_acc_cm.compute()
                assert cm.shape == (_NUM_CLASSES, _NUM_CLASSES)

            def test_has_dtype_float(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """normalize='true' → row-normalized fractions → dtype torch.float32."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                cm = lit_module_cls0.val_base_acc_cm.compute()
                assert cm.dtype == torch.float32

            def test_all_correct_gives_one_in_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → cm[0, 0] = 1.0 (row 0 fully classified correctly)."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                cm = lit_module_cls0.val_base_acc_cm.compute()
                assert cm[0, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[0, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

            def test_all_wrong_gives_one_in_off_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong (pred=0, target=1) → cm[1, 0] = 1.0."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                cm = lit_module_cls0.val_base_acc_cm.compute()
                assert cm[1, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[1, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

        # ------------------------------------------------------------------
        # val_tta_acc_cm
        # ------------------------------------------------------------------

        class TestValTtaAccCm:
            def test_has_shape_n_by_n(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """val_tta_acc_cm is a _NUM_CLASSES×_NUM_CLASSES matrix."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                cm = lit_module_cls0.val_tta_acc_cm.compute()
                assert cm.shape == (_NUM_CLASSES, _NUM_CLASSES)

            def test_has_dtype_float(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """normalize='true' → row-normalized fractions → dtype torch.float32."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                cm = lit_module_cls0.val_tta_acc_cm.compute()
                assert cm.dtype == torch.float32

            def test_all_correct_gives_one_in_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → cm[0, 0] = 1.0."""
                lit_module_cls0.validation_step(batch_cls0, 0)
                cm = lit_module_cls0.val_tta_acc_cm.compute()
                assert cm[0, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[0, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

            def test_all_wrong_gives_one_in_off_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong (pred=0, target=1) → cm[1, 0] = 1.0."""
                lit_module_cls0.validation_step(batch_cls1, 0)
                cm = lit_module_cls0.val_tta_acc_cm.compute()
                assert cm[1, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[1, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

    # ------------------------------------------------------------------
    # Test metrics
    # ------------------------------------------------------------------

    class TestTestMetrics:

        # ------------------------------------------------------------------
        # test_loss
        # ------------------------------------------------------------------

        class TestTestLoss:
            def test_matches_model_step(self, lit_module: BrainPlanesLitModule) -> None:
                batch = _make_batch()
                ref_loss, _, _ = lit_module.model_step(batch)
                lit_module.test_step(batch, 0)
                assert lit_module.test_loss.compute().item() == pytest.approx(ref_loss.item(), rel=1e-4)

        # ------------------------------------------------------------------
        # test_base_acc
        # ------------------------------------------------------------------

        class TestTestBaseAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → test_base_acc = 1.0."""
                lit_module_cls0.test_step(batch_cls0, 0)
                assert lit_module_cls0.test_base_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → test_base_acc = 0.0."""
                lit_module_cls0.test_step(batch_cls1, 0)
                assert lit_module_cls0.test_base_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for all; labels [0, 1] → test_base_acc = 0.5."""
                lit_module_cls0.test_step(batch_mixed, 0)
                assert lit_module_cls0.test_base_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

        # ------------------------------------------------------------------
        # test_base_f1
        # ------------------------------------------------------------------

        class TestTestBaseF1:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → test_base_f1 = 1.0."""
                lit_module_cls0.test_step(batch_cls0, 0)
                assert lit_module_cls0.test_base_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → test_base_f1 = 0.0."""
                lit_module_cls0.test_step(batch_cls1, 0)
                assert lit_module_cls0.test_base_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_partial_when_mixed(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for both; labels [0, 1] → test_base_f1 = 1/3."""
                lit_module_cls0.test_step(batch_mixed, 0)
                assert lit_module_cls0.test_base_f1.compute().item() == pytest.approx(1 / 3, abs=1e-4)

        # ------------------------------------------------------------------
        # test_tta_acc
        # ------------------------------------------------------------------

        class TestTestTtaAcc:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → test_tta_acc = 1.0."""
                lit_module_cls0.test_step(batch_cls0, 0)
                assert lit_module_cls0.test_tta_acc.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → test_tta_acc = 0.0."""
                lit_module_cls0.test_step(batch_cls1, 0)
                assert lit_module_cls0.test_tta_acc.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_half_when_half_correct(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for all; labels [0, 1] → test_tta_acc = 0.5."""
                lit_module_cls0.test_step(batch_mixed, 0)
                assert lit_module_cls0.test_tta_acc.compute().item() == pytest.approx(0.5, abs=1e-4)

            def test_matches_base_for_identity_transform(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """With a single identity TTA transform, TTA acc must equal base acc."""
                lit_module_cls0.test_step(batch_cls0, 0)
                base = lit_module_cls0.test_base_acc.compute().item()
                tta = lit_module_cls0.test_tta_acc.compute().item()
                assert tta == pytest.approx(base, abs=1e-5)

        # ------------------------------------------------------------------
        # test_tta_f1
        # ------------------------------------------------------------------

        class TestTestTtaF1:
            def test_is_one_when_perfect(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → test_tta_f1 = 1.0."""
                lit_module_cls0.test_step(batch_cls0, 0)
                assert lit_module_cls0.test_tta_f1.compute().item() == pytest.approx(1.0, abs=1e-4)

            def test_is_zero_when_all_wrong(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong → test_tta_f1 = 0.0."""
                lit_module_cls0.test_step(batch_cls1, 0)
                assert lit_module_cls0.test_tta_f1.compute().item() == pytest.approx(0.0, abs=1e-4)

            def test_is_partial_when_mixed(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_mixed: tuple[Tensor, Tensor],
            ) -> None:
                """Model predicts class 0 for both; labels [0, 1] → test_tta_f1 = 1/3."""
                lit_module_cls0.test_step(batch_mixed, 0)
                assert lit_module_cls0.test_tta_f1.compute().item() == pytest.approx(1 / 3, abs=1e-4)

        # ------------------------------------------------------------------
        # test_base_acc_cm
        # ------------------------------------------------------------------

        class TestTestBaseAccCm:
            def test_has_shape_n_by_n(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """test_base_acc_cm is a _NUM_CLASSES×_NUM_CLASSES matrix."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_base_acc_cm.compute()
                assert cm.shape == (_NUM_CLASSES, _NUM_CLASSES)

            def test_has_dtype_float(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """normalize='true' → row-normalized fractions → dtype torch.float32."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_base_acc_cm.compute()
                assert cm.dtype == torch.float32

            def test_all_correct_gives_one_in_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → cm[0, 0] = 1.0."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_base_acc_cm.compute()
                assert cm[0, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[0, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

            def test_all_wrong_gives_one_in_off_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong (pred=0, target=1) → cm[1, 0] = 1.0."""
                lit_module_cls0.test_step(batch_cls1, 0)
                cm = lit_module_cls0.test_base_acc_cm.compute()
                assert cm[1, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[1, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

        # ------------------------------------------------------------------
        # test_base_cm
        # ------------------------------------------------------------------

        class TestTestBaseCm:
            def test_has_shape_n_by_n(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """test_base_cm is a _NUM_CLASSES×_NUM_CLASSES matrix."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_base_cm.compute()
                assert cm.shape == (_NUM_CLASSES, _NUM_CLASSES)

            def test_has_dtype_long(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """normalize='none' → raw integer counts → dtype torch.long."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_base_cm.compute()
                assert cm.dtype == torch.long

            def test_all_correct_populates_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → only cm[0, 0] is populated."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_base_cm.compute()
                assert cm[0, 0].item() == _B
                assert cm.sum().item() == _B

            def test_all_wrong_populates_off_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong (pred=0, target=1) → only cm[1, 0] is populated."""
                lit_module_cls0.test_step(batch_cls1, 0)
                cm = lit_module_cls0.test_base_cm.compute()
                assert cm[1, 0].item() == _B
                assert cm.sum().item() == _B

        # ------------------------------------------------------------------
        # test_tta_acc_cm
        # ------------------------------------------------------------------

        class TestTestTtaAccCm:
            def test_has_shape_n_by_n(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """test_tta_acc_cm is a _NUM_CLASSES×_NUM_CLASSES matrix."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_tta_acc_cm.compute()
                assert cm.shape == (_NUM_CLASSES, _NUM_CLASSES)

            def test_has_dtype_float(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """normalize='true' → row-normalized fractions → dtype torch.float32."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_tta_acc_cm.compute()
                assert cm.dtype == torch.float32

            def test_all_correct_gives_one_in_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → cm[0, 0] = 1.0."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_tta_acc_cm.compute()
                assert cm[0, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[0, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

            def test_all_wrong_gives_one_in_off_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong (pred=0, target=1) → cm[1, 0] = 1.0."""
                lit_module_cls0.test_step(batch_cls1, 0)
                cm = lit_module_cls0.test_tta_acc_cm.compute()
                assert cm[1, 0].item() == pytest.approx(1.0, abs=1e-4)
                assert cm[1, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

        # ------------------------------------------------------------------
        # test_tta_cm
        # ------------------------------------------------------------------

        class TestTestTtaCm:
            def test_has_shape_n_by_n(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """test_tta_cm is a _NUM_CLASSES×_NUM_CLASSES matrix."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_tta_cm.compute()
                assert cm.shape == (_NUM_CLASSES, _NUM_CLASSES)

            def test_has_dtype_long(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """normalize='none' → raw integer counts → dtype torch.long."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_tta_cm.compute()
                assert cm.dtype == torch.long

            def test_all_correct_populates_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls0: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions correct → only cm[0, 0] is populated."""
                lit_module_cls0.test_step(batch_cls0, 0)
                cm = lit_module_cls0.test_tta_cm.compute()
                assert cm[0, 0].item() == _B
                assert cm.sum().item() == _B

            def test_all_wrong_populates_off_diagonal(
                self,
                lit_module_cls0: BrainPlanesLitModule,
                batch_cls1: tuple[Tensor, Tensor],
            ) -> None:
                """All predictions wrong (pred=0, target=1) → only cm[1, 0] is populated."""
                lit_module_cls0.test_step(batch_cls1, 0)
                cm = lit_module_cls0.test_tta_cm.compute()
                assert cm[1, 0].item() == _B
                assert cm.sum().item() == _B


# ---------------------------------------------------------------------------
# on_train_start resets best metrics
# ---------------------------------------------------------------------------


def test_on_train_start_resets_best_metrics(lit_module: BrainPlanesLitModule) -> None:
    """Best-metric MaxMetrics must be at −inf after on_train_start."""
    batch = _make_batch()
    lit_module.validation_step(batch, 0)
    lit_module.on_validation_epoch_end()

    lit_module.on_train_start()

    assert lit_module.val_tta_acc_best.compute().item() == float("-inf")
    assert lit_module.val_tta_f1_best.compute().item() == float("-inf")
