import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.segmentation import DiceScore
from torchvision.ops import focal_loss, sigmoid_focal_loss


class WeightedMSELoss(torch.nn.Module):
    """Mean squared error loss that asymmetrically up-weights over-predictions.

    Elements where the prediction exceeds the target (positive residual,
    ``y_hat > y``) are scaled by ``weight``; under-predictions keep a weight of 1.
    Useful when over-predicting is more costly than under-predicting.

    :param weight: Multiplier applied to the squared error of over-predictions.
    """

    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Compute the asymmetrically weighted mean squared error.

        :param y_hat: Predicted values.
        :param y: Target values.
        :return: Scalar weighted MSE loss.
        """
        residual = y_hat - y

        # Build per-element weight: up-weight positive residuals (over-predictions).
        # The mask must be computed on the raw residual before any squaring.
        weight = torch.ones(residual.shape, device=residual.device)
        weight = torch.masked_fill(weight, residual > 0, self.weight)

        # Square first, then apply weight: loss = w * (y_hat - y)²
        loss = torch.mul(residual, residual) * weight
        return torch.mean(loss)


class BinaryDiceScore(Metric):
    """Hard (thresholded) binary Dice similarity coefficient, used as a metric.

    Thin wrapper around `torchmetrics.segmentation.DiceScore` that accepts the
    same input format as the other binary torchmetrics used in this project (for
    example `torchmetrics.F1Score` with ``task="binary"``): predicted
    probabilities/scores of shape ``[B, 1, H, W]`` or ``[B, H, W]`` and a matching
    ground-truth mask. Inputs are binarised at ``threshold`` (default ``0.5``) and the
    channel dimension is squeezed before being forwarded to ``DiceScore`` as index
    tensors.

    Being a proper stateful metric, it accumulates the Dice numerator/denominator
    **globally** across all batches and computes the score once (via ``compute()``).
    This matches how `torchmetrics.F1Score` is aggregated and, on binary masks, yields
    exactly the same value as the foreground pixel F1 — unlike a per-batch mean of Dice
    ratios, which is biased.

    ``include_background=False`` keeps only the foreground class, so ``average="micro"``
    and ``average="macro"`` are equivalent; ``"macro"`` is used to avoid a torchmetrics
    deprecation warning. Returns a similarity score in ``[0, 1]`` (1 = perfect overlap),
    unlike the Dice *loss* which returns ``1 - dice``.

    :param threshold: Binarisation threshold applied to the predictions.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self._dice = DiceScore(
            num_classes=2,
            include_background=False,
            average="macro",
            aggregation_level="global",
            input_format="index",
        )

    @staticmethod
    def _to_index(tensor: Tensor) -> Tensor:
        """Squeeze an optional channel dimension, returning an index tensor ``[B, H, W]``."""
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        return tensor.long()

    def update(self, inputs: Tensor, targets: Tensor) -> None:
        """Update the metric state with a new batch.

        :param inputs: Predicted probabilities or scores; binarised at ``threshold``.
            Shape ``[B, 1, H, W]`` or ``[B, H, W]``.
        :param targets: Ground-truth binary mask, same accepted shapes.
        """
        preds = self._to_index(inputs > self.threshold)
        target = self._to_index(targets)
        self._dice.update(preds, target)

    def compute(self) -> Tensor:
        """Compute the global binary Dice score in ``[0, 1]``."""
        return self._dice.compute()

    def reset(self) -> None:
        """Reset the accumulated metric state."""
        self._dice.reset()
        super().reset()


class BinaryDiceLoss(torch.nn.Module):
    """Soft Dice loss operating on raw logits.

    Applies a sigmoid to the inputs and returns ``1 - dice``, where the soft Dice
    overlap is computed on the resulting probabilities. Being a region-overlap
    measure, it is inherently robust to foreground/background class imbalance.

    :param smooth: Smoothing constant added to numerator and denominator for
        numerical stability.
    """

    def __init__(
        self,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor):
        """Compute the soft Dice loss.

        :param inputs: Raw logits ``[B, ...]``; a sigmoid is applied internally.
        :param targets: Ground-truth binary mask.
        :return: Scalar Dice loss (``1 - dice``).
        """
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BinaryFocalLoss(torch.nn.Module):
    """Binary focal loss operating on raw logits.

    Down-weights the contribution of well-classified pixels via the ``(1 - p_t) ** gamma``
    modulating factor and rebalances the foreground/background classes via ``alpha``.

    :param alpha: Weight for the positive (foreground) class in ``[0, 1]``; the negative
        class receives ``1 - alpha``. Set to ``None`` to disable alpha-balancing.
    :param gamma: Focusing parameter; ``gamma=0`` reduces to (alpha-weighted) BCE.
    :param reduction: One of ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()

        if alpha is not None and not (0 <= alpha <= 1):
            raise ValueError(f"Invalid alpha value: {alpha}. alpha must be in the range [0,1] or None for ignore.")

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha if self.alpha is not None else -1,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class BinaryTverskyLoss(torch.nn.Module):
    """Soft Tversky loss operating on raw logits.

    Generalises Dice loss: ``TP / (TP + alpha * FP + beta * FN)``. With
    ``alpha == beta == 0.5`` it is equivalent to the soft Dice loss. Increasing
    ``alpha`` relative to ``beta`` penalises false positives more heavily (favours
    precision); increasing ``beta`` favours recall.

    :param alpha: Penalty weight on false positives.
    :param beta: Penalty weight on false negatives.
    :param smooth: Smoothing constant for numerical stability.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.sigmoid(inputs)
        targets = targets.type_as(inputs)

        true_pos = (inputs * targets).sum()
        false_pos = (inputs * (1.0 - targets)).sum()
        false_neg = ((1.0 - inputs) * targets).sum()

        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        return 1.0 - tversky


class BinaryDiceCrossEntropyLoss(torch.nn.Module):
    """Combined soft Dice loss and binary cross-entropy loss (summed).

    The Dice term optimises region overlap and is robust to class imbalance, while
    the BCE term provides a stable per-pixel gradient. The BCE component accepts the
    standard `torch.nn.BCEWithLogitsLoss` weighting arguments; ``pos_weight``
    up-weights the positive (foreground) class in the BCE term only (the Dice term
    is unweighted).

    :param weight: Optional per-element rescaling weight for the BCE term.
    :param size_average: Deprecated ``BCEWithLogitsLoss`` averaging flag.
    :param reduce: Deprecated ``BCEWithLogitsLoss`` reduction flag.
    :param reduction: Reduction for the BCE term (``"mean"``, ``"sum"`` or ``"none"``).
    :param pos_weight: Optional weight on positive examples for the BCE term.
    :param smooth: Smoothing constant for the Dice term.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice = BinaryDiceLoss(
            smooth=smooth,
        )
        self.bce = torch.nn.BCEWithLogitsLoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )

    def forward(self, inputs: Tensor, targets: Tensor):
        """Compute the summed Dice + BCE loss.

        :param inputs: Raw logits; a sigmoid is applied internally by both terms.
        :param targets: Ground-truth binary mask.
        :return: Scalar combined loss.
        """
        return self.dice(inputs, targets) + self.bce(inputs, targets)


class BinaryDiceFocalLoss(torch.nn.Module):
    """Combined soft Dice loss and binary focal loss (summed).

    The Dice term directly optimises region overlap (Dice/IoU) and is robust to
    class imbalance, while the focal term focuses learning on hard pixels. Mirrors
    the structure of `BinaryDiceCrossEntropyLoss`.

    :param alpha: Alpha-balancing weight for the focal term (``None`` to disable).
    :param gamma: Focusing parameter for the focal term.
    :param reduction: Reduction for the focal term.
    :param smooth: Smoothing constant for the Dice term.
    """

    def __init__(
        self,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice = BinaryDiceLoss(smooth=smooth)
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(self, inputs: Tensor, targets: Tensor):
        return self.dice(inputs, targets) + self.focal(inputs, targets)


class BinaryFocalTverskyLoss(torch.nn.Module):
    """Focal Tversky loss operating on raw logits.

    Applies a focal modulation ``(1 - TverskyIndex) ** gamma`` to the Tversky loss
    (Abraham & Khan, 2019), focusing learning on harder regions while retaining
    Tversky's asymmetric control over false positives (``alpha``) and false
    negatives (``beta``). With ``alpha == beta == 0.5`` and ``gamma == 1`` it reduces
    to the soft Dice loss. The paper recommends ``gamma = 4/3``.

    :param alpha: Penalty weight on false positives.
    :param beta: Penalty weight on false negatives.
    :param gamma: Focusing exponent; ``gamma == 1`` disables focal modulation.
    :param smooth: Smoothing constant for numerical stability.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.sigmoid(inputs)
        targets = targets.type_as(inputs)

        true_pos = (inputs * targets).sum()
        false_pos = (inputs * (1.0 - targets)).sum()
        false_neg = ((1.0 - inputs) * targets).sum()

        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)
