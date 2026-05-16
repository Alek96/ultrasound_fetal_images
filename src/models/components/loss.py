import torch
from torch import Tensor


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        residual = y_hat - y

        # Build per-element weight: up-weight positive residuals (over-predictions).
        # The mask must be computed on the raw residual before any squaring.
        weight = torch.ones(residual.shape, device=residual.device)
        weight = torch.masked_fill(weight, residual > 0, self.weight)

        # Square first, then apply weight: loss = w * (y_hat - y)²
        loss = torch.mul(residual, residual) * weight
        return torch.mean(loss)


class BinaryDiceScore(torch.nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        inputs = (inputs > 0.5).int()
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return dice


class BinaryDiceLoss(torch.nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BinaryDiceCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
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
        return self.dice(inputs, targets) + self.bce(inputs, targets)
