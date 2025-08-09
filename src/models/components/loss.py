from typing import Optional

import torch
from torch import Tensor


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = y_hat - y

        weight = torch.ones(loss.shape, device=loss.device)
        weight = torch.masked_fill(weight, loss > 0, self.weight)
        loss = loss * weight

        loss = torch.mul(loss, loss)
        return torch.mean(loss)


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
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BinaryDiceCrossEntropyLoss(torch.nn.Module):
    def __init__(
            self,
            smooth: float = 1.0,
            weight: Optional[Tensor] = None,
            size_average=None,
            reduce=None,
            reduction: str = "mean",
            pos_weight: Optional[Tensor] = None,
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
