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
