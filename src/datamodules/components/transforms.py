from typing import List

import torch


class LabelEncoder(torch.nn.Module):
    def __init__(self, labels: List[str]):
        super().__init__()
        self.labels = labels

    def forward(self, target: str) -> torch.Tensor:
        target = self.labels.index(target)
        return torch.as_tensor(target)
