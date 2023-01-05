from math import floor
from typing import List, Tuple

import torch
import torchvision.transforms.functional as F


class RandomPercentCrop(torch.nn.Module):
    """Reduce the given image by percentage. If the image is torch Tensor, it is expected to have.

    [..., H, W] shape, where ... means an arbitrary number of leading dimensions, but if
    non-constant padding is used, the input is expected to have at most 2 leading dimensions.

    Args:
        max_percent (int): By what maximal percentage to reduce the image
    """

    def __init__(self, max_percent: int):
        super().__init__()
        self.max_percent = max_percent

    @staticmethod
    def get_params(img: torch.Tensor, max_percent: int) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random percent crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            max_percent (int): By what maximal percentage to reduce the image

        Returns:
            tuple: params (top, left, height, width) to be passed to ``crop`` for random crop.
        """
        _, height, width = F.get_dimensions(img)

        percent = (torch.rand(1) * max_percent).item()
        crop_height = floor(height * (100 - percent) / 100)
        crop_weight = floor(width * (100 - percent) / 100)

        if width == crop_weight and height == crop_height:
            return 0, 0, height, width

        top = torch.randint(0, height - crop_height + 1, size=(1,)).item()
        left = torch.randint(0, width - crop_weight + 1, size=(1,)).item()
        return top, left, crop_height, crop_weight

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        top, left, crop_height, crop_weight = self.get_params(img, self.max_percent)
        return F.crop(img, top, left, crop_height, crop_weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_percent={self.max_percent})"


class LabelEncoder(torch.nn.Module):
    def __init__(self, labels: List[str]):
        super().__init__()
        self.labels = labels

    def forward(self, target: str) -> torch.Tensor:
        target = self.labels.index(target)
        return torch.as_tensor(target)
