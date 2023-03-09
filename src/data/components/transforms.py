from math import floor

import torch
import torchvision.transforms.functional as TF


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
    def get_params(img: torch.Tensor, max_percent: int) -> tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random percent crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            max_percent (int): By what maximal percentage to reduce the image

        Returns:
            tuple: params (top, left, height, width) to be passed to ``crop`` for random crop.
        """
        _, height, width = TF.get_dimensions(img)

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
        return TF.crop(img, top, left, crop_height, crop_weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_percent={self.max_percent})"


class HorizontalFlip(torch.nn.Module):
    def __init__(self, flip: bool = True) -> None:
        super().__init__()
        self.flip = flip

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Horizontally flipped image.
        """
        return TF.hflip(img=img) if self.flip else img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flip={self.max_percent})"


class Affine(torch.nn.Module):
    def __init__(
        self,
        degrees: float = None,
        translate: tuple[float, float] = None,
        scale: float = None,
        shear: float = None,
    ) -> None:
        super().__init__()
        self.degrees = degrees or 0.0
        self.translate = translate or [0, 0]
        self.scale = scale or 1.0
        self.shear = shear or [0.0, 0.0]

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        _, height, width = TF.get_dimensions(img)
        return TF.affine(
            img=img,
            angle=-self.degrees,
            translate=[int(round(self.translate[0] * width)), int(round(self.translate[1] * height))],
            scale=self.scale,
            shear=self.shear,
        )

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__} ("
        s += f"degrees={self.degrees}, "
        s += f"translate={self.translate}, "
        s += f"scale={self.scale}, "
        s += f"shear={self.shear})"

        return s


class LabelEncoder(torch.nn.Module):
    def __init__(self, labels: list[str]):
        super().__init__()
        self.labels = labels

    def forward(self, target: str) -> torch.Tensor:
        target = self.labels.index(target)
        return torch.as_tensor(target)
