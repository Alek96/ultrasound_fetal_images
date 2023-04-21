import math
from enum import Enum
from math import floor
from typing import Dict, List, Optional, Tuple

import albumentations as A
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor


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


class VerticalFlip(torch.nn.Module):
    def __init__(self, flip: bool = True) -> None:
        super().__init__()
        self.flip = flip

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Vertically flipped image.
        """
        return TF.vflip(img=img) if self.flip else img

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


class RandomCutout(torch.nn.Module):
    def __init__(self, n_holes: int = 1, length: int = 10, p: float = 0.5) -> None:
        super().__init__()
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be flipped.

        Returns:
            PIL Tensor: Randomly cutout image.
        """
        if self.p < torch.rand(1):
            return img
        return cutout(img, self.n_holes, self.length)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_holes={self.n_holes},length={self.length})"


def cutout(img: Tensor, n_holes: int = 1, length: int = 10, fill: None | list[float] = None):
    c, h, w = img.shape
    mask = torch.zeros((h, w), dtype=torch.bool)
    fill = fill if fill is not None else torch.zeros(c)

    for _ in range(n_holes):
        y = torch.randint(h, ())
        x = torch.randint(w, ())

        y1 = torch.clip(y - length // 2, 0, h).int()
        y2 = torch.clip(y + length // 2, 0, h).int()
        x1 = torch.clip(x - length // 2, 0, w).int()
        x2 = torch.clip(x + length // 2, 0, w).int()

        mask[y1:y2, x1:x2] = True

    for i, value in enumerate(fill):
        img_mask = torch.zeros(img.shape, dtype=torch.bool)
        img_mask[i] = mask
        img = img.masked_fill(mask, value)

    return img


def elastic_transform(img, magnitude, scale):
    # 0-100  / 5
    # 0-400 / 10
    sigma = img.size(2) / scale
    alpha = (sigma / 5) ** 2 * magnitude
    return T.ElasticTransform(alpha=alpha, sigma=sigma)(img)


def grid_distortion(img, magnitude, num_steps=5):
    img = img.numpy().transpose(1, 2, 0)  # HWC
    img = A.GridDistortion(num_steps=num_steps, distort_limit=magnitude, normalized=False, p=1.0)(image=img)["image"]
    img = torch.from_numpy(img.transpose(2, 0, 1))  # CHW
    return img


def speckle_noise(img: Tensor, mean: float = 0, std: float = 0.1):
    gauss = torch.empty(img.shape).normal_(mean=mean, std=std)
    img = img + img * gauss
    return img.round().clip(min=0, max=255).type(torch.uint8)


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: TF.InterpolationMode,
    fill: list[float] | None,
    arg=None,
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = TF.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = TF.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = TF.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = TF.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = TF.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = TF.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = TF.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = TF.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = TF.autocontrast(img)
    elif op_name == "Equalize":
        img = TF.equalize(img)
    elif op_name == "Invert":
        img = TF.invert(img)
    elif op_name == "Gamma":
        img = TF.adjust_gamma(img, 1.0 + magnitude)
    elif op_name == "Cutout":
        arg = arg if arg is not None else 5
        img = cutout(img, n_holes=arg, length=int(magnitude), fill=fill)
    elif op_name == "RandomErasing":
        img = T.RandomErasing(p=1.0, scale=(magnitude, magnitude), ratio=(0.3, 3.3), value=0)(img)
    elif op_name == "Elastic":
        arg = arg if arg is not None else 24
        img = elastic_transform(img, magnitude, scale=arg)
    elif op_name == "GridDistortion":
        arg = arg if arg is not None else 5
        img = grid_distortion(img, magnitude, num_steps=arg)
    elif op_name == "Speckle":
        img = speckle_noise(img, mean=0, std=magnitude)

    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class RandAugmentPolicy(Enum):
    RAND_AUGMENT = "RandAugment"
    RAND_AUGMENT_INVERT = "RandAugmentInvert"
    RAND_AUGMENT_GAMMA = "RandAugmentGamma"
    RAND_AUGMENT_CUTOUT = "RandAugmentCutout"
    RAND_AUGMENT_RANDOM_ERASING = "RandAugmentRandomErasing"
    RAND_AUGMENT_ELASTIC = "RandAugmentElastic"
    RAND_AUGMENT_GRID_DISTORTION = "RandAugmentGridDistortion"
    RAND_AUGMENT_SPECKLE = "RandAugmentSpeckle"
    RAND_AUGMENT_14 = "RandAugment14"
    RAND_AUGMENT_18 = "RandAugment18"


class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        policy: RandAugmentPolicy = RandAugmentPolicy.RAND_AUGMENT,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
        fill: list[float] | None = None,
        arg1=None,
        arg2=None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.arg1 = arg1
        self.arg2 = arg2

    def _augmentation_space(
        self, policy: RandAugmentPolicy, num_bins: int, image_size: tuple[int, int]
    ) -> dict[str, tuple[Tensor, bool]]:
        if policy == RandAugmentPolicy.RAND_AUGMENT:
            return {
                # op_name: (magnitudes, signed)
                "Identity": (torch.tensor(0.0), False),
                "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
                "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
                "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
                "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
                "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
                "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
                "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
                "AutoContrast": (torch.tensor(0.0), False),
                "Equalize": (torch.tensor(0.0), False),
            }
        elif policy == RandAugmentPolicy.RAND_AUGMENT_INVERT:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            op_meta["Invert"] = (torch.tensor(0.0), False)
            return op_meta
        elif policy == RandAugmentPolicy.RAND_AUGMENT_GAMMA:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            op_meta["Gamma"] = (torch.linspace(0.0, 0.9, num_bins), True)
            return op_meta
        elif policy == RandAugmentPolicy.RAND_AUGMENT_CUTOUT:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            arg = self.arg1 if self.arg1 is not None else 0.5
            op_meta["Cutout"] = (torch.linspace(0.0, arg * min(image_size), num_bins), False)
            return op_meta
        elif policy == RandAugmentPolicy.RAND_AUGMENT_RANDOM_ERASING:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            arg = self.arg1 if self.arg1 is not None else 0.5
            op_meta["RandomErasing"] = (torch.linspace(0.0, arg * image_size[0] / image_size[1], num_bins), False)
            return op_meta
        elif policy == RandAugmentPolicy.RAND_AUGMENT_ELASTIC:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            arg = self.arg1 if self.arg1 is not None else 100.0
            op_meta["Elastic"] = (torch.linspace(0.0, arg, num_bins), False)
            return op_meta
        elif policy == RandAugmentPolicy.RAND_AUGMENT_GRID_DISTORTION:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            arg = self.arg1 if self.arg1 is not None else 0.5
            op_meta["GridDistortion"] = (torch.linspace(0.0, arg, num_bins), False)
            return op_meta
        elif policy == RandAugmentPolicy.RAND_AUGMENT_SPECKLE:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            arg = self.arg1 if self.arg1 is not None else 0.9
            op_meta["Speckle"] = (torch.linspace(0.0, arg, num_bins), False)
            return op_meta

        elif policy == RandAugmentPolicy.RAND_AUGMENT_14:
            return {
                # op_name: (magnitudes, signed)
                "Identity": (torch.tensor(0.0), False),
                "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
                "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
                "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
                "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
                "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
                "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
                "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
                "AutoContrast": (torch.tensor(0.0), False),
                "Equalize": (torch.tensor(0.0), False),
                "Invert": (torch.tensor(0.0), False),
                "Gamma": (torch.linspace(0.0, 0.9, num_bins), True),
                "Cutout": (torch.linspace(0.0, 0.7 * min(image_size), num_bins), False),
                # "RandomErasing": (torch.linspace(0.0, 0.5 * image_size[0] / image_size[1], num_bins), False),
                # "Elastic": (torch.linspace(0.0, 400.0, num_bins), False),
                "Speckle": (torch.linspace(0.0, 0.9, num_bins), False),
            }

        elif policy == RandAugmentPolicy.RAND_AUGMENT_18:
            return {
                # op_name: (magnitudes, signed)
                "Identity": (torch.tensor(0.0), False),
                "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
                "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
                "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
                "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
                "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
                "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
                "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
                "AutoContrast": (torch.tensor(0.0), False),
                "Equalize": (torch.tensor(0.0), False),
                "Invert": (torch.tensor(0.0), False),
                "Gamma": (torch.linspace(0.0, 0.9, num_bins), True),
                "Cutout": (torch.linspace(0.0, 0.5 * min(image_size), num_bins), False),
                "RandomErasing": (torch.linspace(0.0, 0.5 * image_size[0] / image_size[1], num_bins), False),
                "Elastic": (torch.linspace(0.0, 400.0, num_bins), False),
                "Speckle": (torch.linspace(0.0, 0.9, num_bins), False),
            }
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def forward(self, img: Tensor) -> Tensor:
        """img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = TF.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.policy, self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill, arg=self.arg2)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


class LabelEncoder(torch.nn.Module):
    def __init__(self, labels: list[str]):
        super().__init__()
        self.labels = labels

    def forward(self, target: str) -> torch.Tensor:
        target = self.labels.index(target)
        return torch.as_tensor(target)


class OneHotEncoder(torch.nn.Module):
    def __init__(self, labels: int | list[str]):
        super().__init__()
        self.labels = labels if isinstance(labels, int) else len(labels)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return F.one_hot(target, num_classes=self.labels).float()
