import math
from collections.abc import Sequence
from enum import Enum
from math import floor
from typing import Any, Literal

import albumentations as A
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torch import Tensor
from torchvision import tv_tensors


def _flatten_inputs(inputs: Any):
    if isinstance(inputs, tuple):
        if len(inputs) > 1:
            return inputs
        else:
            return _flatten_inputs(inputs[0])
    return inputs


def _wrap_inputs(inputs: Any):
    inputs = _flatten_inputs(inputs)
    if isinstance(inputs, tuple):
        return inputs
    return (inputs,)


def get_image_shape(*inputs: Any):
    inputs = _flatten_inputs(inputs)
    if isinstance(inputs, Sequence):
        for i in inputs:
            if isinstance(i, tv_tensors.Image):
                return i.shape[1:]
    elif isinstance(inputs, torch.Tensor):
        return inputs.shape[1:]
    else:
        raise TypeError("inputs type not supported")


class PadToAspectRatio(torch.nn.Module):
    def __init__(
        self,
        size: Sequence[int],
        fill: TF._utils._FillType | dict[type | str, TF._utils._FillType] = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    ) -> None:
        super().__init__()
        self.size = list(size)
        self.fill = fill
        self.padding_mode = padding_mode
        self.height, self.width = size
        self.aspect_ratio = self.height / self.width

    def forward(self, *inputs: Any):
        original_height, original_width = get_image_shape(*inputs)
        # Calculate the current aspect ratio of the image
        current_aspect_ratio = original_height / original_width

        # Determine which dimension needs padding
        if current_aspect_ratio > self.aspect_ratio:
            # Image is too tall, need to pad width
            new_width = int(original_height / self.aspect_ratio)
            padding_needed = new_width - original_width

            # Calculate padding on left and right sides
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            top_pad, bottom_pad = 0, 0

        elif current_aspect_ratio < self.aspect_ratio:
            # Image is too wide, need to pad height
            new_height = int(original_width * self.aspect_ratio)
            padding_needed = new_height - original_height

            # Calculate padding on top and bottom sides
            top_pad = padding_needed // 2
            bottom_pad = padding_needed - top_pad
            left_pad, right_pad = 0, 0

        else:
            # Image already has the correct aspect ratio, no padding needed
            inputs = _flatten_inputs(inputs)
            return inputs

        pad = T.Pad(padding=[left_pad, top_pad, right_pad, bottom_pad], fill=self.fill, padding_mode=self.padding_mode)
        inputs = pad(*inputs)
        return _flatten_inputs(inputs)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__} ("
        s += f"size={self.size}, "
        s += f"fill={self.fill}, "
        s += f"padding_mode={self.padding_mode})"

        return s


class Resize(torch.nn.Module):
    def __init__(
        self,
        size: Sequence[int],
        interpolation: (
            T.InterpolationMode | int | dict[type | str, T.InterpolationMode | int | None]
        ) = T.InterpolationMode.BILINEAR,
        antialias: bool | None = True,
    ) -> None:
        super().__init__()
        self.size = list(size)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, *inputs: Any):
        outputs = []
        for inp in inputs:
            interpolation = self._get_interpolation_mode(inp)
            output = inp.as_subclass(torch.Tensor)
            output = TF.resize(output, size=self.size, interpolation=interpolation, antialias=self.antialias)
            output = tv_tensors.wrap(output, like=inp)
            outputs.append(output)

        outputs = tuple(outputs)
        return _flatten_inputs(outputs)

    def _get_interpolation_mode(self, inp: torch.Tensor):
        if not isinstance(self.interpolation, dict):
            return self.interpolation
        default = self.interpolation.get("others")
        return self.interpolation.get(type(inp), default)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__} ("
        s += f"size={self.size}, "
        s += f"interpolation={self.interpolation}, "
        s += f"antialias={self.antialias})"

        return s


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
        return TF.hflip(img) if self.flip else img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flip={self.flip})"


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
        return TF.vflip(img) if self.flip else img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flip={self.flip})"


class Affine(torch.nn.Module):
    def __init__(
        self,
        degrees: float | None = None,
        translate: tuple[float, float] | None = None,
        scale: float | None = None,
        shear: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.degrees = degrees if degrees is not None else 0.0
        self.translate = translate if translate is not None else [0, 0]
        self.scale = scale if scale is not None else 1.0
        self.shear = shear if shear is not None else [0.0, 0.0]

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        _, height, width = TF.get_dimensions(img)
        return TF.affine(
            img,
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

        percent = (torch.rand(()) * max_percent).item()
        crop_height = floor(height * (100 - percent) / 100)
        crop_width = floor(width * (100 - percent) / 100)

        if width == crop_width and height == crop_height:
            return 0, 0, height, width

        top = torch.randint(0, height - crop_height + 1, size=(1,)).item()
        left = torch.randint(0, width - crop_width + 1, size=(1,)).item()
        return top, left, crop_height, crop_width

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        top, left, crop_height, crop_width = self.get_params(img, self.max_percent)
        return TF.crop(img, top, left, crop_height, crop_width)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_percent={self.max_percent})"


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
        if torch.rand(()) >= self.p:
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
        img = img.masked_fill(img_mask, value)

    return img


def elastic_transform(img, magnitude, scale):
    # 0-100 / 5
    # 0-400 / 10
    sigma = img.size(2) / scale
    alpha = (sigma / 5) ** 2 * magnitude
    return T.ElasticTransform(alpha=alpha, sigma=sigma)(img)


def grid_distortion(img, magnitude, num_steps=5):
    img = img.cpu().detach().numpy().transpose(1, 2, 0)  # HWC
    img = A.GridDistortion(num_steps=num_steps, distort_limit=magnitude, normalized=False, p=1.0)(image=img)["image"]
    img = torch.from_numpy(img.transpose(2, 0, 1))  # CHW
    return img


def speckle_noise(img: Tensor, mean: float = 0, std: float = 0.1):
    gauss = torch.empty(img.shape).normal_(mean=mean, std=std)
    img = img + img * gauss
    return img.round().clip(min=0, max=255).type(torch.uint8)


def _apply_op(
    img: Tensor,
    transform_id: str,
    magnitude: float,
    interpolation: TF.InterpolationMode,
    fill: list[float] | None,
):
    if transform_id == "Identity":
        return img
    if transform_id == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        return TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif transform_id == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        return TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif transform_id == "TranslateX":
        return TF.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif transform_id == "TranslateY":
        return TF.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif transform_id == "Rotate":
        return TF.rotate(img, angle=magnitude, interpolation=interpolation, fill=fill)
    elif transform_id == "Brightness":
        return TF.adjust_brightness(img, brightness_factor=1.0 + magnitude)
    elif transform_id == "Color":
        return TF.adjust_saturation(img, saturation_factor=1.0 + magnitude)
    elif transform_id == "Contrast":
        return TF.adjust_contrast(img, contrast_factor=1.0 + magnitude)
    elif transform_id == "Sharpness":
        return TF.adjust_sharpness(img, sharpness_factor=1.0 + magnitude)
    elif transform_id == "Posterize":
        return TF.posterize(img, bits=int(magnitude))
    elif transform_id == "Solarize":
        return TF.solarize(img, threshold=magnitude)
    elif transform_id == "AutoContrast":
        return TF.autocontrast(img)
    elif transform_id == "Equalize":
        return TF.equalize(img)
    elif transform_id == "Invert":
        return TF.invert(img)
    # new
    elif transform_id == "Gamma":
        return TF.adjust_gamma(img, gamma=1.0 + magnitude)
    elif transform_id == "Cutout":
        return cutout(img, n_holes=5, length=int(magnitude), fill=fill)
    elif transform_id == "RandomErasing":
        return T.RandomErasing(p=1.0, scale=(magnitude, magnitude), ratio=(0.3, 3.3), value=0)(img)
    elif transform_id == "Elastic":
        return elastic_transform(img, magnitude=magnitude, scale=24)
    elif transform_id == "GridDistortion":
        return grid_distortion(img, magnitude=magnitude, num_steps=5)
    elif transform_id == "Speckle":
        return speckle_noise(img, mean=0, std=magnitude)
    else:
        raise ValueError(f"No transform available for {transform_id}")


class RandAugmentPolicy(Enum):
    RAND_AUGMENT = "RandAugment"
    RAND_AUGMENT_INVERT = "RandAugmentInvert"
    RAND_AUGMENT_GAMMA = "RandAugmentGamma"
    RAND_AUGMENT_CUTOUT = "RandAugmentCutout"
    RAND_AUGMENT_RANDOM_ERASING = "RandAugmentRandomErasing"
    RAND_AUGMENT_ELASTIC = "RandAugmentElastic"
    RAND_AUGMENT_GRID_DISTORTION = "RandAugmentGridDistortion"
    RAND_AUGMENT_SPECKLE = "RandAugmentSpeckle"


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
            `torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
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
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(
        self, policy: RandAugmentPolicy, num_bins: int, image_size: tuple[int, int]
    ) -> dict[str, tuple[Tensor, bool, bool]]:
        op_meta = {
            # op_name: (magnitudes, signed, support_mask)
            "Identity": (torch.tensor(0.0), False, True),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True, True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True, True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True, True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True, True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True, True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True, False),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True, False),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True, False),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True, False),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False, False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False, False),
            "AutoContrast": (torch.tensor(0.0), False, False),
            "Equalize": (torch.tensor(0.0), False, False),
        }

        if policy == RandAugmentPolicy.RAND_AUGMENT_INVERT:
            op_meta["Invert"] = (torch.tensor(0.0), False, False)
        if policy == RandAugmentPolicy.RAND_AUGMENT_GAMMA:
            op_meta["Gamma"] = (torch.linspace(0.0, 0.9, num_bins), True, False)
        if policy == RandAugmentPolicy.RAND_AUGMENT_CUTOUT:
            op_meta = self._augmentation_space(RandAugmentPolicy.RAND_AUGMENT, num_bins, image_size)
            op_meta["Cutout"] = (torch.linspace(0.0, 0.5 * min(image_size), num_bins), False, False)
        if policy == RandAugmentPolicy.RAND_AUGMENT_RANDOM_ERASING:
            op_meta["RandomErasing"] = (
                torch.linspace(0.0, 0.5 * image_size[0] / image_size[1], num_bins),
                False,
                False,
            )
        if policy == RandAugmentPolicy.RAND_AUGMENT_ELASTIC:
            op_meta["Elastic"] = (torch.linspace(0.0, 100.0, num_bins), False, True)
        if policy == RandAugmentPolicy.RAND_AUGMENT_GRID_DISTORTION:
            op_meta["GridDistortion"] = (torch.linspace(0.0, 0.5, num_bins), False, True)
        if policy == RandAugmentPolicy.RAND_AUGMENT_SPECKLE:
            op_meta["Speckle"] = (torch.linspace(0.0, 0.9, num_bins), False, False)

        return op_meta

    def forward(self, *images: Any) -> Any:
        """Img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = TF.get_dimensions(images[0])
        op_meta = self._augmentation_space(self.policy, self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed, support_mask = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1.0

            images_rs = []
            for img in images:
                if not isinstance(img, tv_tensors.Mask) or support_mask:
                    img = _apply_op(
                        img,
                        transform_id=op_name,
                        magnitude=magnitude,
                        interpolation=self.interpolation,
                        fill=fill,
                    )
                images_rs.append(img)
            images = tuple(images_rs)

        return _flatten_inputs(images)

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
