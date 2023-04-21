from collections.abc import Sequence
from typing import Any, Optional, Tuple

import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.distributions.beta import Beta

from src.data.components.dataset import FetalBrainPlanesDataset
from src.data.components.transforms import OneHotEncoder
from src.data.utils.utils import show_pytorch_images
from src.utils.plots import log_to_wandb


def get_test_dataloader(trainer, dataloader_idx: int = 0):
    test_dataloaders = trainer.test_dataloaders
    if hasattr(test_dataloaders, "__getitem__"):
        return test_dataloaders[dataloader_idx]
    else:
        return test_dataloaders


class ClassImageSampler(Callback):
    def __init__(
        self,
        class_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.class_names: Sequence[str] = class_names
        self.samples: list[list[list[int]]] = torch.zeros((len(self.class_names), len(self.class_names), 0)).tolist()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        preds = outputs["preds"]
        targets = outputs["targets"]
        batch_size = get_test_dataloader(trainer, dataloader_idx).batch_size

        idx = batch_size * batch_idx
        for i, (target, pred) in enumerate(zip(targets, preds)):
            self.samples[target][pred].append(idx + i)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""
        dataset = get_test_dataloader(trainer).dataset

        images = []
        for row in self.samples:
            for cell in row:
                if len(cell) == 0:
                    images.append(None)
                else:
                    idx = torch.randint(0, len(cell), ())
                    image_idx = cell[idx]
                    image, _ = dataset[image_idx]
                    images.append(image)

        fig = show_pytorch_images(
            images=images,
            title="        Predicted",
            ylabel="Actual      ",
            cols_names=FetalBrainPlanesDataset.labels,
            rows_names=FetalBrainPlanesDataset.labels,
        )

        log_to_wandb(lambda: {"test/samples": wandb.Image(fig)}, loggers=pl_module.loggers)


class BaseMixCallback(Callback):
    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__()
        self.alpha = alpha
        self.distribution_fn = Beta(self.alpha, self.alpha)
        self.softmax_target = softmax_target
        self.one_hot_encoder = OneHotEncoder(labels)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        x, y = batch
        if not self.softmax_target and len(y.shape) == 1:
            y = self.one_hot_encoder(y)

        x_new, y_new = self.mix(x, y)

        batch[0] = x_new
        batch[1] = y_new

    def mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def distribution(self, samples: int, device) -> Tensor:
        x = self.distribution_fn.sample(torch.Size((samples,)))
        x = torch.stack([x, 1 - x], dim=1).amax(dim=1)
        return x.to(device)


class MixUpCallback(BaseMixCallback):
    """Callback that perform MixUp augmentation on the input batch.

    Reference:
     - `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_
     - `Facebook's Mixup-CIFAR10 <https://github.com/facebookresearch/mixup-cifar10>`_
    """

    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__(alpha=alpha, softmax_target=softmax_target, labels=labels)

    def mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.alpha > 0:
            lam = self.distribution(samples=1, device=x.device).squeeze()
        else:
            lam = 1

        index = torch.randperm(x.size(0), device=x.device)

        # Combine input batch
        x_new = lam * x + (1 - lam) * x[index, :]

        # Combine targets
        if self.softmax_target:
            y_lam = lam.expand(y.size(0))
            y_new = torch.stack([y.float(), y[index].float(), y_lam, (1 - y_lam)], dim=1)
        else:
            y_new = lam * y + (1 - lam) * y[index]

        return x_new, y_new


class MixUpV2Callback(BaseMixCallback):
    """Callback that perform MixUp augmentation on the input batch.

    Reference:
     - `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_
     - `PyTorch Lightning Spells's implementation <https://github.com/veritable-tech/pytorch-lightning-spells/blob/master/pytorch_lightning_spells/callbacks.py>`_
     - `Fast.ai's implementation <https://github.com/fastai/fastai/blob/master/fastai/callback/mixup.py>`_
    """

    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__(alpha=alpha, softmax_target=softmax_target, labels=labels)

    def mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.alpha > 0:
            lam = self.distribution(samples=x.size(0), device=x.device)
        else:
            lam = torch.ones(x.shape[:1], device=x.device)

        index = torch.randperm(x.size(0), device=x.device)

        # Create the tensor and expand (for batch inputs)
        x_lam = lam.view(-1, *[1 for _ in range(len(x.shape[1:]))]).expand(-1, *x.shape[1:])
        # Combine input batch
        x_new = x_lam * x + (1 - x_lam) * x[index]

        # Create the tensor and expand (for target)
        y_lam = lam.view(-1, *[1 for _ in range(len(y.size()) - 1)]).expand(-1, *y.shape[1:])
        # Combine targets
        if self.softmax_target:
            y_new = torch.stack([y.float(), y[index].float(), y_lam, (1 - y_lam)], dim=1)
        else:
            y_new = y_lam * y + (1 - y_lam) * y[index]

        return x_new, y_new


class VHMixUpCallback(BaseMixCallback):
    """Callback that perform "Vertical Concat” and “Horizontal Concat” with mixup on the input
    batch.

    Reference:
     - `Improved Mixed-Example Data Augmentation <https://arxiv.org/abs/1805.11272>`_
     - `Principled Ultrasound Data Augmentation for Classification of Standard Planes <https://arxiv.org/abs/2103.07895>`_
    """

    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__(alpha=alpha, softmax_target=softmax_target, labels=labels)

    def mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        index = torch.randperm(x.size(0), device=x.device)
        x_perm = x[index]
        y_perm = y[index]

        x_combined = []
        y_combined = []

        c, h, w = x[0].shape
        for i in range(x.size(0)):
            lam = self.distribution(samples=3, device=x.device)
            border_h = int((h * lam[0]).round())
            border_w = int((w * lam[1]).round())

            mask_top_left = torch.zeros([h, w], device=x.device)
            mask_top_left[0:border_h, 0:border_w] = 1

            mask_top_right = torch.zeros([h, w], device=x.device)
            mask_top_right[0:border_h, border_w:w] = 1

            mask_bottom_left = torch.zeros([h, w], device=x.device)
            mask_bottom_left[border_h:h, 0:border_w] = 1

            mask_bottom_right = torch.zeros([h, w], device=x.device)
            mask_bottom_right[border_h:h, border_w:w] = 1

            x_combined.append(
                x[i] * mask_top_left
                + (lam[2] * x[i] + (1 - lam[2]) * x_perm[i]) * mask_top_right
                + ((1 - lam[2]) * x[i] + lam[2] * x_perm[i]) * mask_bottom_left
                + x_perm[i] * mask_bottom_right
            )

            y_lam = lam[2] * lam[0] + (1 - lam[2]) * lam[1]
            y_perm_lam = lam[2] * (1 - lam[0]) + (1 - lam[2]) * (1 - lam[1])
            if self.softmax_target:
                y_combined.append(torch.stack([y[i].float(), y_perm[i].float(), y_lam, y_perm_lam], dim=0))
            else:
                y_combined.append(y_lam * y[i] + y_perm_lam * y_perm[i])

        return torch.stack(x_combined), torch.stack(y_combined)
