from collections.abc import Sequence
from typing import Any, Optional

import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

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


class MixUpCallback(Callback):
    """Callback that perform MixUp augmentation on the input batch.

    Assumes the first dimension is batch.

    Works best with pytorch_lightning_spells.losses.MixupSoftmaxLoss

    Reference: `Fast.ai's implementation <https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py>`_
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False, labels: int = 0):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target
        self.one_hot_encoder = OneHotEncoder(labels)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        if self.alpha <= 0:
            return

        x, y = batch
        if len(y.shape) == 1:
            y = self.one_hot_encoder(y)

        index = torch.randperm(x.size(0), device=x.device)
        x_perm = x[index]
        y_perm = y[index]

        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample(x.shape[:1]).to(x.device)
        lam = torch.stack([lam, 1 - lam], dim=1).amax(dim=1)

        # Create the tensor and expand (for batch inputs)
        x_lam = lam.view(-1, *[1 for _ in range(len(x.shape[1:]))]).expand(-1, *x.shape[1:])
        # Combine input batch
        x_new = x * x_lam + x_perm * (1 - x_lam)

        # Create the tensor and expand (for target)
        y_lam = lam.view(-1, *[1 for _ in range(len(y.size()) - 1)]).expand(-1, *y.shape[1:])
        # Combine targets
        if self.softmax_target:
            y_new = torch.stack([y.float(), y.flip(0).float(), y_lam], dim=1)
        else:
            y_new = y * y_lam + y_perm * (1 - y_lam)

        batch[0] = x_new
        batch[1] = y_new


class VHMixUpCallback(Callback):
    """Callback that perform "Vertical Concat” and “Horizontal Concat” with mixup on the input
    batch.

    Assumes the first dimension is batch.

    Works best with pytorch_lightning_spells.losses.MixupSoftmaxLoss

    Reference: `Fast.ai's implementation <https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py>`_
    """

    def __init__(self, alpha: float = 0.4, labels: int = 0):
        super().__init__()
        self.alpha = alpha
        self.one_hot_encoder = OneHotEncoder(labels)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        if self.alpha <= 0:
            return

        x, y = batch
        if len(y.shape) == 1:
            y = self.one_hot_encoder(y)

        index = torch.randperm(x.size(0), device=x.device)
        x_perm = x[index]
        y_perm = y[index]

        x_combined = []
        y_combined = []

        c, h, w = x[0].shape
        for i in range(x.size(0)):
            lam = torch.distributions.beta.Beta(0.9, 0.9).sample([3]).to(x.device)
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

            y_combined.append(
                (lam[2] * lam[0] + (1 - lam[2]) * lam[1]) * y[i]
                + (lam[2] * (1 - lam[0]) + (1 - lam[2]) * (1 - lam[1])) * y_perm[i]
            )

        batch[0] = torch.stack(x_combined)
        batch[1] = torch.stack(y_combined)
