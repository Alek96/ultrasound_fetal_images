from collections.abc import Sequence
from typing import Any

import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.distributions.beta import Beta

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
            self.samples[target.item()][pred.item()].append(idx + i)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""
        dataset = get_test_dataloader(trainer).dataset

        images = []
        for row in self.samples:
            for cell in row:
                if len(cell) == 0:
                    images.append(None)
                else:
                    idx = torch.randint(0, len(cell), ()).item()
                    image_idx = cell[idx]
                    image = dataset[image_idx][0]
                    images.append(image)

        fig = show_pytorch_images(
            images=images,
            title="        Predicted",
            ylabel="Actual      ",
            cols_names=self.class_names,
            rows_names=self.class_names,
        )

        log_to_wandb(lambda: {"test/samples": wandb.Image(fig)}, loggers=pl_module.loggers)


class BaseMixCallback(Callback):
    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__()
        self.alpha = alpha
        # Beta(0, 0) is undefined; only instantiate when alpha > 0.
        self.distribution_fn = Beta(self.alpha, self.alpha) if self.alpha > 0 else None
        self.softmax_target = softmax_target
        self.one_hot_encoder = OneHotEncoder(labels)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        x, y = batch
        x_new, y_new = self._mix(x, y)

        batch[0] = x_new
        batch[1] = y_new

    def _mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def _distribution(self, samples: int, device) -> Tensor:
        if self.distribution_fn is None:
            return torch.ones(samples, device=device)
        x = self.distribution_fn.sample(torch.Size((samples,)))
        x = torch.stack([x, 1 - x], dim=1).amax(dim=1)
        return x.to(device)

    def _one_hot_encode(self, y: Tensor) -> Tensor:
        if len(y.shape) == 1:
            y = self.one_hot_encoder(y)
        return y


class MixUpCallback(BaseMixCallback):
    """Callback that performs MixUp augmentation on the input batch using a single shared λ.

    A **single** mixing coefficient λ is drawn from Beta(α, α) and clipped to [0.5, 1]
    via max(λ, 1 - λ).  That same scalar is applied to *every* image in the batch:

        x_new[i] = λ * x[i] + (1 - λ) * x[perm[i]]

    where ``perm`` is a random permutation of the batch indices.

    Targets are handled in two ways depending on ``softmax_target``:

    * ``softmax_target=False`` (default): integer labels are one-hot encoded, then
      blended the same way as images — ``y_new[i] = λ * y_oh[i] + (1 - λ) * y_oh[perm[i]]``.

    * ``softmax_target=True``: the raw label indices are kept and the mixed batch
      is stored as a ``(batch, 4)`` tensor — ``[y[i], y[perm[i]], λ, 1 - λ]`` —
      so the loss function can compute the weighted cross-entropy itself.

    Compared to `MixUpV2Callback`, this variant uses **one** λ for the whole
    batch rather than a per-sample draw, which makes the augmentation slightly less
    diverse but cheaper to compute.

    Reference:
     - `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_
     - `Facebook's Mixup-CIFAR10 <https://github.com/facebookresearch/mixup-cifar10>`_
    """

    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__(alpha=alpha, softmax_target=softmax_target, labels=labels)

    def _mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.alpha > 0:
            lam = self._distribution(samples=1, device=x.device).squeeze()
        else:
            lam = x.new_ones(())

        index = torch.randperm(x.size(0), device=x.device)

        # lam is a 0-dim scalar; it broadcasts correctly against (B, C, H, W).
        x_new = lam * x + (1 - lam) * x[index, :]

        if self.softmax_target:
            # Expand the scalar to a (B,) vector so torch.stack produces (B, 4).
            y_lam = lam.expand(y.size(0))
            y_new = torch.stack([y.float(), y[index].float(), y_lam, (1 - y_lam)], dim=1)
        else:
            y = self._one_hot_encode(y)
            y_new = lam * y + (1 - lam) * y[index]

        return x_new, y_new


class MixUpV2Callback(BaseMixCallback):
    """Callback that performs MixUp augmentation with a **per-sample** mixing coefficient.

    Unlike `MixUpCallback`, which draws a single scalar λ for the whole batch,
    this variant draws an independent λᵢ for every image:

        x_new[i] = λᵢ * x[i] + (1 - λᵢ) * x[perm[i]]

    Each λᵢ is sampled from Beta(α, α) and clipped to [0.5, 1] via max(λ, 1 - λ),
    resulting in a ``(batch,)`` vector.  The coefficient is then broadcast over the
    spatial and channel dimensions via ``reshape`` + ``expand``, so the same scalar
    is applied uniformly to all pixels of a given image.

    Targets follow the same two-mode logic as `MixUpCallback`:

    * ``softmax_target=False``: one-hot labels are blended element-wise —
      ``y_new[i] = λᵢ * y_oh[i] + (1 - λᵢ) * y_oh[perm[i]]``.

    * ``softmax_target=True``: output is ``(batch, 4)`` with
      ``[y[i], y[perm[i]], λᵢ, 1 - λᵢ]`` per row.

    The per-sample λ makes the augmentation more diverse and is closer to the
    fast.ai / pytorch-lightning-spells implementation.

    Reference:
     - `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_
     - `PyTorch Lightning Spells's implementation <https://github.com/veritable-tech/pytorch-lightning-spells/blob/master/pytorch_lightning_spells/callbacks.py>`_
     - `Fast.ai's implementation <https://github.com/fastai/fastai/blob/master/fastai/callback/mixup.py>`_
    """

    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__(alpha=alpha, softmax_target=softmax_target, labels=labels)

    def _mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.alpha > 0:
            lam = self._distribution(samples=x.size(0), device=x.device)
        else:
            lam = torch.ones(x.shape[:1], device=x.device)

        index = torch.randperm(x.size(0), device=x.device)

        # Broadcast lam over (C, H, W) so each image gets its own scalar coefficient.
        x_lam = lam.view(-1, *[1 for _ in range(len(x.shape[1:]))]).expand(-1, *x.shape[1:])
        x_new = x_lam * x + (1 - x_lam) * x[index]

        if self.softmax_target:
            y_lam = lam.view(-1, *[1 for _ in range(len(y.size()) - 1)]).expand(-1, *y.shape[1:])
            y_new = torch.stack([y.float(), y[index].float(), y_lam, (1 - y_lam)], dim=1)
        else:
            y = self._one_hot_encode(y)
            y_lam = lam.view(-1, *[1 for _ in range(len(y.size()) - 1)]).expand(-1, *y.shape[1:])
            y_new = y_lam * y + (1 - y_lam) * y[index]

        return x_new, y_new


class VHMixUpCallback(BaseMixCallback):
    """Callback that performs Vertical/Horizontal split MixUp on the input batch.

    Each image is divided into **four quadrants** by a random horizontal border
    (``border_h``) and a random vertical border (``border_w``), then the quadrants
    are filled as follows::

        ┌───────────────────┬────────────────────────┐
        │   x[i]  (pure)    │  lam_mix·x[i] +        │
        │  top-left         │  (1-lam_mix)·x_perm[i] │
        │                   │  top-right             │
        ├───────────────────┼────────────────────────┤
        │ (1-lam_mix)·x[i] +│  x_perm[i]  (pure)     │
        │  lam_mix·x_perm[i]│  bottom-right          │
        │  bottom-left      │                        │
        └───────────────────┴────────────────────────┘

    Three independent λ values are drawn from Beta(α, α) per sample:

    * ``lam_h`` — fraction of image height assigned to the top half (``border_h = round(H * lam_h)``).
    * ``lam_w`` — fraction of image width assigned to the left half (``border_w = round(W * lam_w)``).
    * ``lam_mix`` — pixel-level mixing coefficient for the two off-diagonal quadrants.

    The resulting label weight for ``x[i]`` is derived from the pixel-area contribution
    of each quadrant and simplifies to::

        y_lam      = lam_mix * lam_h + (1 - lam_mix) * lam_w
        y_perm_lam = 1 - y_lam

    Targets follow the same two-mode logic as `MixUpCallback`:

    * ``softmax_target=False``: one-hot labels are blended —
      ``y_new[i] = y_lam * y_oh[i] + y_perm_lam * y_oh[perm[i]]``.

    * ``softmax_target=True``: output is ``(batch, 4)`` with
      ``[y[i], y[perm[i]], y_lam, y_perm_lam]`` per row.

    Reference:
     - `Improved Mixed-Example Data Augmentation <https://arxiv.org/abs/1805.11272>`_
     - `Principled Ultrasound Data Augmentation for Classification of Standard Planes <https://arxiv.org/abs/2103.07895>`_
    """

    def __init__(self, alpha: float = 0.5, softmax_target: bool = False, labels: int = 0):
        super().__init__(alpha=alpha, softmax_target=softmax_target, labels=labels)

    def _mix(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        index = torch.randperm(x.size(0), device=x.device)
        x_perm = x[index]
        y_perm = y[index]

        # One-hot encode once before the loop so y/y_perm are not mutated mid-iteration.
        if not self.softmax_target:
            y = self._one_hot_encode(y)
            y_perm = self._one_hot_encode(y_perm)

        x_combined = []
        y_combined = []

        _, h, w = x[0].shape
        for i in range(x.size(0)):
            lam = self._distribution(samples=3, device=x.device)
            lam_h, lam_w, lam_mix = lam[0], lam[1], lam[2]

            border_h = int((h * lam_h).round())
            border_w = int((w * lam_w).round())

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
                + (lam_mix * x[i] + (1 - lam_mix) * x_perm[i]) * mask_top_right
                + ((1 - lam_mix) * x[i] + lam_mix * x_perm[i]) * mask_bottom_left
                + x_perm[i] * mask_bottom_right
            )

            # Area-weighted label contribution (derivation in class docstring).
            y_lam = lam_mix * lam_h + (1 - lam_mix) * lam_w
            y_perm_lam = lam_mix * (1 - lam_h) + (1 - lam_mix) * (1 - lam_w)

            if self.softmax_target:
                y_combined.append(torch.stack([y[i].float(), y_perm[i].float(), y_lam, y_perm_lam], dim=0))
            else:
                y_combined.append(y_lam * y[i] + y_perm_lam * y_perm[i])

        return torch.stack(x_combined), torch.stack(y_combined)
