import itertools
from collections.abc import Callable
from typing import Any

import torch
import torchvision.transforms.v2 as T
from lightning import LightningModule
from torch import Tensor
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MaxMetric,
    MeanMetric,
    Metric,
)

from src.data.components.dataset import FetalBrainPlanesDataset
from src.data.components.transforms import Affine, HorizontalFlip, VerticalFlip
from src.models.components.utils import get_model
from src.models.utils.wandb import wandb_confusion_matrix
from src.utils.plots import log_to_wandb


class BrainPlanesLitModule(LightningModule):
    """A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net_spec: dict,
        num_classes: int,
        softmax_target: bool,
        vta_transforms: dict,
        tta_transforms: dict,
        criterion: torch.nn.Module,
        lr: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ):
        super().__init__()

        if softmax_target and criterion.keywords["reduction"] != "none":
            raise ValueError(
                "softmax_target=True requires criterion.reduction='none' for correct "
                "per-sample loss weighting; got reduction="
                f"'{self.criterion_fn.reduction}'."
            )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = get_model(**net_spec)

        # loss function
        self.criterion_fn = criterion()
        # softmax function for aggregation
        self.softmax = torch.nn.Softmax(dim=1)

        # metric
        self.train_loss = MeanMetric()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="none")

        self.val_loss = MeanMetric()
        self.val_base_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_base_acc_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
        self.val_base_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_tta_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_tta_acc_best = MaxMetric()
        self.val_tta_acc_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
        self.val_tta_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_tta_f1_best = MaxMetric()

        self.test_loss = MeanMetric()
        self.test_base_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_base_acc_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
        self.test_base_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_base_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="none")
        self.test_tta_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_tta_acc_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
        self.test_tta_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_tta_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="none")

        # tta
        self.vta_transforms = self.create_transforms(vta_transforms)
        self.tta_transforms = self.create_transforms(tta_transforms)

    @staticmethod
    def create_transforms(transforms: dict) -> list[Callable]:
        return [
            T.Compose(
                [
                    HorizontalFlip(flip=horizontal_flip),
                    VerticalFlip(flip=vertical_flips),
                    Affine(degrees=rotate_degree, translate=translate, scale=scale),
                ]
            )
            for horizontal_flip, vertical_flips, rotate_degree, translate, scale in itertools.product(
                transforms["horizontal_flips"] if "horizontal_flips" in transforms else [False],
                transforms["vertical_flips"] if "vertical_flips" in transforms else [False],
                transforms["rotate_degrees"] if "rotate_degrees" in transforms else [0.0],
                transforms["translates"] if "translates" in transforms else [(0.0, 0.0)],
                transforms["scales"] if "scales" in transforms else [1.0],
            )
        ]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.net(x)

    def forward_tta(self, x: Tensor, transforms: None | list[Callable] = None) -> tuple[Tensor, Tensor]:
        transforms = transforms or self.tta_transforms
        assert transforms, "transforms list must not be empty"

        y_hats = []
        logits_0 = None

        for transformer in transforms:
            augmented_x = transformer(x)
            _, logits = self.forward(augmented_x)
            if logits_0 is None:
                logits_0 = logits
            y_hat = self.softmax(logits)
            y_hats.append(y_hat)

        y_hat = torch.mean(torch.stack(y_hats, dim=1), dim=1)

        return logits_0, y_hat

    def model_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x, y = batch
        _, logits = self.forward(x)
        loss, y = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def criterion(self, y_hat, y) -> tuple[Tensor, Tensor]:
        if len(y.shape) == 1:
            # No mix-up (training/val/test without mix-up).
            # y has shape (batch_size,) — integer class indices.
            # torch.mean handles both reduction="none" (reduces (B,) → scalar)
            # and reduction="mean" (already a scalar, mean is a no-op).
            return torch.mean(self.criterion_fn(y_hat, y)), y

        if self.hparams.softmax_target:
            # Mix-up with softmax_target=True; requires criterion.reduction="none".
            # y has shape (batch_size, 4): [y_a, y_b, lam_a, lam_b].
            y_a = y[:, 0].long()
            y_b = y[:, 1].long()
            lam_a = y[:, 2]
            lam_b = y[:, 3]
            loss = torch.mean(lam_a * self.criterion_fn(y_hat, y_a) + lam_b * self.criterion_fn(y_hat, y_b))
            true_y = y_a
        else:
            # Mix-up with softmax_target=False.
            # y has shape (batch_size, num_classes) — blended one-hot labels.
            # torch.mean handles both reduction="none" (reduces (B,) → scalar)
            # and reduction="mean" (already a scalar, mean is a no-op).
            loss = torch.mean(self.criterion_fn(y_hat, y))
            true_y = torch.argmax(y, dim=1)

        return loss, true_y

    def model_tta_step(self, batch: tuple[Tensor, Tensor], transforms) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x, y = batch
        logits, y_hat = self.forward_tta(x, transforms)
        loss, y = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        tta_preds = torch.argmax(y_hat, dim=1)
        return loss, preds, tta_preds, y

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure *_best doesn't store accuracy from these checks
        self.val_tta_acc_best.reset()
        self.val_tta_f1_best.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called in the training loop at the very beginning of the epoch."""
        # reset train_cm before every run
        self.train_cm.reset()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss, weight=targets.shape[0])
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_cm.update(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def on_validation_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        self.val_base_acc_cm.reset()
        self.val_tta_acc_cm.reset()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        """
        loss, preds, tta_preds, targets = self.model_tta_step(batch, transforms=self.vta_transforms)

        # update and log metrics
        self.val_loss(loss, weight=targets.shape[0])
        self.val_base_acc(preds, targets)
        self.val_base_acc_cm.update(preds, targets)
        self.val_base_f1(preds, targets)
        self.val_tta_acc(tta_preds, targets)
        self.val_tta_acc_cm.update(tta_preds, targets)
        self.val_tta_f1(tta_preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/base/acc", self.val_base_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/base/f1", self.val_base_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tta/acc", self.val_tta_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tta/f1", self.val_tta_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": tta_preds, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        val_base_acc_brain = self.brain_acc(self.val_base_acc_cm)
        val_tta_acc_brain = self.brain_acc(self.val_tta_acc_cm)
        self.val_tta_acc_best(self.val_tta_acc.compute())
        self.val_tta_f1_best(self.val_tta_f1.compute())

        # log value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/base/acc_brain", val_base_acc_brain, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tta/acc_brain", val_tta_acc_brain, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tta/acc_best", self.val_tta_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/tta/f1_best", self.val_tta_f1_best.compute(), sync_dist=True, prog_bar=True)

    def on_train_end(self) -> None:
        """Lightning hook that is called when training ends."""
        self.log_confusion_matrix("train/conf", self.train_cm.compute())

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        self.test_base_acc_cm.reset()
        self.test_base_cm.reset()
        self.test_tta_acc_cm.reset()
        self.test_tta_cm.reset()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, tta_preds, targets = self.model_tta_step(batch, transforms=self.tta_transforms)

        # update and log metrics
        self.test_loss(loss, weight=targets.shape[0])
        self.test_base_acc(preds, targets)
        self.test_base_acc_cm.update(preds, targets)
        self.test_base_f1(preds, targets)
        self.test_base_cm.update(preds, targets)
        self.test_tta_acc(tta_preds, targets)
        self.test_tta_acc_cm.update(tta_preds, targets)
        self.test_tta_f1(tta_preds, targets)
        self.test_tta_cm.update(tta_preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/base/acc", self.test_base_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/base/f1", self.test_base_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tta/acc", self.test_tta_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tta/f1", self.test_tta_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": tta_preds, "targets": targets}

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        test_base_acc_brain = self.brain_acc(self.test_base_acc_cm)
        test_tta_acc_brain = self.brain_acc(self.test_tta_acc_cm)

        self.log("test/base/acc_brain", test_base_acc_brain, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tta/acc_brain", test_tta_acc_brain, on_step=False, on_epoch=True, prog_bar=True)
        self.log_confusion_matrix("test/base/conf", self.test_base_cm.compute())
        self.log_confusion_matrix("test/tta/conf", self.test_tta_cm.compute())

    def brain_acc(self, cm: Metric):
        confusion_matrix = cm.compute()
        return self.confusion_matrix_acc(confusion_matrix, [0, 1, 2])

    @staticmethod
    def confusion_matrix_acc(confusion_matrix, class_idx):
        """Return the macro-average recall for the given class indices.

        Expects a row-normalised confusion matrix (normalize="true"), where
        ``confusion_matrix[i][i]`` equals the recall (TP / (TP + FN)) for
        class ``i``.  The result is the unweighted mean of those recall values.

        :param confusion_matrix: Row-normalised confusion matrix tensor.
        :param class_idx: Sequence of class indices to include.
        :return: Mean recall across the specified classes.
        """
        true = torch.sum(torch.cat([confusion_matrix[i][i].view(1) for i in class_idx]))
        return true / len(class_idx)

    def log_confusion_matrix(self, name: str, confusion_matrix: Tensor):
        log_to_wandb(
            lambda: {
                name: wandb_confusion_matrix(
                    cm=confusion_matrix,
                    class_names=FetalBrainPlanesDataset.labels,
                    title=name,
                )
            },
            loggers=self.loggers,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
