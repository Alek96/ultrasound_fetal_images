import itertools
from typing import Any, List
from collections.abc import Callable

import torch
import torchvision.transforms as T
from lightning import LightningModule
from torchmetrics import ConfusionMatrix, F1Score, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.data.components.dataset import FetalBrainPlanesDataset
from src.data.components.transforms import Affine, HorizontalFlip, VerticalFlip
from src.models.components.utils import get_model
from src.models.utils.wandb import wandb_confusion_matrix
from src.utils.plots import log_to_wandb


class FetalLitModule(LightningModule):
    """Example of LightningModule for Fetal classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = get_model(**net_spec)

        # loss function
        self.criterion_fn = criterion()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
        self.val_acc_tta = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_acc_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
        self.test_acc_tta = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for tracking confusion matrix
        self.train_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="none")
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="none")

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

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        # reset train_cm before every run
        self.train_cm.reset()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):
        x, y = batch
        _, logits = self.forward(x)
        loss, y = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def criterion(self, y_hat, y):
        if len(y.shape) == 1:
            return torch.mean(self.criterion_fn(y_hat, y)), y

        if self.hparams.softmax_target:
            y_a = y[:, 0].long()
            y_b = y[:, 1].long()
            lam_a = y[:, 2]
            lam_b = y[:, 3]
            loss = torch.mean(lam_a * self.criterion_fn(y_hat, y_a) + lam_b * self.criterion_fn(y_hat, y_b))
            true_y = y_a
        else:
            loss = torch.mean(self.criterion_fn(y_hat, y))
            true_y = torch.argmax(y, dim=1)

        return loss, true_y

    def model_tta_step(self, batch: Any, transforms):
        x, _ = batch

        result = []
        for transformer in transforms:
            augmented_x = transformer(x)
            _, logits = self.forward(augmented_x)
            result.append(logits)

        logits = torch.mean(torch.stack(result, dim=1), dim=1)
        preds = torch.argmax(logits, dim=1)

        return preds

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_cm.update(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        pass

    def on_validation_start(self):
        self.val_acc_cm.reset()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, targets = self.model_step(batch)
        tta_preds = self.model_tta_step(batch, transforms=self.vta_transforms)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_acc_cm.update(preds, targets)
        self.val_acc_tta(tta_preds, targets)
        self.val_f1(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_tta", self.val_acc_tta, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        confusion_matrix = self.val_acc_cm.compute()
        val_acc_brain_planes = self.confusion_matrix_acc(confusion_matrix, [0, 1, 2])

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.log("val/acc_brain_planes", val_acc_brain_planes, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_start(self) -> None:
        self.test_acc_cm.reset()
        self.test_cm.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        tta_preds = self.model_tta_step(batch, transforms=self.tta_transforms)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_acc_cm.update(preds, targets)
        self.test_acc_tta(tta_preds, targets)
        self.test_f1(preds, targets)
        self.test_cm.update(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_tta", self.test_acc_tta, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        confusion_matrix = self.test_acc_cm.compute()
        test_acc_brain_planes = self.confusion_matrix_acc(confusion_matrix, [0, 1, 2])
        self.log("test/acc_brain_planes", test_acc_brain_planes, on_step=False, on_epoch=True, prog_bar=True)
        self.log_confusion_matrix("train/conf", self.train_cm.compute())
        self.log_confusion_matrix("test/conf", self.test_cm.compute())

    @staticmethod
    def confusion_matrix_acc(confusion_matrix, class_idx):
        true = torch.sum(torch.cat([confusion_matrix[i][i].view(1) for i in class_idx]))
        return true / len(class_idx)

    def log_confusion_matrix(self, name: str, confusion_matrix: torch.Tensor, title: str | None = None):
        log_to_wandb(
            lambda: {
                name: wandb_confusion_matrix(
                    cm=confusion_matrix,
                    class_names=FetalBrainPlanesDataset.labels,
                    title=title,
                )
            },
            loggers=self.loggers,
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
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


if __name__ == "__main__":
    _ = FetalLitModule(None, None, None, None)
