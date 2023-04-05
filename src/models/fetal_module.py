from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import ConfusionMatrix, F1Score, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.data.components.dataset import FetalBrainPlanesDataset
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
        self.criterion = criterion()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_acc_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
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
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

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

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_acc_cm.update(preds, targets)
        self.test_f1(preds, targets)
        self.test_cm.update(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
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
