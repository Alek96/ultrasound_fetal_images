from typing import Any, Union

import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from torch import Tensor
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MaxMetric, MeanMetric

from src.data.components.dataset import FetalBrainPlanesDataset
from src.models.utils.wandb import wandb_confusion_matrix
from src.utils.plots import log_to_wandb


class HeadSegmentationLitModule(LightningModule):
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
        criterion: torch.nn.Module,
        lr: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None,
        )

        # loss function
        self.criterion_fn = criterion()

        # metric
        self.train_loss = MeanMetric()
        self.train_label_f1 = F1Score(task="binary")
        self.train_label_acc = Accuracy(task="binary")
        self.train_pixel_f1 = F1Score(task="binary")
        self.train_pixel_acc = Accuracy(task="binary")

        self.val_loss = MeanMetric()
        self.val_label_f1 = F1Score(task="binary")
        self.val_label_f1_best = MaxMetric()
        self.val_label_acc = Accuracy(task="binary")
        self.val_label_acc_best = MaxMetric()
        self.val_pixel_f1 = F1Score(task="binary")
        self.val_pixel_f1_best = MaxMetric()
        self.val_pixel_acc = Accuracy(task="binary")
        self.val_pixel_acc_best = MaxMetric()

        self.test_loss = MeanMetric()
        self.test_label_f1 = F1Score(task="binary")
        self.test_label_acc = Accuracy(task="binary")
        self.test_label_cm = ConfusionMatrix(task="binary", normalize="none")
        self.test_pixel_f1 = F1Score(task="binary")
        self.test_pixel_acc = Accuracy(task="binary")

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure *_best metrics didn't store accuracy from these checks
        self.val_label_f1_best.reset()
        self.val_label_acc_best.reset()
        self.val_pixel_f1_best.reset()
        self.val_pixel_acc_best.reset()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def model_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        images, masks, labels = batch
        logits = self.forward(images)
        loss = self.criterion_fn(logits, masks)

        # calculate prediction label
        prediction_mask = torch.sigmoid(logits)  # [B, 1, H, W], values 0-1
        binary_mask = (prediction_mask > 0.5).int()  # [B, 1, H, W], values 0 or 1
        binary_mask = binary_mask.squeeze(1)  # [B, H, W]
        total_pixels = binary_mask[0].numel()  # H * W
        ones_counts = binary_mask.sum(dim=(1, 2))  # [B]
        ones_percent = ones_counts.float() / total_pixels  # [B]
        prediction_label = (ones_percent >= 0.05).int()  # [B], 1 if >=5% ones, else 0

        return loss, logits, prediction_mask, masks, prediction_label, labels

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Union[Tensor, dict]:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, prediction_mask, masks, prediction_label, labels = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss, weight=logits.shape[0])
        self.train_label_f1(prediction_label, labels)
        self.train_label_acc(prediction_label, labels)
        self.train_pixel_f1(prediction_mask, masks)
        self.train_pixel_acc(prediction_mask, masks)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/label/f1", self.train_label_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/label/acc", self.train_label_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/pixel/f1", self.train_pixel_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/pixel/acc", self.train_pixel_acc, on_step=False, on_epoch=True, prog_bar=True)

        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def on_validation_start(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        """
        loss, logits, prediction_mask, masks, prediction_label, labels = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss, weight=logits.shape[0])
        self.val_label_f1(prediction_label, labels)
        self.val_label_acc(prediction_label, labels)
        self.val_pixel_f1(prediction_mask, masks)
        self.val_pixel_acc(prediction_mask, masks)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/label/f1", self.val_label_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/label/acc", self.val_label_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pixel/f1", self.val_pixel_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pixel/acc", self.val_pixel_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_label_f1_best(self.val_label_f1.compute())
        self.val_label_acc_best(self.val_label_acc.compute())
        self.val_pixel_f1_best(self.val_pixel_f1.compute())
        self.val_pixel_acc_best(self.val_pixel_acc.compute())

        # log a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/label/f1_best", self.val_label_f1_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/label/acc_best", self.val_label_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/pixel/f1_best", self.val_pixel_f1_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/pixel/acc_best", self.val_pixel_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        self.test_label_cm.reset()

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Any:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, prediction_mask, masks, prediction_label, labels = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss, weight=logits.shape[0])
        self.test_label_f1(prediction_label, labels)
        self.test_label_acc(prediction_label, labels)
        self.test_label_cm.update(prediction_label, labels)
        self.test_pixel_f1(prediction_mask, masks)
        self.test_pixel_acc(prediction_mask, masks)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/label/f1", self.test_label_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/label/acc", self.test_label_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pixel/f1", self.test_pixel_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pixel/acc", self.test_pixel_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log_confusion_matrix("test/pixel/conf", self.test_label_cm.compute())

    @staticmethod
    def confusion_matrix_acc(confusion_matrix, class_idx):
        true = torch.sum(torch.cat([confusion_matrix[i][i].view(1) for i in class_idx]))
        return true / len(class_idx)

    def log_confusion_matrix(self, name: str, confusion_matrix: Tensor, title: str | None = None):
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

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
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


if __name__ == "__main__":
    _ = HeadSegmentationLitModule(None, None, None, None)
