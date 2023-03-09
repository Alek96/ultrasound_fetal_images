from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric


class QualityLitModule(LightningModule):
    """Example of LightningModule for Fetal classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        hidden_size = 512
        self.rnn = torch.nn.GRU(input_size=1280, hidden_size=hidden_size, num_layers=2, dropout=0.1, batch_first=True)

        self.fn = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid(),
        )

        # loss function
        self.criterion = torch.nn.MSELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_loss_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def forward(self, x: torch.Tensor):
        # x of (batch_size, seq_len, n_features) shape
        output, h_n = self.rnn(x)
        # output of (batch_size, seq_len, hidden_size) shape
        # h_n of (num_layers, batch_size, hidden_size) shape
        batch_size, seq_len, hidden_size = output.shape
        output = output.contiguous().view(-1, hidden_size)

        y_hat = self.fn(output)
        y_hat = y_hat.contiguous().view(batch_size, seq_len)

        return y_hat

    def model_step(self, batch: Any):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: list[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: list[Any]):
        acc = self.val_loss.compute()  # get current val acc
        self.val_loss_best(acc)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/lost_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: list[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
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
    _ = QualityLitModule(None, None, None)
