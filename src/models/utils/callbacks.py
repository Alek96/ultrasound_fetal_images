from collections.abc import Sequence
from typing import Any, Optional

import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from src.data.components.dataset import FetalBrainPlanesDataset
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
