from typing import Any, Dict, List, Sequence

import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

from src.data.components.dataset import FetalBrainPlanesDataset
from src.data.utils.utils import show_pytorch_images


class ClassImageSampler(Callback):
    def __init__(
        self,
        class_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.class_names: Sequence[str] = class_names
        self.samples: List[List[List[int]]] = torch.zeros((len(self.class_names), len(self.class_names), 0)).tolist()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        preds = outputs["preds"]
        targets = outputs["targets"]
        batch_size = trainer.test_dataloaders[dataloader_idx].batch_size

        idx = batch_size * batch_idx
        for i, (target, pred) in enumerate(zip(targets, preds)):
            self.samples[target][pred].append(idx + i)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""
        dataset = trainer.test_dataloaders[0].dataset

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

        pl_module.log_to_wandb(lambda: {"test/samples": wandb.Image(fig)})
