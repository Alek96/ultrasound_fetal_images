from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.data.components.dataset import (
    FetalBrainPlanesDataset,
    FetalBrainPlanesSamplesDataset,
    TransformDataset,
    USVideosDataset,
)
from src.data.components.transforms import LabelEncoder, RandomPercentCrop
from src.data.utils import group_split
from src.data.utils.utils import get_over_sampler, get_under_sampler


class FetalPlanesDataModule(LightningDataModule):
    """A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        sample: bool = False,
        input_size: tuple[int, int] = (55, 80),
        train_val_split: float = 0.2,
        train_val_split_seed: float = 79,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampler: Literal[None, "under", "over"] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = FetalBrainPlanesSamplesDataset if sample else FetalBrainPlanesDataset

        # data transformations
        self.train_transforms = T.Compose(
            [
                T.Grayscale(),
                # RandomPercentCrop(max_percent=20),
                T.Resize(input_size),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                # T.RandAugment(),
                # T.TrivialAugmentWide(),
                # T.AugMix(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # T.RandomAffine(degrees=0, translate=(0, 0), scale=(1.0, 1.2)),
                # T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(1.0, 1.2)),
                # T.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(1.0, 1.2)),
                # T.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(1.0, 1.2)),
                T.ConvertImageDtype(torch.float32),
                # T.Normalize(mean=0.17, std=0.19),  # FetalBrain
                # T.Normalize(mean=0.449, std=0.226),  # ImageNet
            ]
        )
        self.test_transforms = T.Compose(
            [
                T.Grayscale(),
                T.Resize(input_size),
                T.ConvertImageDtype(torch.float32),
                # T.Normalize(mean=0.17, std=0.19),  # FetalBrain
                # T.Normalize(mean=0.449, std=0.226),  # ImageNet
            ]
        )
        self.labels = FetalBrainPlanesDataset.labels
        self.target_transform = LabelEncoder(labels=self.labels)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    @property
    def num_classes(self):
        return len(self.labels)

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: str | None = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train = self.dataset(
                data_dir=self.hparams.data_dir,
                train=True,
            )
            data_train, data_val = group_split(
                dataset=train,
                test_size=self.hparams.train_val_split,
                groups=train.img_labels["Patient_num"],
                random_state=self.hparams.train_val_split_seed,
            )
            self.data_train = TransformDataset(
                dataset=data_train,
                transform=self.train_transforms,
                target_transform=self.target_transform,
            )
            self.data_val = TransformDataset(
                dataset=data_val,
                transform=self.test_transforms,
                target_transform=self.target_transform,
            )
            self.data_test = self.dataset(
                data_dir=self.hparams.data_dir,
                train=False,
                transform=self.test_transforms,
                target_transform=self.target_transform,
            )

    def train_dataloader(self):
        if self.hparams.sampler == "under":
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler=get_under_sampler(self.data_train),
            )
        elif self.hparams.sampler == "over":
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler=get_over_sampler(self.data_train),
            )
        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = FetalPlanesDataModule()
