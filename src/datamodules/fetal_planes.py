from typing import Any, Dict, Optional

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.datamodules.components.dataset import (
    FetalPlanesDataset,
    TransformDataset,
    USVideosDataset,
)
from src.datamodules.components.transforms import LabelEncoder, RandomPercentCrop
from src.datamodules.utils import group_split


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
        train_val_split: float = 0.2,
        train_val_split_seed: float = 79,
        extend_train: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.input_size = (55, 80)
        self.train_transforms = T.Compose(
            [
                T.Grayscale(),
                # T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                # T.ConvertImageDtype(torch.float32),
                # T.RandomHorizontalFlip(p=0.5),
                # T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                # RandomPercentCrop(max_percent=20),
                # T.Resize(self.input_size),
                T.Resize(self.input_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(1.0, 1.2)),
                T.ConvertImageDtype(torch.float32),
            ]
        )
        self.test_transforms = T.Compose(
            [
                T.Grayscale(),
                T.ConvertImageDtype(torch.float32),
                T.Resize(self.input_size),
            ]
        )
        self.labels = [
            "Other",
            "Maternal cervix",
            "Fetal abdomen",
            "Fetal brain",
            "Fetal femur",
            "Fetal thorax",
        ]
        self.target_transform = LabelEncoder(labels=self.labels)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 6

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train = FetalPlanesDataset(
                data_dir=self.hparams.data_dir,
                train=True,
            )
            data_train, data_val, = group_split(
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
            if self.hparams.extend_train:
                self.data_train = ConcatDataset(
                    (
                        self.data_train,
                        USVideosDataset(
                            data_dir=self.hparams.data_dir,
                            max_images=5000,
                            transform=self.train_transforms,
                            target_transform=self.target_transform,
                        ),
                    )
                )
            self.data_val = TransformDataset(
                dataset=data_val,
                transform=self.test_transforms,
                target_transform=self.target_transform,
            )
            self.data_test = FetalPlanesDataset(
                data_dir=self.hparams.data_dir,
                train=False,
                transform=self.test_transforms,
                target_transform=self.target_transform,
            )

    def train_dataloader(self):
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

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "fetal_planes.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
