import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

from src.datamodules.components.transforms import LabelEncoder
from src.datamodules.utils import group_split


class FetalPlanesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        data_dir = f"{data_dir}/FETAL_PLANES"
        img_labels = pd.read_csv(f"{data_dir}/FETAL_PLANES_DB_data.csv", sep=";")
        img_labels = img_labels[img_labels["Train "] == (1 if train else 0)]
        img_labels = img_labels[["Image_name", "Patient_num", "Plane"]]
        self.img_labels = img_labels.reset_index(drop=True)

        self.img_dir = f"{data_dir}/Images"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("list index out of range")

        img_path = os.path.join(self.img_dir, self.img_labels.Image_name[idx] + ".png")
        image = read_image(img_path)
        if image.shape[0] == 4:
            image = image[:3, :, :]
        if self.transform:
            image = self.transform(image)

        label = self.img_labels.Plane[idx]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


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
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((55, 80)),
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
                transform=self.transforms,
                target_transform=self.target_transform,
            )
            self.data_train, self.data_val, = group_split(
                dataset=train,
                test_size=self.hparams.train_val_split,
                groups=train.img_labels["Patient_num"],
                random_state=self.hparams.train_val_split_seed,
            )

            self.data_test = FetalPlanesDataset(
                data_dir=self.hparams.data_dir,
                train=False,
                transform=self.transforms,
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
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    for (x, y) in datamodule.test_dataloader():
        print(x.shape)
        print(y.shape)
        print(x.dtype)
        print(y.dtype)
        break
