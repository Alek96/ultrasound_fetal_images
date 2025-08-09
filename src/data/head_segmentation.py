from typing import Any

import torch
import torchvision.transforms.v2 as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

from src.data.components.dataset import (
    HeadSegmentationDataset,
    HeadSegmentationSamplesDataset,
)


class HeadSegmentationDataModule(LightningDataModule):
    """A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        sample: bool = False,
        input_size: tuple[int, int] = (55, 80),
        train_transforms: list = None,
        test_transforms: list = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = HeadSegmentationSamplesDataset if sample else HeadSegmentationDataset

        if train_transforms is not None:
            self.train_transforms = T.Compose(train_transforms)
        else:
            self.train_transforms = T.Compose(
                [
                    T.Grayscale(),
                    # RandomPercentCrop(max_percent=20),
                    T.Resize(input_size, interpolation=T.InterpolationMode.NEAREST),
                    # T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                    # T.RandAugment(magnitude=11),
                    # T.TrivialAugmentWide(),
                    # T.AugMix(),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(1.0, 1.2)),
                    T.ToDtype(
                        dtype={
                            tv_tensors.Image: torch.float32,
                            tv_tensors.Mask: torch.float32,
                            "other": None,
                        },
                        scale=True),
                    # T.Normalize(mean=0.17, std=0.19),  # FetalBrain
                    # T.Normalize(mean=0.449, std=0.226),  # ImageNet
                ]
            )

        if test_transforms is not None:
            self.test_transforms = T.Compose(test_transforms)
        else:
            self.test_transforms = T.Compose(
                [
                    T.Grayscale(),
                    T.Resize(input_size, interpolation=T.InterpolationMode.NEAREST),
                    T.ToDtype(
                        dtype={
                            tv_tensors.Image: torch.float32,
                            tv_tensors.Mask: torch.float32,
                            "other": None,
                        },
                        scale=True),
                    # T.Normalize(mean=0.17, std=0.19),  # FetalBrain
                    # T.Normalize(mean=0.449, std=0.226),  # ImageNet
                ]
            )

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 2

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset(
                data_dir=self.hparams.data_dir,
                subset="train",
                transform=self.train_transforms,
            )
            self.data_val = self.dataset(
                data_dir=self.hparams.data_dir,
                subset="val",
                transform=self.test_transforms,
            )
            self.data_test = self.dataset(
                data_dir=self.hparams.data_dir,
                subset="test",
                transform=self.test_transforms,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size * 2,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size * 3,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = HeadSegmentationDataModule()
