from collections.abc import Sequence
from typing import Any, Literal

import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.components.dataset import (
    FetalBrainPlanesDataset,
    FetalBrainPlanesSamplesDataset,
    SsimFrameDataset,
    TransformDataset,
)
from src.data.components.transforms import LabelEncoder
from src.data.utils import group_split
from src.data.utils.utils import get_over_sampler, get_under_sampler


class FetalPlanesDataModule(LightningDataModule):
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
        train_val_split: float = 0.2,
        train_val_split_seed: float = 79,
        ssim: bool = False,
        ssim_dataset_name: str = "US_VIDEOS_ssim_0.6",
        ssim_min_probabilities: Sequence[float] = (0.8, 0.8, 0.8, 0.8, 0.8),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampler: Literal[None, "under", "over"] = None,
        sampler_max_sizes: Sequence[Sequence[int]] = ((-1, -1, -1, -1, 500),),
    ):
        super().__init__()

        if not ssim:
            assert len(sampler_max_sizes) == 1
        else:
            assert len(sampler_max_sizes) == 2

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = FetalBrainPlanesSamplesDataset if sample else FetalBrainPlanesDataset

        if train_transforms is not None:
            self.train_transforms = T.Compose(train_transforms)
        else:
            self.train_transforms = T.Compose(
                [
                    T.Grayscale(),
                    # RandomPercentCrop(max_percent=20),
                    T.Resize(input_size),
                    # T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                    T.RandAugment(magnitude=11),
                    # T.TrivialAugmentWide(),
                    # T.AugMix(),
                    T.RandomHorizontalFlip(p=0.5),
                    # T.RandomVerticalFlip(p=0.5),
                    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(1.0, 1.2)),
                    T.ConvertImageDtype(torch.float32),
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
                    T.Resize(input_size),
                    T.ConvertImageDtype(torch.float32),
                    # T.Normalize(mean=0.17, std=0.19),  # FetalBrain
                    # T.Normalize(mean=0.449, std=0.226),  # ImageNet
                ]
            )

        self.labels = FetalBrainPlanesDataset.labels
        self.target_transform = LabelEncoder(labels=self.labels)

        self.data_train_base: Dataset | None = None
        self.data_train_ssim: Dataset | None = None
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return len(self.labels)

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
            train = self.dataset(
                data_dir=self.hparams.data_dir,
                subset="train",
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
                subset="test",
                transform=self.test_transforms,
                target_transform=self.target_transform,
            )

            self.data_train_base = self.data_train
            if self.hparams.ssim:
                self.data_train_ssim = SsimFrameDataset(
                    data_dir=self.hparams.data_dir,
                    dataset_name=self.hparams.ssim_dataset_name,
                    min_probabilities=self.hparams.ssim_min_probabilities,
                    transform=self.train_transforms,
                    target_transform=self.target_transform,
                )
                self.data_train = ConcatDataset([self.data_train_base, self.data_train_ssim])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.hparams.sampler == "under":
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=False,
                sampler=get_under_sampler(
                    datasets=[self.data_train_base, self.data_train_ssim],
                    labels=torch.arange(self.num_classes),
                    # max_sizes=[[-1, -1, -1, -1, 500]],
                    max_sizes=self.hparams.sampler_max_sizes,
                ),
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
    _ = FetalPlanesDataModule()
