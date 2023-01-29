import os
from pathlib import Path
from typing import Callable, Optional
from zipfile import ZipFile

import gdown
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class TransformDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class FetalPlanesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str = "FETAL_PLANES",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        dataset_dir = f"{data_dir}/{data_name}"
        img_labels = pd.read_csv(f"{dataset_dir}/FETAL_PLANES_DB_data.csv", sep=";")
        img_labels = img_labels[img_labels["Train "] == (1 if train else 0)]
        img_labels = img_labels[["Image_name", "Patient_num", "Plane"]]
        self.img_labels = img_labels.reset_index(drop=True)

        self.img_dir = f"{dataset_dir}/Images"
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


class FetalPlanesSamplesDataset(FetalPlanesDataset):
    google_file_id = "1Toy4M7BzGppjlQRURXdSVxgQI_jl3zA7"

    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        data_name = "FETAL_PLANES_SAMPLES"
        self.download(data_dir, data_name)
        super().__init__(
            data_dir=data_dir,
            data_name=data_name,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )

    @staticmethod
    def download(data_dir, data_name):
        dataset_dir = f"{data_dir}/{data_name}"
        if os.path.exists(dataset_dir):
            return

        zip_file = f"{data_dir}/{data_name}.zip"
        gdown.download(id=FetalPlanesSamplesDataset.google_file_id, output=zip_file, quiet=False)

        with ZipFile(zip_file, "r") as zObject:
            zObject.extractall(path=data_dir)

        os.remove(zip_file)


class USVideosDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_images: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        data_dir = Path(data_dir) / "US_VIDEOS" / "labeled"
        images = self.find_images(data_dir)
        self.items = []
        for key, item in images.items():
            idxs = np.random.permutation(len(item))[:max_images]
            self.items.extend([(str(item[idx]), key) for idx in idxs])

        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def find_images(images_path: Path):
        images = {}
        for video_dir in images_path.iterdir():
            for label_dir in video_dir.iterdir():
                label = label_dir.name
                if label not in images:
                    images[label] = []
                images[label].extend(label_dir.iterdir())
        return images

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("list index out of range")

        img_path, label = self.items[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
