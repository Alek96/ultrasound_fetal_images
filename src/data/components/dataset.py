import os
from collections.abc import Callable, Sequence
from math import ceil
from pathlib import Path
from zipfile import ZipFile

import cv2
import gdown
import pandas as pd
import PIL
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.io import read_image


class Subset(Dataset):
    r"""Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.dataset[self.indices[idx[0]], idx[1]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class TransformDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rs = self.dataset[idx]

        if self.transform:
            if isinstance(idx, tuple):
                if idx[1] == 0:
                    rs = self.transform(rs)
            else:
                rs = (self.transform(rs[0]), rs[1])

        if self.target_transform:
            if isinstance(idx, tuple):
                if idx[1] == 1:
                    rs = self.target_transform(rs)
            else:
                rs = (rs[0], self.target_transform(rs[1]))

        return rs


class FetalBrainPlanesDataset(Dataset):
    labels = [
        "Trans-ventricular",
        "Trans-thalamic",
        "Trans-cerebellum",
        "Other",
        "Not A Brain",
    ]

    def __init__(
        self,
        data_dir: str,
        data_name: str = "FETAL_PLANES",
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.dataset_dir = f"{data_dir}/{data_name}"
        self.img_labels = self.load_img_labels(train)
        self.img_dir = f"{self.dataset_dir}/Images"
        self.transform = transform
        self.target_transform = target_transform

    def load_img_labels(self, train: bool):
        img_labels = pd.read_csv(f"{self.dataset_dir}/FETAL_PLANES_DB_data.csv", sep=";")
        img_labels = img_labels[img_labels["Train "] == (1 if train else 0)]
        img_labels = img_labels[["Image_name", "Patient_num", "Brain_plane"]]
        return img_labels.reset_index(drop=True)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, sub_idx = idx
            if sub_idx == 0:
                return self.get_image(idx)
            elif sub_idx == 1:
                return self.get_label(idx)

        return self.get_image(idx), self.get_label(idx)

    def get_image(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        img_path = os.path.join(self.img_dir, self.img_labels.Image_name[idx] + ".png")
        image = read_image(img_path)
        if image.shape[0] == 4:
            image = image[:3, :, :]

        if self.transform:
            image = self.transform(image)
        return image

    def get_label(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        label = self.img_labels.Brain_plane[idx]

        if self.target_transform:
            label = self.target_transform(label)
        return label


class FetalBrainPlanesSamplesDataset(FetalBrainPlanesDataset):
    google_file_id = "1Toy4M7BzGppjlQRURXdSVxgQI_jl3zA7"

    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
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
        gdown.download(id=FetalBrainPlanesSamplesDataset.google_file_id, output=zip_file, quiet=False)

        with ZipFile(zip_file, "r") as zObject:
            zObject.extractall(path=data_dir)

        os.remove(zip_file)


class VideoQualityDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "US_VIDEOS",
        train: bool = True,
        seq_len: int = 32,
        seq_step: int = None,
        reverse: bool = False,
        transform: bool = False,
        normalize: bool = False,
        target_transform: Callable | None = None,
        label_transform: Callable | None = None,
    ):
        self.train = train
        self.dataset_dir = Path(data_dir) / dataset_name / "data"
        self.data_dir = self.dataset_dir / ("train" if self.train else "test")
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.reverse = reverse
        self.transform = transform
        self.clips = self.load_clips()
        self.normalize = normalize
        self.std_mean = torch.load(f"{self.dataset_dir}/std_mean.pt")

        self.target_transform = target_transform
        self.label_transform = label_transform

    def load_clips(self):
        clips = []
        for video_path in sorted(self.data_dir.iterdir()):
            transforms = [transform_path.name for transform_path in sorted(video_path.iterdir())]

            transform_path = sorted(video_path.iterdir())[0]
            logits, quality, _ = torch.load(transform_path)

            seq_len = self.seq_len or len(quality)
            seq_step = self.seq_step or max(1, ceil(seq_len / 2))

            for from_idx in range(0, len(quality) - seq_len + 1, seq_step):
                to_idx = from_idx + seq_len
                clips.append((video_path.name, transforms, from_idx, to_idx, False))
                if self.train and self.reverse:
                    clips.append((video_path.name, transforms, from_idx, to_idx, True))

        return pd.DataFrame(clips, columns=["Video", "Transforms", "From", "To", "Flip"])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"list index {idx} out of range")

        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        transforms = self.clips.Transforms[idx]
        transform_idx = torch.randint(0, len(transforms), ()) if (self.train and self.transform) else 0

        video = self.data_dir / self.clips.Video[idx] / transforms[transform_idx]
        logits, quality, preds = torch.load(video)

        from_idx = self.clips.From[idx]
        to_idx = self.clips.To[idx]
        x = logits[from_idx:to_idx]
        y = quality[from_idx:to_idx]
        p = preds[from_idx:to_idx]

        if self.clips.Flip[idx]:
            x = torch.flip(x, dims=[0])
            y = torch.flip(y, dims=[0])
            p = torch.flip(p, dims=[0])

        if self.normalize is not None:
            x = (x - self.std_mean[1]) / self.std_mean[0]

        if self.target_transform:
            y = self.target_transform(y)
        if self.label_transform:
            p = self.label_transform(p)

        return x, y, p


class VideoQualityMemoryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "US_VIDEOS",
        train: bool = True,
        all_transforms: bool = True,
        seq_len: int = 32,
        normalize: bool = False,
        target_transform: Callable | None = None,
        label_transform: Callable | None = None,
    ):
        self.dataset_dir = Path(data_dir) / dataset_name / "data"
        self.data_dir = self.dataset_dir / ("train" if train else "test")
        self.all_transforms = all_transforms
        self.seq_len = seq_len
        self.data = {}
        self.clips = self.load_clips()
        self.normalize = normalize
        self.std_mean = torch.load(f"{self.dataset_dir}/std_mean.pt")

        self.target_transform = target_transform
        self.label_transform = label_transform

    def load_clips(self):
        clips = []
        for video_path in sorted(self.data_dir.iterdir()):
            for transform_path in sorted(video_path.iterdir()):
                logits, quality, pred = torch.load(transform_path)
                self.add_data(video_path.name, transform_path.name, logits, quality, pred)

                seq_len = self.seq_len or len(quality)
                for i in range(len(quality) - seq_len + 1):
                    clips.append((video_path.name, transform_path.name, i, i + seq_len))

                # for the testing we only need base transformation and base transformation
                if not self.all_transforms:
                    break

        return pd.DataFrame(clips, columns=["Video", "Transform", "From", "To"])

    def add_data(self, video, transform, logits, quality, pred):
        if video not in self.data:
            self.data[video] = {}
            self.data[video]["quality"] = quality
            self.data[video]["pred"] = pred

        self.data[video][transform] = logits

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"list index {idx} out of range")

        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        video = self.clips.Video[idx]
        logits = self.data[video][self.clips.Transform[idx]]
        quality = self.data[video]["quality"]
        preds = self.data[video]["pred"]

        from_idx = self.clips.From[idx]
        to_idx = self.clips.To[idx]
        x = logits[from_idx:to_idx]
        y = quality[from_idx:to_idx]
        p = preds[from_idx:to_idx]

        if self.normalize is not None:
            x = (x - self.std_mean[1]) / self.std_mean[0]

        if self.target_transform:
            y = self.target_transform(y)
        if self.label_transform:
            p = self.label_transform(p)

        return x, y, p


class USVideosFrameDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "US_VIDEOS",
        train: bool = True,
        transform: Callable | None = None,
    ):
        videos_dir = Path(data_dir) / dataset_name / "videos" / ("train" if train else "test")
        self.videos = sorted(videos_dir.iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_idx, frame_idx = idx

        if video_idx >= len(self):
            raise IndexError(f"Video index {idx} out of range")

        frame = self.read_frame(self.videos[video_idx], frame_idx)
        if self.transform:
            frame = self.transform(frame)
        return frame

    @staticmethod
    def read_frame(video_path: Path, frame_idx: int):
        cap = cv2.VideoCapture(str(video_path))

        # get total number of frames
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # check for valid frame number
        if frame_idx < 0 or frame_idx >= total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range for video {video_path.name}")

        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()

        cap.release()

        frame = PIL.Image.fromarray(frame)
        frame = TF.to_tensor(frame)
        return frame


class USVideosSsimFrameDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "US_VIDEOS",
        transform: Callable | None = None,
    ):
        data_dir = Path(data_dir) / dataset_name / "selected"
        self.items = self.find_images(data_dir)
        self.transform = transform

    @staticmethod
    def find_images(images_path: Path):
        images = []
        for video_dir in sorted(images_path.iterdir()):
            images.extend(sorted(video_dir.iterdir()))
        return images

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"list index {idx} out of range")

        img_path = self.items[idx]
        image = read_image(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image


class USVideosDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "US_VIDEOS",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        data_dir = Path(data_dir) / dataset_name / "labeled"
        images = self.find_images(data_dir)
        self.items = []
        for key, items in images.items():
            self.items.extend([(str(item), key) for item in items])

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
            raise IndexError(f"list index {idx} out of range")

        img_path, label = self.items[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
