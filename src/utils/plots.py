from math import ceil
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import PIL
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import wandb
from tqdm import tqdm

from src.data.components.dataset import FetalBrainPlanesDataset


class PlotProbabilities:
    def __init__(
        self,
        enabled: bool = False,
        data_dir: str = "data/",
        video_dataset_dir: str = "US_VIDEOS",
        batch_size: int = 32,
        input_size: tuple[int, int] = (55, 80),
        min_probabilities: List[int] = (),
        probability_norm: float = 1.0,
    ):
        super().__init__()
        self.enabled = enabled
        self.data_dir = data_dir
        self.video_dataset_dir = video_dataset_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.min_probabilities = min_probabilities
        self.probability_norm = probability_norm
        self.label_names = FetalBrainPlanesDataset.labels

        self.counts = self._init_counts()
        self.transforms = T.Compose(
            [
                T.Grayscale(),
                T.Resize(self.input_size),
                T.ConvertImageDtype(torch.float32),
            ]
        )

    def _init_counts(self):
        counts = {}
        for min_probability in self.min_probabilities:
            count = {}
            for label in self.label_names:
                count[label] = 0
            counts[min_probability] = count
        return counts

    def label_video_dataset(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        if not self._skip(trainer):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.eval()
            model.to(device)

            self._label_videos(model)
            self._log_probabilities(model)

    def _skip(self, trainer: pl.Trainer) -> bool:
        return not self.enabled or bool(trainer.fast_dev_run)

    def _label_videos(self, model: pl.LightningModule):
        selected_path = Path(self.data_dir) / self.video_dataset_dir / "selected"
        videos = list(selected_path.iterdir())
        for i, frames_path in enumerate(tqdm(videos, desc="Label videos")):
            self._label_video(model, frames_path)

    def _label_video(self, model: pl.LightningModule, frames_path: Path):
        frames_paths = list(frames_path.iterdir())
        epochs = ceil(len(frames_paths) / self.batch_size)
        for i in range(epochs):
            frames = frames_paths[(i * self.batch_size) : ((i + 1) * self.batch_size)]
            self._label_frames(model, frames)

    def _label_frames(self, model: pl.LightningModule, frames):
        with torch.no_grad():
            frames = self._get_frames_tensor(frames)
            frames = frames.to(model.device)
            frames = self.transforms(frames)

            _, logits = model(frames)
            y_hats = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            self._count_labels(y_hats, preds)

    def _get_frames_tensor(self, frame_paths):
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            frame = PIL.Image.fromarray(frame)
            frame = TF.to_tensor(frame)
            frame = frame.unsqueeze(0)
            frames.append(frame)
        return torch.cat(frames)

    def _count_labels(self, y_hats, preds):
        for (y_hat, pred) in zip(y_hats, preds):
            for min_probability in self.min_probabilities:
                if self._is_acceptable(y_hat, pred, min_probability):
                    self.counts[min_probability][self.label_names[pred]] += 1

    def _is_acceptable(self, y_hat, pred, min_prob):
        prob = min_prob * self.probability_norm if pred < 3 else min_prob
        return y_hat[pred] > prob

    def _log_probabilities(self, model: pl.LightningModule):
        with plt.style.context("seaborn-v0_8-muted"):
            fig, ax = plt.subplots(figsize=(15, 8))

        for i, (min_prob, count) in enumerate(self.counts.items()):
            labels = list(count.keys())
            values = list(count.values())
            ax.bar(labels, values, label=min_prob)

        ax.legend()
        ax.set_title("Probabilities on video dataset")

        model.log_to_wandb(lambda: {"test/probabilities": wandb.Image(fig)})
