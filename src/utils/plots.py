from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from tqdm import tqdm

from src.data.components.dataset import (
    FetalBrainPlanesDataset,
    USVideosFrameDataset,
    USVideosSsimFrameDataset,
    VideoQualityDataset,
    VideosSsimFrameDataset,
    batch_tensor,
)


class PlotExtras:
    def __init__(
        self,
        enabled: bool,
    ):
        super().__init__()
        self.enabled = enabled

    def run(self, trainer: Trainer, model: LightningModule) -> None:
        if not self._skip(trainer):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self._run(trainer, model)

    def _skip(self, trainer: Trainer) -> bool:
        return not self.enabled or bool(trainer.fast_dev_run)

    def _run(self, trainer: Trainer, model: LightningModule) -> None:
        pass


class PlotVideosProbabilities(PlotExtras):
    def __init__(
        self,
        enabled: bool,
        data_dir: str = "data/",
        video_dataset_dir: str = "US_VIDEOS",
        batch_size: int = 32,
        input_size: tuple[int, int] = (55, 80),
        min_probabilities: list[int] = (),
        probability_norm: float = 1.0,
    ):
        super().__init__(enabled)

        self.ssim_dataset = USVideosSsimFrameDataset(
            data_dir=data_dir,
            dataset_name=video_dataset_dir,
            transform=T.Compose(
                [
                    T.Grayscale(),
                    T.Resize(input_size, antialias=False),
                    T.ConvertImageDtype(torch.float32),
                ]
            ),
        )
        self.batch_size = batch_size
        self.min_probabilities = min_probabilities
        self.probability_norm = probability_norm
        self.label_names = FetalBrainPlanesDataset.labels

        self.counts = self._init_counts()

    def _init_counts(self):
        counts = {}
        for min_probability in self.min_probabilities:
            count = {}
            for label in self.label_names:
                count[label] = 0
            counts[min_probability] = count
        return counts

    def _run(self, trainer: Trainer, model: LightningModule) -> None:
        self._label_videos(model)
        self._log_probabilities(model)

    def _label_videos(self, model: LightningModule):
        for video in tqdm(self.ssim_dataset, desc="Label videos"):
            for frames in batch_tensor(video, self.batch_size):
                self._label_frames(model, frames)

    def _label_frames(self, model: LightningModule, frames: torch.Tensor):
        with torch.no_grad():
            frames = frames.to(model.device)

            _, y_hat = model.forward_tta(frames)
            preds = torch.argmax(y_hat, dim=1)
            self._count_labels(y_hat, preds)

    def _count_labels(self, y_hats, preds):
        for y_hat, pred in zip(y_hats, preds):
            for min_probability in self.min_probabilities:
                if self._is_acceptable(y_hat, pred, min_probability):
                    self.counts[min_probability][self.label_names[pred]] += 1

    def _is_acceptable(self, y_hat, pred, min_prob):
        prob = min_prob * self.probability_norm if pred < 3 else min_prob
        return y_hat[pred] > prob

    def _log_probabilities(self, model: LightningModule):
        with plt.style.context("seaborn-v0_8-muted"):
            fig, ax = plt.subplots(figsize=(15, 8))

        for i, (min_prob, count) in enumerate(self.counts.items()):
            labels = list(count.keys())
            values = list(count.values())
            ax.bar(labels, values, label=str(min_prob))

        ax.legend()
        ax.set_title("Probabilities on video dataset")

        log_to_wandb(lambda: {"test/probabilities": wandb.Image(fig)}, loggers=model.loggers)


class PlotVideoQuality(PlotExtras):
    def __init__(
        self,
        enabled: bool,
        data_dir: str,
        dataset_name: str = "US_VIDEOS",
        video_dataset: str = "US_VIDEOS",
        normalize: bool = False,
        min_quality: float = 0.3,
        samples: int = 5,
        beans: int = 10,
        img_size: list[int] = (165, 240),
    ):
        super().__init__(enabled)
        self.min_quality = min_quality
        self.samples = samples
        self.beans = beans
        self.img_size = img_size

        self.dataset = VideoQualityDataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            train=False,
            seq_len=0,
            normalize=normalize,
        )
        self.videos = USVideosFrameDataset(
            data_dir=data_dir,
            dataset_name=video_dataset,
            train=False,
            transform=torch.nn.Sequential(
                T.Grayscale(),
                T.Resize(self.img_size, antialias=False),
            ),
        )
        self.labels = FetalBrainPlanesDataset.labels

    def _run(self, trainer: Trainer, model: LightningModule) -> None:
        data = self.test_video_qualities(model)
        self.plot_quality(data, model)
        self.plot_best_planes(data, model)

    def test_video_qualities(self, model: LightningModule):
        data = []
        for i in range(len(self.dataset)):
            x, y, preds = self.dataset[i]
            x = x.to(device=model.device)
            y = y.to(device=model.device)
            with torch.no_grad():
                y_hat = model(x.unsqueeze(0)).squeeze()
            data.append((y.cpu(), y_hat.cpu(), preds))
        return data

    def plot_quality(self, data, model: LightningModule):
        nrows = len(data)
        fig, axes = plt.subplots(ncols=1, nrows=nrows, tight_layout=True, figsize=(10, 5 * nrows))

        for i, (y, y_hat, preds) in enumerate(data):
            x = list(range(len(y)))
            axes[i].plot(x, y, label="true", color="tab:gray")
            axes[i].plot(x, y_hat, label="predicted", color="tab:cyan")

            # for j in range(3):
            #     label = self.labels[j]
            #     mask = torch.ne(preds, j)
            #     y_hat_label = torch.masked_fill(y_hat, mask, 0)
            #     best_idx = torch.argmax(y_hat_label)
            #     axes[i].plot([best_idx], [y_hat_label[best_idx]], "o", label=label)

            for j, (label, color) in enumerate(zip(self.labels[:3], ["tab:blue", "tab:orange", "tab:green"])):
                y = preds.double().numpy()
                y[y != j] = np.nan
                y[y == j] = 0.002
                axes[i].plot(x, y, label=label, color=color)

            axes[i].legend()

        for ax in axes:
            ax.set_ylim(bottom=0, top=1)

        log_to_wandb(lambda: {"test/quality": wandb.Image(fig)}, loggers=model.loggers)

    def plot_best_planes(self, data, model: LightningModule):
        nrows = len(data)
        figsize = 1.5
        scale = self.img_size[0] / self.img_size[1]
        fig, axes = plt.subplots(
            ncols=1,
            nrows=nrows,
            squeeze=True,
            tight_layout=True,
            figsize=(figsize * self.samples, figsize * scale * nrows * 3),
            dpi=300,
        )

        # For each test video
        for i, (y, y_hat, preds) in enumerate(data):
            rows = []
            qualities = []
            true_qualities = []

            # For each label
            for j in range(3):
                mask = torch.ne(preds, j)
                y_hat_label = torch.masked_fill(y_hat, mask, 0)
                # y_hat_label[:25] = 0  # omit first 50 frames
                best = torch.argsort(y_hat_label, descending=True)
                # delete frames from other classes and low quality frames
                best = [idx.item() for idx in best if y_hat_label[idx] != 0 and y_hat_label[idx] > self.min_quality]

                bean_size = int(len(best) / self.beans) + 1
                samples = [idx for i, idx in enumerate(best) if i % bean_size == 0][: self.samples]

                # for each sample
                frames = [self.videos[i, idx] for idx in reversed(samples)]
                row = torch.cat(frames, dim=2)
                rows.append(row)

                qualities.extend([y_hat[idx].item() for idx in reversed(samples)])
                true_qualities.extend([y[idx].item() for idx in reversed(samples)])

            img = torch.cat(rows, dim=1)
            img = TF.to_pil_image(img)
            img = TF.to_grayscale(img)
            img = np.asarray(img)
            axes[i].imshow(img, cmap="gray")
            axes[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            for col_idx in range(self.samples):
                for label_idx in range(3):
                    axes[i].text(
                        self.img_size[1] * col_idx + self.img_size[1] - 10,
                        self.img_size[0] * label_idx + 10,
                        f"{qualities[label_idx * self.samples + col_idx]:.2f}",
                        size="large",
                        color="tab:cyan",
                        bbox={"pad": 0, "color": (0, 0, 0, 0.3)},
                        horizontalalignment="right",
                        verticalalignment="top",
                    )
                    axes[i].text(
                        self.img_size[1] * col_idx + self.img_size[1] - 10,
                        self.img_size[0] * label_idx + 40,
                        f"{true_qualities[label_idx * self.samples + col_idx]:.2f}",
                        size="large",
                        color="tab:gray",
                        bbox={"pad": 0, "color": (0, 0, 0, 0.3)},
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

        for row_idx in range(nrows):
            for label_idx in range(3):
                axes[row_idx].text(
                    -30,
                    self.img_size[0] * label_idx + self.img_size[0] / 2,
                    self.labels[label_idx].replace("-", "\n"),
                    rotation=90,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        log_to_wandb(lambda: {"test/samples": wandb.Image(fig)}, loggers=model.loggers)


def log_to_wandb(get_date: Callable[[], dict[str, Any]], loggers: list[Logger]) -> None:
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.log(get_date())
