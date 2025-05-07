import itertools
import shutil
from collections.abc import Callable
from math import ceil
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import PIL
import rootutils
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.data.components.dataset import FetalBrainPlanesDataset
from src.data.components.transforms import Affine, HorizontalFlip, VerticalFlip
from src.data.utils.google import download
from src.models.fetal_module import FetalLitModule

log = utils.get_pylogger(__name__)

horizontal_flips: list[bool]
vertical_flips: list[bool]
rotate_degrees: list[float]
translates: list[tuple[float, float]]
scales: list[float]
transforms: list[Callable]

label_def = FetalBrainPlanesDataset.labels
model: LightningModule
batch_size: int
window: int
gma_width: int
gma_window: int
gma_sigma: float
temperature: float


def create_dataset(videos_path: Path, dataset_path: Path):
    videos_path = videos_path / "videos"
    data_path = dataset_path / "data"
    plots_path = dataset_path / "plots"
    shutil.rmtree(data_path, ignore_errors=True)
    shutil.rmtree(plots_path, ignore_errors=True)
    data_path.mkdir()
    plots_path.mkdir()

    label_videos(videos_path=videos_path, data_path=data_path, plots_path=plots_path, sub_dir="train")
    label_videos(videos_path=videos_path, data_path=data_path, plots_path=plots_path, sub_dir="test")

    logits = load_logits(data_path=data_path / "train")
    save_std_mean(data_path=data_path, logits=logits)


def label_videos(videos_path: Path, data_path: Path, plots_path: Path, sub_dir: str):
    videos_path = videos_path / sub_dir
    data_path = data_path / sub_dir
    plots_path = plots_path / sub_dir
    data_path.mkdir()
    plots_path.mkdir()

    videos = sorted(videos_path.iterdir())
    for i, video_path in enumerate(tqdm(videos, desc="Label videos", position=0)):
        dense, y_hats = label_video(video_path)
        preds = torch.argmax(y_hats[0], dim=1)
        y_hats, quality = calculate_quality(y_hats)
        save_processed_video(data_path, video_path.stem, dense, quality, preds)
        save_quality_plot(plots_path, video_path.stem, y_hats, quality)


def label_video(video_path: Path):
    video_dense = []
    video_y_hats = []

    vidcap = cv2.VideoCapture(str(video_path))
    for frame in frame_iter(vidcap, "Label frames"):
        frame = PIL.Image.fromarray(frame)
        frame = TF.to_tensor(frame)
        frame = frame.to(model.device)

        batches = ceil(len(transforms) / batch_size)
        batch_dense = []
        batch_y_hats = []
        for i in range(batches):
            batch_transforms = transforms[i * batch_size : (i + 1) * batch_size]
            frames = torch.stack([transform(frame) for transform in batch_transforms])

            with torch.no_grad():
                dense, logits = model(frames)
                y_hats = F.softmax(logits, dim=1)

            batch_dense.append(dense.cpu())
            batch_y_hats.append(y_hats.cpu())

        video_dense.append(torch.cat(batch_dense, dim=0))
        video_y_hats.append(torch.cat(batch_y_hats, dim=0))

    return torch.stack(video_dense, dim=1), torch.stack(video_y_hats, dim=1)


def frame_iter(capture, description):
    def iterator():
        while capture.grab():
            yield capture.retrieve()[1]

    return tqdm(
        iterator(),
        desc=description,
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        position=1,
        leave=False,
    )


def calculate_quality(y_hats: Tensor):
    # select highest prediction
    pred = torch.argmax(y_hats, dim=2)
    y_hats = y_hats * F.one_hot(pred, num_classes=y_hats.shape[2])

    # remove predictions that are inconsistent
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            min_j = max(0, j - window)
            max_j = min(j + window + 1, pred.shape[1])

            if not torch.all(torch.eq(pred[i, min_j:max_j], pred[i, j])):
                y_hats[i, j, pred[i, j]] = 0

    # average of all transformations
    y_hats = torch.mean(y_hats, dim=0)

    # gaussian moving average
    distance = torch.arange(-gma_width, gma_width + 1, dtype=torch.float)
    gaussian = torch.exp(-(distance**2) / (2 * gma_sigma**2))

    weight_sum = torch.ones(y_hats.shape)
    weight_sum = F.pad(weight_sum, pad=(0, 0, gma_width, gma_width), mode="constant", value=0)
    weight_sum = weight_sum.unfold(dimension=0, size=gma_window, step=1)
    weight_sum = weight_sum * gaussian
    weight_sum = torch.nansum(weight_sum, dim=2)

    y_hats = F.pad(y_hats, pad=(0, 0, gma_width, gma_width), mode="constant", value=0)
    y_hats = y_hats.unfold(dimension=0, size=gma_window, step=1)
    y_hats = y_hats * gaussian
    y_hats = torch.nansum(y_hats, dim=2)

    y_hats = y_hats / weight_sum

    # (the best prediction - sum of the rest prediction)
    plates = y_hats[:, :3]
    quality = torch.amax(plates, dim=1)
    quality = (quality * 2) - torch.sum(y_hats, dim=1)
    zaro_mask = torch.eq(quality > 0, False)
    quality.masked_fill_(zaro_mask, 0.0)

    return y_hats, quality


def save_processed_video(data_path: Path, video: str, dense: Tensor, quality: Tensor, preds: Tensor):
    video_path = data_path / video
    video_path.mkdir()

    for i in range(len(dense)):
        torch.save([dense[i].clone(), quality, preds], f"{video_path}/{i:05d}.pt")


def save_quality_plot(plots_path: Path, video: str, y_hats: Tensor, quality: Tensor):
    fig, axes = plt.subplots(ncols=1, nrows=3, tight_layout=True, figsize=(10, 15))

    for i, label in enumerate(label_def):
        x, y = extract_nonzero_values(y_hats[:, i])
        axes[0].plot(x, y, "o", markersize=2, label=label)
        axes[0].legend()
    axes[1].plot(range(len(quality)), quality, "o", markersize=2, color="tab:gray")
    axes[2].plot(range(len(quality)), quality, color="tab:gray")

    for ax in axes:
        ax.set_xlim(left=0, right=len(quality))
        ax.set_ylim(bottom=0, top=1)

    fig.savefig(f"{plots_path}/{video}.jpg")
    plt.close()


def extract_nonzero_values(y_hats):
    x = []
    y = []
    for i, y_hat in enumerate(y_hats):
        if y_hat > 0:
            x.append(i)
            y.append(y_hat)
    return x, y


def load_logits(data_path: Path):
    dense = []
    for video_path in sorted(data_path.iterdir()):
        path = sorted(video_path.iterdir())[0]
        logits, _, _ = torch.load(path)
        dense.append(logits)
    return torch.cat(dense)


def save_std_mean(data_path: Path, logits):
    std_mean = torch.std_mean(logits, unbiased=False, dim=0)
    torch.save(std_mean, f"{data_path}/std_mean.pt")


def create_quality_dataset(cfg: DictConfig):
    global model, horizontal_flips, vertical_flips, rotate_degrees, translates, scales, transforms, batch_size, window, gma_width, gma_window, gma_sigma, temperature

    root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    checkpoint_file = sorted((root / cfg.model_path / "checkpoints").glob("epoch_*.ckpt"))[-1]
    log.info(f"Load model from <{checkpoint_file}>")
    model = FetalLitModule.load_from_checkpoint(checkpoint_file)
    # disable randomness, dropout, etc...
    model.eval()
    device = cfg.device if (cfg.device != "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log.info(f"Instantiating transformations for image size <{cfg.image_height}/{cfg.image_width}>")
    horizontal_flips = cfg.horizontal_flips
    vertical_flips = cfg.vertical_flips
    rotate_degrees = cfg.rotate_degrees
    translates = cfg.translates
    scales = cfg.scales
    transforms = [
        T.Compose(
            [
                T.Grayscale(),
                T.Resize(size=(cfg.image_height, cfg.image_width), antialias=False),
                HorizontalFlip(flip=horizontal_flip),
                VerticalFlip(flip=vertical_flip),
                Affine(degrees=rotate_degree, translate=translate, scale=scale),
                T.ConvertImageDtype(torch.float32),
            ]
        )
        for horizontal_flip, vertical_flip, rotate_degree, translate, scale in itertools.product(
            horizontal_flips, vertical_flips, rotate_degrees, translates, scales
        )
    ]

    data_dir = root / "data"
    if cfg.sample:
        cfg.video_dir = "US_VIDEOS_SAMPLES"
        cfg.dataset_dir = "US_VIDEOS_SAMPLES"
        log.info(f"Start downloading dataset {cfg.dataset_dir}")
        download(data_dir, cfg.dataset_dir, cfg.google_file_id)

    log.info(f"Start creating dataset {cfg.dataset_dir}")
    batch_size = cfg.batch_size
    window = cfg.window
    gma_width = cfg.gma_width
    gma_window = gma_width * 2 + 1
    gma_sigma = gma_window / 4
    temperature = cfg.temperature
    create_dataset(
        videos_path=data_dir / f"{cfg.video_dir}",
        dataset_path=data_dir / f"{cfg.dataset_dir}",
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="create_quality_dataset.yaml")
def main(cfg: DictConfig):
    create_quality_dataset(cfg)


if __name__ == "__main__":
    main()
