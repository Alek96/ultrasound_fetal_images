import pathlib
import shutil
from collections.abc import Callable
from math import ceil

import cv2
import hydra
import PIL
import rootutils
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
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
from src.models.fetal_module import FetalLitModule

log = utils.get_pylogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
label_def = FetalBrainPlanesDataset.labels
batch_size: int
min_prob: float
prob_norm: float
transforms: Callable
model: LightningModule


def label_videos(path: pathlib.Path):
    selected_path = path / "selected"
    images_path = path / "labeled"
    videos = list(selected_path.iterdir())
    for i, frames_path in enumerate(tqdm(videos, desc="Label videos")):
        label_video(frames_path, images_path)


def label_video(frames_path: pathlib.Path, images_path: pathlib.Path):
    imgs_path = images_path / frames_path.stem
    shutil.rmtree(imgs_path, ignore_errors=True)
    imgs_path.mkdir(parents=True)

    frames_paths = list(frames_path.iterdir())

    epochs = ceil(len(frames_paths) / batch_size)
    for i in tqdm(range(epochs), desc="Label video", leave=False):
        frames = frames_paths[(i * batch_size) : ((i + 1) * batch_size)]
        labels = label_frames(frames)

        for frame_path, label in zip(frames, labels):
            if label:
                img_path = imgs_path / label / f"{frame_path.stem}.jpg"
                if not img_path.parent.exists():
                    img_path.parent.mkdir(parents=True)
                shutil.copy(frame_path, img_path)


def label_frames(frames):
    with torch.no_grad():
        frames = get_frames_tensor(frames)
        frames = frames.to(device)
        frames = transforms(frames)

        y = torch.zeros(frames.shape[0], dtype=torch.int64, device=model.device)
        y_hats, _, preds, _ = model.model_step((frames, y))
        return [label_def[pred] if is_reliable(y_hat, pred) else None for y_hat, pred in zip(y_hats, preds)]


def is_reliable(y_hat, pred):
    prob = min_prob * prob_norm if pred < 3 else min_prob
    return y_hat[pred] > prob


def get_frames_tensor(frame_paths):
    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        frame = PIL.Image.fromarray(frame)
        frame = F.to_tensor(frame)
        frame = frame.unsqueeze(0)
        frames.append(frame)
    return torch.cat(frames)


def count_labeled_images(path: pathlib.Path):
    images_path = path / "labeled"

    count = {}
    for label in label_def:
        count[label] = 0
    for video_dir in images_path.iterdir():
        for label_dir in video_dir.iterdir():
            count[label_dir.name] += len(list(label_dir.iterdir()))

    for key, item in count.items():
        log.info(f"{key}: {item}")
    log.info(f"Total: {sum(count.values())} / {count_selected_images(path)}")


def count_selected_images(path: pathlib.Path):
    images_path = path / "selected"
    count = 0
    for video_dir in images_path.iterdir():
        count += len(list(video_dir.iterdir()))
    return count


def find_latest_experiment(path: pathlib.Path):
    experiments = path / "logs" / "train" / "runs"
    experiment = sorted(experiments.iterdir())[-1]
    return experiment


@hydra.main(version_base="1.3", config_path="configs", config_name="label_videos.yaml")
def main(cfg: DictConfig):
    global batch_size, min_prob, prob_norm, model, transforms

    root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    model_path = cfg.model_path
    if model_path is None or model_path == "":
        model_path = find_latest_experiment(root)
    else:
        model_path = root / model_path

    log.info(f"Load model from <{model_path}>")
    checkpoint_file = sorted(model_path.glob("checkpoints/epoch_*.ckpt"))[-1]
    model = FetalLitModule.load_from_checkpoint(checkpoint_file)
    # disable randomness, dropout, etc...
    model.eval()
    model.to(device)

    log.info(f"Instantiating transformations for image size <{cfg.image_height}/{cfg.image_width}>")
    transforms = T.Compose(
        [
            T.Grayscale(),
            T.Resize((cfg.image_height, cfg.image_width)),
            T.ConvertImageDtype(torch.float32),
        ]
    )

    log.info(f"Delete old labeling from data/{cfg.video_dataset_dir}")
    path = root / "data" / f"{cfg.video_dataset_dir}"
    shutil.rmtree(path / "labeled", ignore_errors=True)

    log.info(f"Start labeling with min_prob {cfg.min_prob} and prob_norm {cfg.prob_norm}")
    batch_size = cfg.batch_size
    min_prob = cfg.min_prob
    prob_norm = cfg.prob_norm
    label_videos(path)

    log.info("Count all images")
    count_labeled_images(path)


if __name__ == "__main__":
    main()
