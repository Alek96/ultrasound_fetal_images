import itertools
import pathlib
import shutil
from math import ceil
from typing import Callable, List, Tuple

import cv2
import hydra
import PIL
import pyrootutils
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.data.components.dataset import FetalBrainPlanesDataset
from src.data.components.transforms import Affine, HorizontalFlip
from src.models.fetal_module import FetalLitModule

log = utils.get_pylogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
horizontal_flips: List[bool]
rotate_degrees: List[float]
translates: List[Tuple[float, float]]
scales: List[float]
transforms: List[Callable]

horizontal_flips = [False, True]
rotate_degrees = [0]
translates = [(0.0, 0.0)]
scales = [1.0]

# horizontal_flips = [False, True]
# rotate_degrees = [-15, 0, 15]
# translates = [(0.0, 0.0)]
# scales = [1.0]

label_def = FetalBrainPlanesDataset.labels
model: LightningModule


def create_dataset(path: pathlib.Path):
    videos_path = path / "videos"
    # images_path = path / "labeled"
    # shutil.rmtree(path / "labeled", ignore_errors=True)

    videos = list(videos_path.iterdir())
    for i, video_path in enumerate(tqdm(videos, desc="Label videos")):
        y_hats = label_video(video_path)
        print(y_hats.shape)
        break


def label_video(video_path: pathlib.Path):
    y_hats = []
    vidcap = cv2.VideoCapture(str(video_path))
    for i, frame in enumerate(frame_iter(vidcap, "Label frames")):
        frame = PIL.Image.fromarray(frame)
        frame = TF.to_tensor(frame)
        frame = frame.to(model.device)
        frames = torch.cat([transform(frame).unsqueeze(0) for transform in transforms])

        with torch.no_grad():
            y = torch.zeros(frames.shape[0], dtype=torch.int64, device=model.device)
            y_hat, _, preds, _ = model.model_step((frames, y))
            y_hats.append(y_hat.unsqueeze(0))

    return torch.cat(y_hat)


def frame_iter(capture, description):
    def iterator():
        while capture.grab():
            yield capture.retrieve()[1]

    return tqdm(
        iterator(),
        desc=description,
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


@hydra.main(version_base="1.3", config_path="../scripts/configs", config_name="label_videos.yaml")
def main(cfg: DictConfig):
    global batch_size, model, transforms

    root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    checkpoint_file = sorted((root / cfg.model_path / "checkpoints").glob("epoch_*.ckpt"))[-1]
    log.info(f"Load model from <{checkpoint_file}>")
    model = FetalLitModule.load_from_checkpoint(checkpoint_file)
    # disable randomness, dropout, etc...
    model.eval()
    model.to(device)

    log.info(f"Instantiating transformations for image size <{cfg.image_height}/{cfg.image_width}>")
    transforms = [
        T.Compose(
            [
                T.Grayscale(),
                T.Resize((cfg.image_height, cfg.image_width)),
                HorizontalFlip(flip=horizontal_flip),
                Affine(degrees=rotate_degree, translate=translate, scale=scale),
                T.ConvertImageDtype(torch.float32),
            ]
        )
        for horizontal_flip, rotate_degree, translate, scale in itertools.product(
            horizontal_flips, rotate_degrees, translates, scales
        )
    ]

    log.info(f"Start creating dataset {cfg.dataset_dir}")
    path = root / "data" / f"{cfg.dataset_dir}"
    batch_size = cfg.batch_size
    create_dataset(path)


if __name__ == "__main__":
    main()
