import pathlib
import shutil

import cv2
import hydra
import PIL
import pyrootutils
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
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
from src.models.fetal_module import FetalLitModule

log = utils.get_pylogger(__name__)

labels = [
    "Other",
    "Maternal cervix",
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
]


def label_videos(model, transforms, path: pathlib.Path):
    videos_path = path / "videos"
    images_path = path / "labeled"
    videos = len(list(videos_path.iterdir()))
    for i, video_path in enumerate(videos_path.iterdir()):
        label_video(model, transforms, video_path, images_path, i + 1, videos)


def label_video(
    model, transforms, video_path: pathlib.Path, images_path: pathlib.Path, it: int, videos: int
):
    if not video_path.exists():
        print(f"path {video_path} not exist")

    vidcap = cv2.VideoCapture(str(video_path))
    for i, frame in enumerate(frame_iter(vidcap, f"label video {it}/{videos}")):
        label = label_frame(model, transforms, frame)
        img_path = images_path / video_path.stem / label / ("frame%d.jpg" % i)
        if not img_path.parent.exists():
            img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), frame)

    count_images(images_path / video_path.stem)


def frame_iter(capture, description):
    def iterator():
        while capture.grab():
            yield capture.retrieve()[1]

    return tqdm(
        iterator(),
        desc=description,
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def label_frame(model, transforms, frame):
    with torch.no_grad():
        frame = PIL.Image.fromarray(frame)
        frame = transforms(frame)
        frame = frame.unsqueeze(0)
        y = model(frame)
        pred = y.max(1).indices[0]
        return labels[pred]


def count_images(images_path: pathlib.Path):
    count = {}
    for label in labels:
        count[label] = 0
    for label_dir in images_path.iterdir():
        count[label_dir.name] = len(list(label_dir.iterdir()))
    print(count)


def count_all_images(path: pathlib.Path):
    images_path = path / "labeled"

    count = {}
    for label in labels:
        count[label] = 0
    for video_dir in images_path.iterdir():
        for label_dir in video_dir.iterdir():
            count[label_dir.name] += len(list(label_dir.iterdir()))

    for key, item in count.items():
        print(f"{key}: {item}")


@hydra.main(version_base="1.3", config_path="configs", config_name="label_videos.yaml")
def main(cfg: DictConfig):
    root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    log.info(f"Load model from <{cfg.model_path}>")
    checkpoint_file = str(root / cfg.model_path)
    model = FetalLitModule.load_from_checkpoint(checkpoint_file)
    # disable randomness, dropout, etc...
    model.eval()

    log.info(
        f"Instantiating transformations for image size <{cfg.image_height}/{cfg.image_width}>"
    )
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Grayscale(),
            T.Resize((cfg.image_height, cfg.image_width)),
            T.ConvertImageDtype(torch.float32),
        ]
    )

    log.info("Delete old labeling")
    path = root / "data" / "US_VIDEOS"
    shutil.rmtree(path / "labeled", ignore_errors=True)

    log.info("Start labeling")
    label_videos(model, transforms, path)

    log.info("Count all images")
    count_all_images(path)


if __name__ == "__main__":
    main()
