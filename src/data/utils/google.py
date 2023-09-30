import os
from pathlib import Path
from zipfile import ZipFile

import gdown

from src import utils

log = utils.get_pylogger(__name__)


def download(data_dir: str | Path, data_name: str, google_file_id: str):
    dataset_dir = f"{data_dir}/{data_name}"
    if os.path.exists(dataset_dir):
        return

    zip_file = f"{data_dir}/{data_name}.zip"
    gdown.download(id=google_file_id, output=zip_file, quiet=False)

    with ZipFile(zip_file, "r") as zObject:
        zObject.extractall(path=str(data_dir))

    os.remove(zip_file)
