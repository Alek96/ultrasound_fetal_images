from pathlib import Path

import pytest
import torch

from src.data.fetal_planes import FetalPlanesDataModule


@pytest.mark.parametrize("batch_size", [16, 32])
def test_fetal_planes_datamodule(batch_size):
    data_dir = "data"

    dm = FetalPlanesDataModule(data_dir=data_dir, sample=True, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "FETAL_PLANES_SAMPLES").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
