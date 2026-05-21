"""Unit tests for src/models/components/mobilenet."""

import pytest
import torch

from src.models.components.mobilenet import MobileNet
from tests.models.components.helpers import assert_output_shapes

BATCH = 2
OUTPUT_SIZE = 4
IMG = torch.zeros(BATCH, 1, 64, 64)


def _assert_output_shapes(model):
    assert_output_shapes(model, img=IMG, batch=BATCH, output_size=OUTPUT_SIZE)


@pytest.mark.parametrize("model_name", MobileNet.supported_models)
def test_model_forward(model_name: str):
    model = MobileNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    _assert_output_shapes(model)


@pytest.mark.parametrize("model_name", MobileNet.supported_models)
def test_first_conv_is_1channel(model_name: str):
    model = MobileNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    assert model.model.features[0][0].in_channels == 1


def test_unsupported_name_raises():
    with pytest.raises(AssertionError):
        MobileNet(name="mobilenet_v99", output_size=OUTPUT_SIZE, pretrain=False)
