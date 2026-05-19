"""Unit tests for src/models/components/resnext."""

import pytest
import torch

from models.components.helpers import assert_output_shapes
from src.models.components.resnext import ResNeXt

BATCH = 2
OUTPUT_SIZE = 4
IMG = torch.zeros(BATCH, 1, 64, 64)


def _assert_output_shapes(model):
    assert_output_shapes(model, img=IMG, batch=BATCH, output_size=OUTPUT_SIZE)


@pytest.mark.parametrize("model_name", ResNeXt.supported_models)
def test_model_forward(model_name: str):
    model = ResNeXt(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    _assert_output_shapes(model)


@pytest.mark.parametrize("model_name", ResNeXt.supported_models)
def test_first_conv_is_1channel(model_name: str):
    model = ResNeXt(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    assert model.model.conv1.in_channels == 1


def test_unsupported_name_raises():
    with pytest.raises(AssertionError):
        ResNeXt(name="resnext999", output_size=OUTPUT_SIZE, pretrain=False)
