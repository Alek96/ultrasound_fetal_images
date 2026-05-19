"""Unit tests for src/models/components/utils (get_model factory)."""

import pytest
import torch

from models.components.helpers import assert_output_shapes
from src.models.components.densenet import DenseNet
from src.models.components.efficientnet import EfficientNet
from src.models.components.mobilenet import MobileNet
from src.models.components.resnet import ResNet
from src.models.components.resnext import ResNeXt
from src.models.components.utils import get_model

BATCH = 2
OUTPUT_SIZE = 4
IMG = torch.zeros(BATCH, 1, 64, 64)

_ALL_MODELS = (
    [(name, DenseNet) for name in DenseNet.supported_models]
    + [(name, MobileNet) for name in MobileNet.supported_models]
    + [(name, ResNet) for name in ResNet.supported_models]
    + [(name, ResNeXt) for name in ResNeXt.supported_models]
    + [(name, EfficientNet) for name in EfficientNet.supported_models]
)


@pytest.mark.parametrize("model_name,expected_cls", _ALL_MODELS)
def test_factory_returns_correct_type(model_name: str, expected_cls):
    model = get_model(model_name, output_size=OUTPUT_SIZE, pretrain=False)
    assert isinstance(model, expected_cls)


@pytest.mark.parametrize("model_name,_", _ALL_MODELS)
def test_factory_forward_pass(model_name: str, _):
    model = get_model(model_name, output_size=OUTPUT_SIZE, pretrain=False)
    assert_output_shapes(model, img=IMG, batch=BATCH, output_size=OUTPUT_SIZE)


def test_factory_passes_kwargs_to_efficientnet():
    """dropout kwarg must be forwarded to EfficientNet only."""
    model = get_model("efficientnet_b0", output_size=OUTPUT_SIZE, pretrain=False, dropout=0.3)
    assert isinstance(model, EfficientNet)


def test_factory_unsupported_name_raises():
    with pytest.raises(KeyError):
        get_model("unsupported_model_xyz", output_size=OUTPUT_SIZE, pretrain=False)
