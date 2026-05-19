"""Unit tests for src/models/components/resnet."""

import pytest
import torch

from models.components.helpers import assert_output_shapes
from src.models.components.resnet import ResNet

BATCH = 2
OUTPUT_SIZE = 4
IMG = torch.zeros(BATCH, 1, 64, 64)


def _assert_output_shapes(model):
    assert_output_shapes(model, img=IMG, batch=BATCH, output_size=OUTPUT_SIZE)


@pytest.mark.parametrize("model_name", ResNet.supported_models)
def test_model_forward(model_name: str):
    model = ResNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    _assert_output_shapes(model)


@pytest.mark.parametrize("model_name", ResNet.supported_models)
def test_first_conv_is_1channel_when_not_frozen(model_name: str):
    model = ResNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=0)
    assert model.model.conv1.in_channels == 1


@pytest.mark.parametrize("model_name", ResNet.supported_models)
def test_freeze_layers_keeps_3channel_conv(model_name: str):
    model = ResNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=2)
    assert model.model.conv1.in_channels == 3


@pytest.mark.parametrize("model_name", ResNet.supported_models)
def test_freeze_layers_expands_channels_in_forward(model_name: str):
    """When freez_layers > 0 the model must accept 1-channel input via channel expansion."""
    model = ResNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=2)
    assert model.freez_layers is True
    _assert_output_shapes(model)


@pytest.mark.parametrize("model_name", ResNet.supported_models)
def test_frozen_layers_are_non_trainable(model_name: str):
    model = ResNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=2)
    for p in model.model.conv1.parameters():
        assert not p.requires_grad


@pytest.mark.parametrize("model_name", ResNet.supported_models)
def test_classifier_remains_trainable(model_name: str):
    model = ResNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=2)
    for p in model.classifier.parameters():
        assert p.requires_grad


def test_unsupported_name_raises():
    with pytest.raises(AssertionError):
        ResNet(name="resnet999", output_size=OUTPUT_SIZE, pretrain=False)
