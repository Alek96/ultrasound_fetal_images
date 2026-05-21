"""Unit tests for src/models/components/efficientnet."""

import pytest
import torch

from src.models.components.efficientnet import EfficientNet
from tests.models.components.helpers import assert_output_shapes

BATCH = 2
OUTPUT_SIZE = 4
IMG = torch.zeros(BATCH, 1, 64, 64)


def _assert_output_shapes(model):
    assert_output_shapes(model, img=IMG, batch=BATCH, output_size=OUTPUT_SIZE)


@pytest.mark.parametrize("model_name", EfficientNet.supported_models)
def test_model_forward(model_name: str):
    model = EfficientNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    _assert_output_shapes(model)


@pytest.mark.parametrize("model_name", EfficientNet.supported_models)
def test_first_conv_is_1channel_when_not_frozen(model_name: str):
    model = EfficientNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=0)
    assert model.model.features[0][0].in_channels == 1


@pytest.mark.parametrize("model_name", EfficientNet.supported_models)
def test_freeze_layers_expands_channels_in_forward(model_name: str):
    """When freez_layers > 0 the model must accept 1-channel input via channel expansion."""
    model = EfficientNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=2)
    assert model.freez_layers is True
    _assert_output_shapes(model)


@pytest.mark.parametrize("model_name", EfficientNet.supported_models)
def test_frozen_layers_are_non_trainable(model_name: str):
    model = EfficientNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=2)
    for p in model.model.features[0].parameters():
        assert not p.requires_grad


@pytest.mark.parametrize("model_name", EfficientNet.supported_models)
def test_classifier_remains_trainable(model_name: str):
    model = EfficientNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False, freez_layers=2)
    for p in model.classifier.parameters():
        assert p.requires_grad


def test_unsupported_name_raises():
    with pytest.raises(AssertionError):
        EfficientNet(name="efficientnet_b99", output_size=OUTPUT_SIZE, pretrain=False)
