"""Unit tests for src/models/components/densenet.

All backbone models are instantiated with pretrain=False so no network access
"""

import pytest
import torch

from src.models.components.densenet import DenseNet
from tests.models.components.helpers import assert_output_shapes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 2
OUTPUT_SIZE = 4
IMG = torch.zeros(BATCH, 1, 64, 64)  # 1-channel grayscale, small resolution


def _assert_output_shapes(model, img=IMG, expected_output_size=OUTPUT_SIZE):
    assert_output_shapes(model, img=img, batch=BATCH, output_size=expected_output_size)


# ---------------------------------------------------------------------------
# DenseNet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", DenseNet.supported_models)
def test_model_forward(model_name: str):
    model = DenseNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    _assert_output_shapes(model)


@pytest.mark.parametrize("model_name", DenseNet.supported_models)
def test_first_conv_is_1channel(model_name: str):
    model = DenseNet(name=model_name, output_size=OUTPUT_SIZE, pretrain=False)
    assert model.model.features.conv0.in_channels == 1


def test_unsupported_name_raises():
    with pytest.raises(AssertionError):
        DenseNet(name="densenet999", output_size=OUTPUT_SIZE, pretrain=False)
