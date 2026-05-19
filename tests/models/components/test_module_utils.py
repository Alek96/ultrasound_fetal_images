"""Unit tests for src/models/components/module_utils."""

import torch.nn as nn

from src.models.components.module_utils import _freeze_model_layer, freeze_model_layers


class TestFreezeModelLayers:
    def test_freezes_target_params(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        freeze_model_layers(model, layers_name=["0"], freeze_batch_norm=True)
        for p in model[0].parameters():
            assert not p.requires_grad

    def test_leaves_other_params_trainable(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.Conv2d(4, 8, 3),
        )
        freeze_model_layers(model, layers_name=["0"], freeze_batch_norm=True)
        for p in model[1].parameters():
            assert p.requires_grad

    def test_respects_nested_path(self):
        """Dotted layer path (e.g. 'features.0') must resolve correctly."""

        class Nested(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Conv2d(4, 8, 3))

        model = Nested()
        freeze_model_layers(model, layers_name=["features.0"], freeze_batch_norm=True)
        for p in model.features[0].parameters():
            assert not p.requires_grad
        for p in model.features[1].parameters():
            assert p.requires_grad

    def test_skips_bn_when_false(self):
        model = nn.Sequential(
            nn.BatchNorm2d(4),
        )
        freeze_model_layers(model, layers_name=["0"], freeze_batch_norm=False)
        for p in model.parameters():
            assert p.requires_grad

    def test_freezes_bn_when_true(self):
        model = nn.Sequential(
            nn.BatchNorm2d(4),
        )
        freeze_model_layers(model, layers_name=["0"], freeze_batch_norm=True)
        for p in model.parameters():
            assert not p.requires_grad
