import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class SonoNet(nn.Module):
    supported_models = ["SN16", "SN32", "SN64"]

    feature_cfg_dict = {
        "SN16": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128],
        "SN32": [32, 32, "M", 64, 64, "M", 128, 128, 128, "M", 256, 256, 256, "M", 256, 256, 256],
        "SN64": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    }

    def __init__(
        self,
        name: str = "SN64",
        output_size: int = 6,
        pretrain: bool = True,
    ):
        super().__init__()

        assert name in self.supported_models

        feature_cfg = self.feature_cfg_dict[name]
        self.features = self._make_feature_layers(feature_cfg, 1)

        feature_channels = feature_cfg[-1]
        adaption_channels = feature_channels // 2
        self.adaption = self._make_adaption_layer(feature_channels, adaption_channels, 14)

        if pretrain:
            weights_path = os.path.join(os.path.dirname(__file__), "weights", f"SonoNet{name[2:]}.pth")
            # weights_path = root / "src" / "models" / "components" / "weights" / f"SonoNet{name[2:]}.pth"
            self.load_weights(weights_path)
        else:
            self.apply(self._initialize_weights)

        self.adaption[3] = nn.Conv2d(adaption_channels, output_size, 1, bias=False)
        self.adaption[4] = nn.BatchNorm2d(output_size)
        self._initialize_weights(self.adaption[3])
        self._initialize_weights(self.adaption[4])

    def forward(self, x):
        x = self.features(x)

        x = self.adaption(x)
        y = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        # y = F.softmax(y, dim=1)

        return x, y

    @classmethod
    def _make_feature_layers(cls, feature_cfg, in_channels):
        layers = []
        conv_layers = []
        for v in feature_cfg:
            if v == "M":
                conv_layers.append(nn.MaxPool2d(2))
                layers.append(nn.Sequential(*conv_layers))
                conv_layers = []
            else:
                conv_layers.append(cls._conv_layer(in_channels, v))
                in_channels = v
        layers.append(nn.Sequential(*conv_layers))
        return nn.Sequential(*layers)

    @staticmethod
    def _conv_layer(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-4),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_adaption_layer(feature_channels, adaption_channels, num_labels):
        return nn.Sequential(
            nn.Conv2d(feature_channels, adaption_channels, 1, bias=False),
            nn.BatchNorm2d(adaption_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(adaption_channels, num_labels, 1, bias=False),
            nn.BatchNorm2d(num_labels),
        )

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def load_weights(self, weights_path):
        state = torch.load(weights_path)
        self.load_state_dict(state, strict=True)
