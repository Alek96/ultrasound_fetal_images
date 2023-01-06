import torch
from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
)


class DenseNet(nn.Module):
    def __init__(
        self,
        densenet_name: str = "densenet121",
        output_size: int = 6,
        pretrain: bool = True,
    ):
        super().__init__()

        if densenet_name == "densenet121":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrain else None)
        elif densenet_name == "densenet161":
            self.model = densenet161(weights=DenseNet161_Weights.DEFAULT if pretrain else None)
        elif densenet_name == "densenet169":
            self.model = densenet169(weights=DenseNet169_Weights.DEFAULT if pretrain else None)
        elif densenet_name == "densenet201":
            self.model = densenet201(weights=DenseNet201_Weights.DEFAULT if pretrain else None)

        # input
        old_conv = self.model.features.conv0
        conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )
        conv.weight = nn.Parameter(torch.mean(old_conv.weight, 1, True))
        self.model.features.conv0 = conv

        # output
        self.model.classifier = nn.Linear(
            in_features=self.model.classifier.in_features,
            out_features=output_size,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    _ = DenseNet()
