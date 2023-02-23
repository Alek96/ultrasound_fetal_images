import torch
from torch import nn
from torchvision.models import get_model


class EfficientNet(nn.Module):
    supported_efficientnet_models = ["efficientnet_b0"]
    supported_efficientnet_v2_models = ["efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"]
    supported_models = supported_efficientnet_models + supported_efficientnet_v2_models

    def __init__(
        self,
        name: str = "efficientnet_b0",
        output_size: int = 6,
        pretrain: bool = True,
    ):
        super().__init__()

        assert name in self.supported_models
        self.model = get_model(name=name, weights="DEFAULT" if pretrain else None)

        # input
        old_conv = self.model.features[0][0]
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
        conv.weight = nn.Parameter(torch.mean(old_conv.weight, dim=1, keepdim=True))
        self.model.features[0][0] = conv

        # output
        self.model.classifier[-1] = nn.Linear(
            in_features=self.model.classifier[-1].in_features,
            out_features=output_size,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    _ = EfficientNet()
