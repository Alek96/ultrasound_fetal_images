import torch
from torch import nn
from torchvision.models import get_model


class ResNeXt(nn.Module):
    supported_models = ["resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"]

    def __init__(
        self,
        name: str = "resnext50_32x4d",
        output_size: int = 6,
        pretrain: bool = True,
    ):
        super().__init__()

        assert name in self.supported_models
        self.model = get_model(name=name, weights="DEFAULT" if pretrain else None)

        # input
        old_conv = self.model.conv1
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
        self.model.conv1 = conv

        # output
        self.classifier = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=output_size,
        )
        self.model.fc = nn.Identity()

    def forward(self, x):
        dense_logits = self.model(x)
        return dense_logits, self.classifier(dense_logits)


if __name__ == "__main__":
    _ = ResNeXt()
