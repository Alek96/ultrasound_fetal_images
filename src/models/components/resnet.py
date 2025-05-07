import torch
from torch import nn
from torchvision.models import get_model

from src.models.components.module_utils import freez_model_layers


class ResNet(nn.Module):
    supported_models = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    freez_layers_name = [
        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]

    def __init__(
        self,
        name: str = "resnet18",
        output_size: int = 6,
        pretrain: bool = True,
        freez_layers: int = 0,
        freez_batch_norm: bool = True,
    ):
        super().__init__()

        assert name in self.supported_models
        self.model = get_model(name=name, weights="DEFAULT" if pretrain else None)

        if freez_layers == 0:
            # replace input 3 channels with 1 channel
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

        freez_model_layers(
            model=self.model,
            layers_name=self.freez_layers_name[:freez_layers],
            freez_batch_norm=freez_batch_norm,
        )

    def forward(self, x):
        dense_logits = self.model(x)
        return dense_logits, self.classifier(dense_logits)


if __name__ == "__main__":
    _ = ResNet()
