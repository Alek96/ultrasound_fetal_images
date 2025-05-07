import torch
from torch import nn
from torchvision.models import get_model

from src.models.components.module_utils import freez_model_layers


class EfficientNet(nn.Module):
    supported_efficientnet_models = ["efficientnet_b0"]
    supported_efficientnet_v2_models = ["efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"]
    supported_models = supported_efficientnet_models + supported_efficientnet_v2_models

    freez_layers_name = [
        "features.0",
        "features.1",
        "features.2",
        "features.3",
        "features.4",
        "features.5",
        "features.6",
        "features.7",
        "features.8",
    ]

    def __init__(
        self,
        name: str = "efficientnet_b0",
        output_size: int = 6,
        pretrain: bool = True,
        freez_layers: int = 0,
        freez_batch_norm: bool = True,
        dropout: float | None = None,
    ):
        super().__init__()

        assert name in self.supported_models

        get_model_param = {"name": name, "weights": "DEFAULT" if pretrain else None}
        if dropout is not None:
            get_model_param["dropout"] = dropout
        self.model = get_model(**get_model_param)

        self.freez_layers = freez_layers > 0

        if not self.freez_layers:
            # replace input 3 channels with 1 channel
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
        self.classifier = nn.Linear(
            in_features=self.model.classifier[-1].in_features,
            out_features=output_size,
        )
        self.model.classifier = self.model.classifier[:-1]

        freez_model_layers(
            model=self.model,
            layers_name=self.freez_layers_name[:freez_layers],
            freez_batch_norm=freez_batch_norm,
        )

    def forward(self, x):
        if self.freez_layers:
            x = x.expand(-1, 3, -1, -1)

        dense_logits = self.model(x)
        return dense_logits, self.classifier(dense_logits)


if __name__ == "__main__":
    _ = EfficientNet()
