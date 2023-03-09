from torch import nn

from src.models.components.densenet import DenseNet
from src.models.components.efficientnet import EfficientNet
from src.models.components.mobilenet import MobileNet
from src.models.components.resnet import ResNet
from src.models.components.resnext import ResNeXt


def get_model(
    name: str = "densenet121",
    output_size: int = 6,
    pretrain: bool = True,
    **kwargs,
) -> nn.Module:
    if name in DenseNet.supported_models:
        return DenseNet(name=name, output_size=output_size, pretrain=pretrain)
    if name in MobileNet.supported_models:
        return MobileNet(name=name, output_size=output_size, pretrain=pretrain)
    if name in ResNet.supported_models:
        return ResNet(name=name, output_size=output_size, pretrain=pretrain)
    if name in ResNeXt.supported_models:
        return ResNeXt(name=name, output_size=output_size, pretrain=pretrain)
    if name in EfficientNet.supported_models:
        return EfficientNet(name=name, output_size=output_size, pretrain=pretrain, **kwargs)

    raise KeyError(f"Model with name {name} is not supported")
