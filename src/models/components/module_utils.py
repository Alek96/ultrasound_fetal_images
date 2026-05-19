from torch import nn

batch_norm_classes = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
]


def _freeze_model_layer(module: nn.Module, freeze_batch_norm: bool = True):
    if len(list(module.named_children())) == 0:
        if freeze_batch_norm or module.__class__.__name__ not in batch_norm_classes:
            module.requires_grad_(requires_grad=False)
    else:
        for layer in module.children():
            _freeze_model_layer(layer, freeze_batch_norm)


def freeze_model_layers(model: nn.Module, layers_name: list[str], freeze_batch_norm: bool = True):
    for layer_name in layers_name:
        module = model
        for layer in layer_name.split("."):
            module = getattr(module, layer)

        _freeze_model_layer(module, freeze_batch_norm=freeze_batch_norm)
