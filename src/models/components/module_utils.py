from torch import nn


def freez_model_layers(model: nn.Module, layers_name: list[str], freez_batch_norm: bool = True):
    for layer_name in layers_name:
        module = model
        for layer in layer_name.split("."):
            module = getattr(module, layer)

        freez_model_layer(module, freez_batch_norm=freez_batch_norm)


batch_norm_classes = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
]


def freez_model_layer(module: nn.Module, freez_batch_norm: bool = True):
    if len(list(module.named_children())) == 0:
        if freez_batch_norm or module.__class__.__name__ not in batch_norm_classes:
            module.requires_grad_(requires_grad=False)
    else:
        for layer in module.children():
            freez_model_layer(layer, freez_batch_norm)
