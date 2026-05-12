import torch
import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    torch.nn.init.trunc_normal_(layer.weight)
    nn.init.trunc_normal_(layer.weight, std=0.01)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
