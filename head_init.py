import torch
import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    torch.nn.init.xavier_uniform_(layer.weight)
    # layer.weight.data.mul_(0.5)
    torch.nn.init.zeros_(layer.bias)
