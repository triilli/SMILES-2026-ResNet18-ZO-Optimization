from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        eps: float = 1e-3,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model

        self.lr = 1e-2
        self.eps = 1e-2

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        self.momentum: dict[str, torch.Tensor] = {}
        self.gamma = 0.9

        self.layer_names: list[str] = [
            "fc.weight", 
            "fc.bias"
        ]

    def _active_params(self) -> dict[str, nn.Parameter]:
        
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:

        if self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        else:  # uniform
            u = torch.rand_like(param) * 2.0 - 1.0

        norm = u.norm()
        if norm > 0:
            u = u / norm
        return u

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        
        grads: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            directions = {name: self._sample_direction(p) for name, p in params.items()}
            # f(x + eps)
            for name, p in params.items():
                p.data.add_(self.eps * directions[name])
            f_plus = loss_fn()

            # f(x - eps)
            for name, p in params.items():
                p.data.sub_(2.0 * self.eps * directions[name])
            f_minus = loss_fn()

            for name, p in params.items():
                p.data.add_(self.eps * directions[name])

            diff = (f_plus - f_minus) / (2.0 * self.eps)
            
            for name in params:
                grads[name] = diff * directions[name]

        return grads

        
    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        total_norm = torch.sqrt(sum(g.norm()**2 for g in grads.values())) + 1e-8

        with torch.no_grad():
            for name, p in params.items():
                if name not in self.momentum:
                    self.momentum[name] = torch.zeros_like(p.data)

                g_normed = grads[name] / total_norm

                self.momentum[name] = self.gamma * self.momentum[name] + g_normed

                p.data.sub_(self.lr * torch.sign(self.momentum[name]))


    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)
