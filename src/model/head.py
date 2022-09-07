from typing import Callable, List, Optional

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        layers: int = 3,
        residual: float = 0,
        dropout: Optional[float] = None,
        activation: Callable = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        if residual > 0 and input_dim != out_dim:
            raise ValueError(
                f"Input and output dimensions must match for residual connections. "
                f"Got input_dim={input_dim} and output_dim={out_dim}"
            )

        self.residual = residual
        step: int = int(round((out_dim - input_dim) / layers))

        modules: List[nn.Module] = []
        for _ in range(layers - 1):
            modules.append(nn.Linear(input_dim, input_dim + step))
            modules.append(activation())
            if dropout:
                modules.append(nn.Dropout(dropout))
            input_dim = input_dim + step
        modules.append(nn.Linear(input_dim, out_dim))
        self.module = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)
        return (
            self.residual * x + (1 - self.residual) * output
            if self.residual > 0
            else output
        )
