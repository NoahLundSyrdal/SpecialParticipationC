from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
    """Simple multilayer perceptron (baseline model)."""

    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class ScaledMLP(MLP):
    """μP-scaled MLP with custom initialization scaling rules."""

    def __init__(self, in_dim, width, depth, out_dim, scale: float = 1.0):
        super().__init__(in_dim, width, depth, out_dim)
        self.scale = scale
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(scale)