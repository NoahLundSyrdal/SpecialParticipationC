from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


class MLP(nn.Module):
    """
    Equal-width feedforward MLP.
    Mirrors the structure from the notebook version (bias=True).
    Returns:
      logits: Tensor of shape [B, num_classes]
      activations: list of Tensors for hidden-layer activations (starting from layer 2 per your original pattern)
    """
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [8, 16, 32, 64, 128],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        all_sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(all_sizes[i], all_sizes[i + 1], bias=True) for i in range(len(all_sizes) - 1)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations: List[Tensor] = []
        # Flatten (batch_size, 28*28) — matches original
        x = x.view(x.size(0), -1)

        # Hidden layers (all but last)
        for layer in self.layers[:-1]:
            x = layer(x)

            ## TODO
            # Keep this hook exactly where it was in your notebook.
            # Put any μP / scaling / normalization experiment here.
            pass
            ##

            x = self.sigmoid(x)
            activations.append(x)

        # Final linear head (no activation)
        x = self.layers[-1](x)

        # Your original notebook trimmed the first activation: activations = activations[1:]
        activations = activations[1:]
        return x, [a.detach() for a in activations]


class ScaledMLP(nn.Module):
    """
    Bias-free variant (as in your notebook) intended for scaling experiments.
    Keeps the same forward() flow and the TODO hook location.
    """
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [8, 16, 32, 64, 128],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        all_sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(all_sizes[i], all_sizes[i + 1], bias=False) for i in range(len(all_sizes) - 1)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations: List[Tensor] = []
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 28*28)

        for layer in self.layers[:-1]:
            x = layer(x)

            ## TODO
            # Keep this hook exactly where it was in your notebook.
            # Put any μP / scaling / normalization experiment here.
            pass
            ##

            x = self.sigmoid(x)
            activations.append(x)

        x = self.layers[-1](x)
        activations = activations[1:]
        return x, [a.detach() for a in activations]
