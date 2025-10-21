from __future__ import annotations
from typing import List, Tuple, Literal
import torch
import torch.nn as nn
from torch import Tensor


InitKind = Literal["default", "xavier_uniform", "kaiming_uniform"]


class MLP(nn.Module):
    """
    Equal-width MLP identical to the original public interface.

    Args:
        input_size: Flattened input dimension (28*28 for MNIST).
        hidden_sizes: Hidden layer widths in order.
        num_classes: Number of output classes.

    Behavior preserved:
      - Constructor signature and defaults are unchanged.
      - forward(x) returns (logits, activations) where the first hidden
        activation is dropped (activations = activations[1:]).
      - TODO hook remains between Linear and Sigmoid in each hidden layer.

    ML engineering upgrades:
      - Type hints & docstrings
      - Optional weight init via `init` (default "default" means PyTorch default)
      - Input validation/guardrails
      - Centralized layer building and reset_parameters()

    Note: We DO NOT change initialization by default to maintain parity.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [8, 16, 32, 64, 128],
        num_classes: int = 10,
        *,
        init: InitKind = "default",  # optional, non-breaking
    ) -> None:
        super().__init__()

        # Guardrails (fail fast on common mistakes)
        if input_size <= 0 or num_classes <= 0:
            raise ValueError("input_size and num_classes must be > 0")
        if not hidden_sizes or any(h <= 0 for h in hidden_sizes):
            raise ValueError("hidden_sizes must be non-empty and all > 0")

        self.input_size = int(input_size)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.num_classes = int(num_classes)
        self.init_kind: InitKind = init

        sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]
        self.layers = self._build_linears(sizes, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Do NOT force a new init by default; keep PyTorch defaults for parity.
        if self.init_kind != "default":
            self.reset_parameters()

    def _build_linears(self, sizes: List[int], *, bias: bool) -> nn.ModuleList:
        return nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(len(sizes) - 1)]
        )

    def reset_parameters(self) -> None:
        """
        Optional deterministic init you can call in experiments.
        Default ("default"): keep PyTorch’s internal reset for each layer.
        """
        if self.init_kind == "default":
            # Respect PyTorch's built-in reset_parameters() as-is.
            for layer in self.layers:
                layer.reset_parameters()
            return

        for i, layer in enumerate(self.layers):
            if not isinstance(layer, nn.Linear):
                continue
            if self.init_kind == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif self.init_kind == "kaiming_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
            else:
                raise ValueError(f"Unknown init kind: {self.init_kind}")

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        if x.dim() < 2:
            raise ValueError(f"Expected input with at least 2 dims, got shape {tuple(x.shape)}")

        activations: List[Tensor] = []
        # Flatten batch to [B, input_size] — matches original behavior
        x = x.flatten(1)

        # Hidden layers
        for layer in self.layers[:-1]:
            x = layer(x)


            x = self.sigmoid(x)
            activations.append(x)

        # Final linear head (logits)
        x = self.layers[-1](x)

        # Original notebook dropped the first activation
        activations = activations[1:]
        # Detach activations as in original
        return x, [a.detach() for a in activations]

    def __repr__(self) -> str:
        return (
            f"MLP(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, "
            f"num_classes={self.num_classes}, init='{self.init_kind}')"
        )


class ScaledMLP(nn.Module):
    """
    Bias-free variant with the same public interface and hook positions.

    Args:
        input_size: Flattened input dimension.
        hidden_sizes: Hidden layer widths in order.
        num_classes: Number of output classes.
        init: Optional weight init (default keeps PyTorch defaults).

    Notes:
      - Keeps the TODO hook and activation trimming behavior.
      - All Linear layers are constructed with bias=False.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [8, 16, 32, 64, 128],
        num_classes: int = 10,
        *,
        init: InitKind = "default",  # optional, non-breaking
    ) -> None:
        super().__init__()

        if input_size <= 0 or num_classes <= 0:
            raise ValueError("input_size and num_classes must be > 0")
        if not hidden_sizes or any(h <= 0 for h in hidden_sizes):
            raise ValueError("hidden_sizes must be non-empty and all > 0")

        self.input_size = int(input_size)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.num_classes = int(num_classes)
        self.init_kind: InitKind = init

        sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]
        self.layers = self._build_linears(sizes, bias=False)
        self.sigmoid = nn.Sigmoid()

        if self.init_kind != "default":
            self.reset_parameters()

    def _build_linears(self, sizes: List[int], *, bias: bool) -> nn.ModuleList:
        return nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(len(sizes) - 1)]
        )

    def reset_parameters(self) -> None:
        if self.init_kind == "default":
            for layer in self.layers:
                layer.reset_parameters()
            return

        for layer in self.layers:
            if not isinstance(layer, nn.Linear):
                continue
            if self.init_kind == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif self.init_kind == "kaiming_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
            else:
                raise ValueError(f"Unknown init kind: {self.init_kind}")
            # bias is always None in ScaledMLP

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        if x.dim() < 2:
            raise ValueError(f"Expected input with at least 2 dims, got shape {tuple(x.shape)}")

        activations: List[Tensor] = []
        x = x.flatten(1)

        for layer in self.layers[:-1]:
            x = layer(x)

            ## TODO
            pass
            ##

            x = self.sigmoid(x)
            activations.append(x)

        x = self.layers[-1](x)
        activations = activations[1:]
        return x, [a.detach() for a in activations]

    def __repr__(self) -> str:
        return (
            f"ScaledMLP(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, "
            f"num_classes={self.num_classes}, init='{self.init_kind}')"
        )

