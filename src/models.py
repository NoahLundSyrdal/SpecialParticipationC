from __future__ import annotations
from typing import List, Tuple, Literal
import torch
import torch.nn as nn
from torch import Tensor


InitKind = Literal["default", "xavier_uniform", "kaiming_uniform"]


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden sizes.

    Args:
        input_size: Flattened input dimension (e.g., 28*28 for MNIST).
        hidden_sizes: Hidden layer widths in order.
        num_classes: Number of output classes.
        init: Optional initialization scheme ("default", "xavier_uniform",
              or "kaiming_uniform"). Default preserves PyTorch behavior.

    Behavior preserved:
      - Public interface identical to the original.
      - forward(x) returns (logits, activations) where the first hidden
        activation is dropped (activations = activations[1:]).
      - A μP/experiment hook remains available between Linear and Sigmoid.

    Software engineering improvements:
      - Type hints and Google-style docstrings (PEP 257)
      - Input validation and fail-fast guardrails
      - Centralized layer creation and initialization
      - Custom __repr__ for quick configuration inspection

    ML engineering improvements:
      - Optional deterministic initialization (Xavier or Kaiming)
      - Support for reproducibility studies (set global torch seeds externally)

    Note: Default initialization uses PyTorch’s internal reset_parameters()
    to ensure parity with the original notebook behavior.
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

        # Guardrails — fail fast on invalid input
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

        # Maintain PyTorch default initialization unless explicitly overridden
        if self.init_kind != "default":
            self.reset_parameters()

    def _build_linears(self, sizes: List[int], *, bias: bool) -> nn.ModuleList:
        """Construct a sequential list of Linear layers."""
        return nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(len(sizes) - 1)]
        )

    def reset_parameters(self) -> None:
        """
        Optional deterministic initialization.

        Default ("default"): use PyTorch’s built-in reset_parameters().
        For custom schemes, supports:
          - "xavier_uniform"
          - "kaiming_uniform" (note: designed for ReLU activations)
        """
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

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        if x.dim() < 2:
            raise ValueError(f"Expected input with at least 2 dims, got shape {tuple(x.shape)}")

        activations: List[Tensor] = []
        # Flatten to [batch_size, input_size] — matches original notebook
        x = x.flatten(1)

        # Hidden layers
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.sigmoid(x)
            activations.append(x)

        # Final linear head (logits)
        x = self.layers[-1](x)

        # Match original behavior — drop first activation and detach
        activations = activations[1:]
        return x, [a.detach() for a in activations]

    def __repr__(self) -> str:
        return (
            f"MLP(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, "
            f"num_classes={self.num_classes}, init='{self.init_kind}')"
        )


class ScaledMLP(nn.Module):
    """
    Bias-free MLP variant for μP scaling experiments.

    Args:
        input_size: Flattened input dimension.
        hidden_sizes: Hidden layer widths in order.
        num_classes: Number of output classes.
        init: Optional initialization scheme ("default", "xavier_uniform",
              or "kaiming_uniform"). Default preserves PyTorch behavior.

    Notes:
      - Keeps the μP/experiment hook and activation trimming behavior.
      - All Linear layers are constructed with bias=False.
      - Initialization behavior mirrors MLP for reproducibility.
    """
    def __init__(self, input_size=784, hidden_sizes = [8, 16, 32, 64, 128], num_classes=10):
        super(ScaledMLP, self).__init__()
        all_hidden_sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList()
        for i in range(len(all_hidden_sizes)-1):
            self.layers.append(nn.Linear(all_hidden_sizes[i], all_hidden_sizes[i+1], bias=False))
        self.sigmoid = nn.Sigmoid()

        ## Rescale weight initializations to account for pre-activation scaling
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.mul_(layer.weight.shape[1])
        ##

    def forward(self, x):
        activations = []
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 28*28)
        for layer in self.layers[:-1]:
            x = layer(x)
            ###############################################
            ###############################################
            # TODO: Apply μP scaling to pre-activations here
            ###############################################
            ###############################################
            raise NotImplementedError("MuP scaling not yet implemented in ScaledMLP.")
            ###############################################
            ###############################################
            
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