from __future__ import annotations
from typing import Sequence, Tuple, List, Type
import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import Literal, Optional


ActivationFactory = Type[nn.Module]  # e.g., nn.Sigmoid or nn.ReLU


InitKind = Literal["torch_default", "xavier_uniform", "kaiming_uniform"]
ScaleKind = Literal["none", "width", "muP_placeholder"]

@dataclass(frozen=True)
class MLPConfig:
    in_dim: int = 784
    width: int = 128
    depth: int = 3
    out_dim: int = 10
    activation: ActivationFactory = nn.Sigmoid
    bias: bool = True
    init: InitKind = "torch_default"
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

def _init_linear(m: nn.Module, kind: InitKind) -> None:
    if not isinstance(m, nn.Linear):
        return
    if kind == "xavier_uniform":
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif kind == "kaiming_uniform":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    else:
        # torch defaults
        pass



class MLP(nn.Module):
    """
    Simple equal-width MLP (baseline).

    Parameters
    ----------
    in_dim : int
        Flattened input dimension (e.g., 28*28=784 for MNIST).
    width : int
        Hidden layer width (same for all hidden layers).
    depth : int
        Number of hidden layers (>= 1).
    out_dim : int
        Number of output classes.
    activation : ActivationFactory
        Activation class to use; defaults to nn.Sigmoid to match the original.
    bias : bool
        Whether to include bias terms in Linear layers.
    """

    def __init__(self, cfg: MLPConfig = MLPConfig()) -> None:
        super().__init__()
        if cfg.in_dim <= 0 or cfg.width <= 0 or cfg.depth < 1 or cfg.out_dim <= 0:
            raise ValueError("in_dim, width, out_dim must be >0 and depth >=1")
        layers: List[nn.Module] = [nn.Linear(cfg.in_dim, cfg.width, bias=cfg.bias), cfg.activation()]
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.width, cfg.width, bias=cfg.bias), cfg.activation()]
        layers += [nn.Linear(cfg.width, cfg.out_dim, bias=cfg.bias)]
        self.net = nn.Sequential(*layers)
        # Optional device/dtype placement
        if cfg.device or cfg.dtype:
            self.to(device=cfg.device, dtype=cfg.dtype)
        # Optional explicit init
        if cfg.init != "torch_default":
            self.apply(lambda m: _init_linear(m, cfg.init))

    def forward(self, x: Tensor) -> Tensor:
        """Return logits of shape (batch, out_dim)."""
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ScaledMLP(nn.Module):
    """
    Variable-width MLP with an explicit TODO hook before activation.

    This preserves the teaching value: students fill the TODO (e.g., μP/width scaling).
    Returns (logits, activations) where activations are the post-activation tensors
    (detached), and we intentionally drop the first one to match the original scaffold.
    """

    def __init__(self, ..., scale: ScaleKind = "none", init: InitKind = "torch_default"):
        super().__init__()
        ...
        self.scale = scale
        if init != "torch_default":
            self.apply(lambda m: _init_linear(m, init))

    def _maybe_scale(self, x: Tensor, layer: nn.Linear) -> Tensor:
        if self.scale == "width":
            return x / (layer.out_features ** 0.5)
        elif self.scale == "muP_placeholder":
            # Keep as a placeholder to avoid mis-teaching; document true μP separately.
            return x / (layer.out_features ** 0.5)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations: List[Tensor] = []
        x = x.reshape(x.size(0), -1)
        for layer in self.hidden:
            x = layer(x)
            x = self._maybe_scale(x, layer)  # hook now functional but off by default
            x = self.act(x)
            activations.append(x)
        logits = self.out(x)
        if activations:
            activations = activations[1:]
        return logits, [a.detach() for a in activations]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
