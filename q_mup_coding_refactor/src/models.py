from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import math
import torch
from torch import nn


ActivationFactory = Callable[[], nn.Module]

@dataclass(frozen=True)
class MLPConfig:
    """Configuration for an MLP-style classifier/regressor."""
    in_dim: int
    width: int
    depth: int                      # number of hidden layers (>=1 recommended)
    out_dim: int
    activation: ActivationFactory = nn.ReLU          # e.g., nn.ReLU, nn.GELU
    dropout_p: float = 0.0                           # 0.0 disables dropout
    use_bias: bool = True
    norm: Optional[str] = None                       # {"layernorm", "batchnorm", None}
    # Initialization
    init: str = "xavier_uniform"                     # {"xavier_uniform","kaiming_uniform","none"}
    # Device/dtype (optional)
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


def _make_norm(norm: Optional[str], hidden_dim: int) -> Optional[nn.Module]:
    if norm is None:
        return None
    if norm.lower() == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm.lower() == "batchnorm":
        return nn.BatchNorm1d(hidden_dim)
    raise ValueError(f"Unknown norm: {norm}")


def _init_linear(layer: nn.Linear, scheme: str):
    if scheme == "none":
        return
    if scheme == "xavier_uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif scheme == "kaiming_uniform":
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    else:
        raise ValueError(f"Unknown init scheme: {scheme}")
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)


class MLP(nn.Module):
    """Multilayer perceptron with configurable activation, norm, dropout, and init.

    Parameters
    ----------
    cfg : MLPConfig
        Configuration object controlling architecture and initialization.
    """

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        if cfg.in_dim <= 0 or cfg.out_dim <= 0 or cfg.width <= 0:
            raise ValueError("in_dim, out_dim, and width must be > 0")
        if cfg.depth < 1:
            raise ValueError("depth must be >= 1")

        act = cfg.activation  # factory

        layers: list[nn.Module] = []
        # First layer
        layers += [nn.Linear(cfg.in_dim, cfg.width, bias=cfg.use_bias)]
        if cfg.norm:
            norm = _make_norm(cfg.norm, cfg.width)
            if isinstance(norm, nn.BatchNorm1d):
                layers += [norm]  # BN goes before activation for MLPs commonly
            else:
                layers += [norm]
        layers += [act()]
        if cfg.dropout_p > 0:
            layers += [nn.Dropout(cfg.dropout_p)]

        # Hidden layers
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.width, cfg.width, bias=cfg.use_bias)]
            if cfg.norm:
                norm = _make_norm(cfg.norm, cfg.width)
                layers += [norm]
            layers += [act()]
            if cfg.dropout_p > 0:
                layers += [nn.Dropout(cfg.dropout_p)]

        # Output layer
        layers += [nn.Linear(cfg.width, cfg.out_dim, bias=cfg.use_bias)]

        # Build sequential and move to device/dtype if requested
        self.net = nn.Sequential(*layers)
        if cfg.device is not None or cfg.dtype is not None:
            self.net.to(device=cfg.device, dtype=cfg.dtype)

        # Apply initialization
        if cfg.init != "none":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    _init_linear(m, cfg.init)

        self.cfg = cfg  # keep around for repr/debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ScaledMLP(MLP):
    """μP-flavored MLP shell.

    Note
    ----
    This class exposes hooks for μP scaling but does **not** implement full μP rules.
    To stay faithful to μP, prefer width-aware init/optimizer scaling rather than
    a post-hoc uniform multiply of all parameters.
    """

    def __init__(
        self,
        cfg: MLPConfig,
        weight_scale: Optional[float] = None,
    ):
        super().__init__(cfg)
        if weight_scale is not None:
            # Safer: apply to *weights* only, not biases; do it right after init.
            with torch.no_grad():
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.mul_(weight_scale)
