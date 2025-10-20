"""
optimizer.py

Refactored SimpleAdamMuP optimizer with preserved original variable naming
(`m`, `v`, `m_hat`, `v_hat`, `u`, `b1`, `b2`), improved structure, and clear
documentation for educational readability.

Implements a simplified Adam optimizer with optional MuP (Maximum Update
Parametrization) scaling logic placeholder. Compatible with the PyTorch
Optimizer API.

Example:
    >>> import torch
    >>> from optimizer import SimpleAdamMuP
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = SimpleAdamMuP(model.parameters(), lr=1e-3)
    >>> loss_fn = torch.nn.MSELoss()
    >>> x, y = torch.randn(8, 10), torch.randn(8, 2)
    >>> for _ in range(100):
    ...     optimizer.zero_grad()
    ...     y_pred = model(x)
    ...     loss = loss_fn(y_pred, y)
    ...     loss.backward()
    ...     optimizer.step()
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional, Tuple, Dict


class SimpleAdamMuP(Optimizer):
    """
    Simplified Adam optimizer supporting optional per-layer MuP scaling.

    Args:
        params: Iterable of model parameters to optimize.
        lr: Learning rate (default 1e-3).
        betas: Tuple of (b1, b2) coefficients for gradient and squared gradient moving averages.
        eps: Small constant to avoid division by zero (default 1e-8).
        weight_decay: L2 penalty term (default 0.0).
        layer_scales: Optional mapping from parameter shapes to per-layer scaling factors.

    Notes:
        - This version maintains the same logic as the original homework Adam variant.
        - The MuP scaling TODO is preserved for student implementation.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        layer_scales: Optional[Dict] = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Optional MuP scaling factors keyed by parameter shapes
        self.layer_scales = layer_scales or {}

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("SimpleAdamMuP does not support sparse gradients")

                # Retrieve the optimizer state for this parameter
                state = self.state[p]

                # Initialize state if first update
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)
                    state["variance"] = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["momentum"], state["variance"]

                # Apply weight decay if requested
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                # Update exponential moving averages
                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                # Compute bias-corrected estimates
                m_hat = m / (1 - b1 ** state["step"])
                v_hat = v / (1 - b2 ** state["step"])

                # Compute parameter update direction
                u = m_hat / (torch.sqrt(v_hat) + eps)

                ############################
                ### Todo: Adjust the per-layer learning rate scaling factor so per-layer RMS activation deltas are constant.
                ### Hint for part e: The following tricks will help you retain performance when using muP scaling.
                ###  - Treat biases as a hidden layer with size (d_out, 1). You will need to use a fudge-factor of around 0.01 -- we want to keep the change in bias terms low.
                ###  - For the input layer, a fudge factor of 10 appears to help.
                ###  - For the output layer, we find it is best to ignore the muP scaling, and instead use a fixed learning rate (e.g. 0.003).
                ############################
                raise NotImplementedError
                ############################
                ############################

                # Update parameters
                p.add_(u, alpha=-lr)

        return None
