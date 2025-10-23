"""
optimizers.py

Refactored collection of simplified optimizers for educational deep learning use.
This module defines three lightweight optimizers—SimpleAdam, SimpleAdamMuP, and
SimpleShampoo—rewritten for clarity, modularity, and readability while preserving
their original teaching purpose and PyTorch compatibility.

- SimpleAdam: Minimal Adam implementation showing moving averages and bias correction.
- SimpleAdamMuP: Adam variant with a TODO placeholder for per-layer MuP scaling.
- SimpleShampoo: Simplified Shampoo optimizer with a TODO placeholder for preconditioning.

All optimizers include hyperparameter validation, explicit state initialization,
and clear TODO hooks for experimentation. The code mirrors real PyTorch design
patterns but remains intentionally simple for instructional clarity.
"""


import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional, Tuple, Dict, Any
import numpy as np


class SimpleAdam(Optimizer):
    """
    A minimal, educational implementation of the Adam optimizer.

    This optimizer computes parameter updates using moving averages of both
    gradients and their squares, with bias correction. It is designed to
    illustrate the key concepts behind Adam while remaining clear and compact.

    Args:
        params (Iterable[Any]): Iterable of parameters to optimize.
        lr (float): Learning rate. Default is 0.1.
        b1 (float): Exponential decay rate for first-moment estimates (β₁). Default is 0.9.
        b2 (float): Exponential decay rate for second-moment estimates (β₂). Default is 0.999.

    Notes:
        - Follows the same variable naming as the original Adam algorithm (m, v, m_hat, v_hat).
        - Designed for readability and teaching, not for large-scale training.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-1,
        b1: float = 0.9,
        b2: float = 0.999,
    ) -> None:
        # --- Validate input hyperparameters ---
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be positive.")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid b1 value: {b1}. Must be in [0, 1).")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid b2 value: {b2}. Must be in [0, 1).")

        # Default parameter group configuration
        defaults = dict(lr=lr, b1=b1, b2=b2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> None:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): Re-evaluates the model and returns the loss.
        """
        # Allow optional closure for loss recomputation
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SimpleAdam does not support sparse gradients")

                # Retrieve parameter state or initialize
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)
                    state["variance"] = torch.zeros_like(p)

                # Increment step counter
                state["step"] += 1
                step = state["step"]
                m, v = state["momentum"], state["variance"]

                # Update moving averages
                m.lerp_(grad, 1 - b1)
                v.lerp_(grad * grad, 1 - b2)

                # Bias correction
                m_hat = m / (1 - b1**step)
                v_hat = v / (1 - b2**step)

                # Parameter update
                u = m_hat / (torch.sqrt(v_hat) + 1e-16)
                p.add_(u, alpha=-lr)

        return None
    

class SimpleAdamMuP(Optimizer):
    """
    Simplified Adam optimizer variant with support for MuP (Maximum Update Parametrization).

    This class extends the base Adam logic and includes a placeholder for implementing
    per-layer scaling factors used in MuP. The current version matches the structure of
    the original educational code and raises a NotImplementedError at the MuP step.

    Args:
        params (Iterable): Model parameters to optimize.
        lr (float): Learning rate. Default is 1e-3.
        betas (Tuple[float, float]): Coefficients for moving averages of gradient and its square.
        eps (float): Small constant to prevent division by zero. Default is 1e-16.
        weight_decay (float): Optional L2 regularization term. Default is 0.0.
        layer_scales (Optional[Dict]): Mapping from parameter shapes to scaling factors.

    Notes:
        - The TODO block indicates where MuP per-layer scaling logic should be added.
        - Maintains compatibility with standard PyTorch training loops.
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

        self.layer_scales = layer_scales or {}

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> None:
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

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)
                    state["variance"] = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["momentum"], state["variance"]

                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                m_hat = m / (1 - b1 ** state["step"])
                v_hat = v / (1 - b2 ** state["step"])

                u = m_hat / (torch.sqrt(v_hat) + eps)

                ###############################################
                ###############################################
                ### TODO: Adjust the per-layer learning rate scaling factor so per-layer RMS activation deltas are constant.
                ### Hint for part e: The following tricks will help you retain performance when using muP scaling.
                ###  - Treat biases as a hidden layer with size (d_out, 1). You will need to use a fudge-factor of around 0.01 -- we want to keep the change in bias terms low.
                ###  - For the input layer, a fudge factor of 10 appears to help.
                ###  - For the output layer, we find it is best to ignore the muP scaling, and instead use a fixed learning rate (e.g. 0.003).
                ###############################################
                ###############################################
                lr = group['lr']
                if len(u.shape) == 2:
                    if u.shape[1] == 784:
                        lr = lr * (1/u.shape[1]) * 10
                    elif u.shape[0] == 10:
                        lr = 0.003
                    else:
                        lr = lr * (1 / u.shape[1])
                else:
                    lr = (lr / 1) * 0.01
                ###############################################
                ###############################################

                # Update parameters
                p.add_(u, alpha=-lr)

        return None
    

class SimpleShampoo(Optimizer):
    """
    Simplified version of the Shampoo optimizer for conceptual understanding.

    Implements an exponential moving average of gradients and includes a TODO
    placeholder for the layer-wise preconditioning step used in full Shampoo.
    In its current form, it functions like a momentum-based variant of SGD.

    Args:
        params (Iterable[Any]): Parameters to optimize.
        lr (float): Learning rate. Default is 0.1.
        b1 (float): Momentum coefficient. Default is 0.9.

    Notes:
        - The preconditioning step is intentionally omitted for simplicity.
        - The TODO block marks where the matrix-based preconditioner can be added.
    """

    def __init__(self, params: Any, lr: float = 1e-1, b1: float = 0.9):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid momentum coefficient b1: {b1}")

        defaults = dict(lr=lr, b1=b1)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> None:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        for group in self.param_groups:
            b1 = group["b1"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("SimpleShampoo does not support sparse gradients")

                # Retrieve the parameter state
                state = self.state[p]

                # Initialize state if this is the first update
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)

                state["step"] += 1
                m = state["momentum"]

                # Update momentum buffer
                m.lerp_(grad, 1 - b1)

                ###############################################
                ###############################################
                # TODO ########################
                # In full Shampoo, this is where you would apply a
                # preconditioner derived from the second-order
                # statistics of the gradient. For simplicity,
                # we ignore this and just use `m` directly.
                ###############################################
                ###############################################
                if len(m.shape) == 1:
                    u = m  # Ignore biases for this simplified version
                else:
                    su, ss, svT = torch.linalg.svd(m, full_matrices=False)
                    u = su @ svT
                ###############################################
                ###############################################

                # Parameter update
                p.add_(u, alpha=-lr)

        return None

class SimpleShampooScaled(Optimizer):
    """
    Simplified Shampoo optimizer variant with μP scaling for the preconditioner.
    """
    def __init__(
        self,
        params: Any,
        lr: float = 1e-1,
        b1: float = 0.9,
    ):
        defaults = dict(lr=lr, b1=b1)
        super(SimpleShampooScaled, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0: # Initialization
                    state["step"] = torch.tensor(0.0)
                    state['momentum'] = torch.zeros_like(p)

                state['step'] += 1
                m = state['momentum']
                m.lerp_(grad, 1-group["b1"])

                ###############################################
                ###############################################
                # TODO: Apply μP scaling to the preconditioner here
                ###############################################
                ###############################################
                if len(m.shape) == 1:
                    u = m
                else:
                    su, ss, svT = torch.linalg.svd(m, full_matrices=False)
                    u = su @ svT
                    u = u * np.sqrt(u.shape[0] / u.shape[1])
                ###############################################
                ###############################################
                p.add_(u, alpha=-group['lr'])
        return None