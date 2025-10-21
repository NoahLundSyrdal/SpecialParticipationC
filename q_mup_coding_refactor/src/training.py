from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Sequence
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import Tensor


def set_seed(seed: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(t: Tensor, device: Optional[torch.device]) -> Tensor:
    return t if device is None else t.to(device)


def rms(x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=keepdim) + 1e-12)


def train_one_step(
    mlp: Callable,                    # factory/class, must accept hidden_sizes=
    hiddens: Optional[Sequence[int]] = None,
    optimizer: Callable = None,       # e.g., torch.optim.Adam or SimpleAdam
    label: str = "Adam",
    lr: float = 0.01,
    *,
    train_images: Tensor,
    train_labels: Tensor,
    batch_idx: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Builds a model, runs two steps on the same batch, and plots RMS of activation deltas per layer.
    The model's forward must return (logits, activations).
    """
    if seed is not None:
        set_seed(seed)

    if hiddens is None:
        hiddens = [8, 16, 32, 64, 128]

    if optimizer is None:
        raise ValueError("optimizer class must be provided")

    device = get_device() if device is None else device

    model = mlp(hidden_sizes=list(hiddens)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)

    # pick a batch if none given
    if batch_idx is None:
        batch_idx = np.random.randint(0, len(train_images), size=64)

    prev_activations = None
    activation_deltas_rms: List[float] = []

    for i in range(2):
        images_batch = to_device(train_images[batch_idx], device)
        labels_batch = to_device(train_labels[batch_idx], device)

        optim_inst.zero_grad(set_to_none=True)
        logits, activations = model(images_batch)  # (logits, activations)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optim_inst.step()

        if i > 0 and prev_activations is not None:
            deltas = [a - pa for a, pa in zip(activations, prev_activations)]
            activation_deltas_rms = [rms(d, dim=-1).mean().item() for d in deltas]

        prev_activations = [a.detach() for a in activations]

    # plot
    if activation_deltas_rms:
        xs = np.arange(len(activation_deltas_rms))
        plt.figure(figsize=(8, 4))
        plt.title(f"RMS of activation deltas per layer ({label})")
        plt.xlabel("Hidden layer index")
        plt.bar(xs, activation_deltas_rms)
        # labels correspond to returned activations (first hidden dropped)
        tick_labels = list(hiddens[1:]) if len(hiddens) > 1 else list(hiddens)
        plt.xticks(xs, tick_labels)
        plt.show()

    # return last-step loss/acc for logging
    with torch.no_grad():
        acc = (logits.argmax(1) == labels_batch).float().mean().item()
    return float(loss.item()), float(acc)


def train_one_step_matrices(
    mlp: Optional[Callable] = None,
    hiddens: Optional[Sequence[int]] = None,
    optimizer: Optional[Callable] = None,
    label: str = "Adam",
    lr: float = 0.01,
    *,
    train_images: Optional[Tensor] = None,
    train_labels: Optional[Tensor] = None,
    batch_idx: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        set_seed(seed)
    if hiddens is None:
        hiddens = [8, 16, 64, 64, 64, 256, 256, 1024]
    if mlp is None:
        from models import MLP as _MLP
        mlp = _MLP
    if optimizer is None:
        raise ValueError("optimizer class must be provided")
    if train_images is None or train_labels is None:
        raise ValueError("train_images and train_labels must be provided")

    device = get_device() if device is None else device

    model = mlp(hidden_sizes=list(hiddens)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)

    if batch_idx is None:
        batch_idx = np.random.randint(0, len(train_images), size=64)

    old_params = [p.detach().clone() for p in model.parameters()]

    images_batch = to_device(train_images[batch_idx], device)
    labels_batch = to_device(train_labels[batch_idx], device)

    optim_inst.zero_grad(set_to_none=True)
    outputs, activations = model(images_batch)  # (logits, activations)
    loss = criterion(outputs, labels_batch)
    loss.backward()
    optim_inst.step()

    new_params = [p.detach().clone() for p in model.parameters()]
    delta_params = [new - old for new, old in zip(new_params, old_params)]

    frob_norms: List[float] = []
    spectral_norms: List[float] = []
    induced_norms: List[float] = []
    p_shapes: List[Tuple[int, ...]] = []

    for p in delta_params:
        if p.ndim == 2:
            cpu_p = p.detach().float().cpu()
            p_shapes.append(tuple(cpu_p.shape))
            frob_norms.append(float(torch.linalg.norm(cpu_p)))
            try:
                # faster + stable for tall/wide matrices
                _, s, _ = torch.linalg.svd(cpu_p, full_matrices=False)
                spectral_norms.append(float(s[0]))
            except Exception:
                spectral_norms.append(0.0)
            # Induced RMS-RMS placeholder: mean RMS of rows
            induced_norms.append(float(torch.mean(rms(cpu_p, dim=1))))

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].set_title(f"Frobenius norm of update per layer ({label})")
    axs[0].set_xlabel("Layer shape")
    axs[0].bar(np.arange(len(frob_norms)), frob_norms)
    if p_shapes:
        axs[0].set_xticks(np.arange(len(frob_norms)))
        axs[0].set_xticklabels([str(s) for s in p_shapes], rotation=45, ha="right")

    axs[1].set_title(f"Spectral norm of update per layer ({label})")
    axs[1].set_xlabel("Layer shape")
    axs[1].bar(np.arange(len(spectral_norms)), spectral_norms)
    if p_shapes:
        axs[1].set_xticks(np.arange(len(spectral_norms)))
        axs[1].set_xticklabels([str(s) for s in p_shapes], rotation=45, ha="right")

    axs[2].set_title(f"Induced norm of update per layer ({label})")
    axs[2].set_xlabel("Layer shape")
    axs[2].bar(np.arange(len(induced_norms)), induced_norms)
    if p_shapes:
        axs[2].set_xticks(np.arange(len(induced_norms)))
        axs[2].set_xticklabels([str(s) for s in p_shapes], rotation=45, ha="right")

    plt.show()


def train_with_lr(
    hiddens: Optional[Sequence[int]] = None,
    optimizer: Optional[Callable] = None,
    lr: float = 0.01,
    *,
    train_images: Optional[Tensor] = None,
    train_labels: Optional[Tensor] = None,
    valid_images: Optional[Tensor] = None,
    valid_labels: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    steps: int = 100,
    seed: Optional[int] = None,
) -> float:
    """
    Train for a number of steps and return mean of last validation losses (same
    behavior used in the notebook sweep).
    """
    if seed is not None:
        set_seed(seed)

    if hiddens is None:
        hiddens = [64, 64, 64]

    if optimizer is None:
        raise ValueError("optimizer class must be provided")
    if train_images is None or train_labels is None:
        raise ValueError("train_images and train_labels must be provided")
    if valid_images is None or valid_labels is None:
        raise ValueError("valid_images and valid_labels must be provided")

    device = get_device() if device is None else device

    from models import MLP
    model = MLP(hidden_sizes=list(hiddens)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)
    losses: List[float] = []

    for _ in range(steps):
        batch_idx = np.random.randint(0, len(train_images), size=64)

        images_batch = to_device(train_images[batch_idx], device)
        labels_batch = to_device(train_labels[batch_idx], device)

        optim_inst.zero_grad(set_to_none=True)
        outputs, _ = model(images_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optim_inst.step()

        with torch.inference_mode():
            outputs_valid, _ = model(to_device(valid_images, device))
            valid_losses = criterion(outputs_valid, to_device(valid_labels, device))
            losses.append(float(valid_losses.item()))

    return float(np.mean(np.array(losses)[-5:]))


def train_one_epoch(*args, **kwargs):
    # thin wrapper placeholder
    return train_with_lr(*args, **kwargs)


def evaluate(*args, **kwargs):
    # alias for convenience
    return train_with_lr(*args, **kwargs)


def fit(*args, **kwargs):
    return train_with_lr(*args, **kwargs)


def legacy_train_one_step(*args, **kwargs):
    return train_one_step(*args, **kwargs)
