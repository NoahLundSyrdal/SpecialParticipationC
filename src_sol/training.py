from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Sequence
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import Tensor
from .models import MLP
from .optimizers import SimpleAdam


def set_seed(seed: int = 4) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU (unified device handling)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(t: Tensor, device: Optional[torch.device]) -> Tensor:
    """Move tensor to device if provided (no-op if device is None)."""
    return t if device is None else t.to(device)


def rms(x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    """Numerically-stable RMS with small epsilon to avoid sqrt(0)."""
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=keepdim) + 1e-12)


def train_one_step(
    mlp: Callable = MLP,                    # factory/class; must accept hidden_sizes=
    hiddens: Optional[Sequence[int]] = [8, 16, 64, 64, 64, 256, 256, 1024],
    optimizer: Callable = SimpleAdam,       # SimpleAdam/SimpleAdamMuP/etc. 
    label: str = "Adam",
    lr: float = 0.01,
    *,
    train_images: Tensor,
    train_labels: Tensor,
    batch_idx: Optional[np.ndarray] = None,
    device: Optional[torch.device] = get_device(),
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
        Train for one step and report loss, accuracy, and activation deltas.
    """
    if seed is not None:
        set_seed(seed)

    model = mlp(hidden_sizes=list(hiddens)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)

    if batch_idx is None:
        if len(train_images) == 0:
            raise ValueError("train_images must be non-empty")
        batch_idx = np.random.randint(0, len(train_images), size=64)

    prev_activations: Optional[List[Tensor]] = None
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

    if activation_deltas_rms:
        xs = np.arange(len(activation_deltas_rms))
        plt.figure(figsize=(8, 4))
        plt.title(f"RMS of activation deltas per layer ({label})")
        plt.xlabel("Hidden layer index (returned)")
        plt.bar(xs, activation_deltas_rms)

        tick_labels = list(hiddens[1:]) if len(hiddens) > 1 else list(hiddens)
        plt.xticks(xs, tick_labels)
        plt.show()

    with torch.no_grad():
        acc = (logits.argmax(1) == labels_batch).float().mean().item()
    return float(loss.item()), float(acc)


def train_one_step_matrices(
    mlp: Optional[Callable] = MLP,
    hiddens: Optional[Sequence[int]] = [8, 16, 64, 64, 64, 256, 256, 1024],
    optimizer: Optional[Callable] = SimpleAdam,
    label: str = "Adam",
    lr: float = 0.01,
    *,
    train_images: Optional[Tensor] = None,
    train_labels: Optional[Tensor] = None,
    batch_idx: Optional[np.ndarray] = None,
    device: Optional[torch.device] = get_device(),
    seed: Optional[int] = None,
) -> None:
    """
        Train for one step and report parameter update norms.
    """
    if seed is not None:
        set_seed(seed)

    model = mlp(hidden_sizes=list(hiddens)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)

    if batch_idx is None:
        if len(train_images) == 0:
            raise ValueError("train_images must be non-empty")
        batch_idx = np.random.randint(0, len(train_images), size=64)

    # Snapshot parameters before the step
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
            cpu_p = p.detach().float().cpu()  # Compute once and reuse
            p_shapes.append(tuple(cpu_p.shape))
            ###############################################
            ###############################################
            # TODO: Implement the norms here
            # Frobenius norm
            # Spectral norm
            # RMS-RMS Induced norm
            ###############################################
            # Frobenius norm
            frob_norm = torch.linalg.norm(cpu_p, ord='fro').item()
            frob_norms.append(frob_norm)

            # Spectral norm (only largest singular value)
            spectral_norm = torch.linalg.svdvals(cpu_p)[0].item()
            spectral_norms.append(spectral_norm)

            # RMS-RMS Induced norm
            induced_norm = spectral_norm * np.sqrt(cpu_p.shape[1] / cpu_p.shape[0])
            induced_norms.append(induced_norm)
            ###############################################
            ###############################################

    # Visualization of update norms
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
    mlp: Callable = MLP,
    hiddens: Optional[Sequence[int]] = [8, 16, 64, 64, 64, 256, 256, 1024],
    optimizer: Optional[Callable] = SimpleAdam,
    lr: float = 0.01,
    *,
    train_images: Optional[Tensor] = None,
    train_labels: Optional[Tensor] = None,
    valid_images: Optional[Tensor] = None,
    valid_labels: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    steps: int = 100,
    seed: Optional[int] = None,
) -> None:
    """
    Train a model for "steps" with a specific learning rate.
    """
    if seed is not None:
        set_seed(seed)

    torch.manual_seed(4)
    np.random.seed(4)
    model = mlp(hidden_sizes=hiddens).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    losses = []

    for i in range(steps):
        batch_idx = np.random.randint(0, len(train_images), size=64)

        images_batch = train_images[batch_idx]
        labels_batch = train_labels[batch_idx]
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(images_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs_valid, _ = model(valid_images)
            valid_losses = criterion(outputs_valid, valid_labels)
            losses.append(valid_losses.item())

    return np.mean(np.array(losses)[-5:])


def train_one_epoch(*args, **kwargs):
    """Thin wrapper retained for API parity with the original notebook."""
    return train_with_lr(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Alias for convenience; mirrors original usage in the notebook."""
    return train_with_lr(*args, **kwargs)


def fit(*args, **kwargs):
    """Alias retained for backward compatibility with prior cells."""
    return train_with_lr(*args, **kwargs)


def legacy_train_one_step(*args, **kwargs):
    """Legacy name preserved to ease migration of older notebook cells."""
    return train_one_step(*args, **kwargs)
