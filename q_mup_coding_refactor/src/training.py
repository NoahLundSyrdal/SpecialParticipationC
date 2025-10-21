from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import Tensor


def set_seed(seed: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(t: Tensor, device: Optional[torch.device]) -> Tensor:
    if device is None:
        return t
    return t.to(device)


def rms(x: torch.Tensor, dim=None, keepdim=False):
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=keepdim) + 1e-12)

def train_one_step(
    mlp: Callable,
    hiddens: List[int],
    optimizer: Callable,
    label: str = "Adam",
    lr: float = 0.01,
    *,
    train_images: Tensor,
    train_labels: Tensor,
    batch_idx: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Builds a model, runs two steps on the same batch, and plots RMS of activation deltas per layer.
    The model's forward must return (logits, activations).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model; expects your factory to accept `hidden_sizes=...`
    model = mlp(hidden_sizes=hiddens).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)

    # pick a batch if none given
    if batch_idx is None:
        batch_idx = np.random.randint(0, len(train_images), size=64)

    prev_activations = None
    activation_deltas_rms = []

    for i in range(2):
        images_batch = train_images[batch_idx].to(device)
        labels_batch = train_labels[batch_idx].to(device)

        optim_inst.zero_grad(set_to_none=True)
        logits, activations = model(images_batch)      # <- must return (logits, activations)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optim_inst.step()

        if i > 0:
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
        plt.xticks(xs, hiddens[1:] if len(hiddens) > 1 else hiddens)
        plt.show()

    # return last-step loss/acc in case you want to log it
    with torch.no_grad():
        acc = (logits.argmax(1) == labels_batch).float().mean().item()
    return loss.item(), acc


def train_one_step_matrices(
    mlp: Callable = None,
    hiddens: List[int] = [8, 16, 64, 64, 64, 256, 256, 1024],
    optimizer: Callable = None,
    label: str = "Adam",
    lr: float = 0.01,
    *,
    train_images: Optional[Tensor] = None,
    train_labels: Optional[Tensor] = None,
    batch_idx: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
):
    if mlp is None:
        # resolve at runtime to avoid stale references
        from models import MLP as _MLP
        mlp = _MLP
    if optimizer is None:
        raise ValueError("optimizer class must be provided")
    if train_images is None or train_labels is None:
        raise ValueError("train_images and train_labels must be provided")

    device = get_device() if device is None else device

    model = mlp(hidden_sizes=hiddens).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)

    if batch_idx is None:
        batch_idx = np.random.randint(0, len(train_images), size=64)

    old_params = [p.detach().clone() for p in model.parameters()]

    images_batch = train_images[batch_idx].to(device)
    labels_batch = train_labels[batch_idx].to(device)

    optim_inst.zero_grad()
    outputs, activations = model(images_batch)  # forward must return (logits, activations)
    loss = criterion(outputs, labels_batch)
    loss.backward()
    optim_inst.step()

    new_params = [p.detach().clone() for p in model.parameters()]
    delta_params = [new_p - old_p for new_p, old_p in zip(new_params, old_params)]

    frob_norms: List[float] = []
    spectral_norms: List[float] = []
    induced_norms: List[float] = []
    p_shapes: List[Tuple[int, ...]] = []

    for p in delta_params:
        if len(p.shape) == 2:
            p_shapes.append(tuple(p.shape))
            frob_norms.append(float(torch.norm(p).cpu().numpy()))
            try:
                # approximate spectral norm via svd (may be slow)
                u, s, v = torch.linalg.svd(p)
                spectral_norms.append(float(s[0].cpu().numpy()))
            except Exception:
                spectral_norms.append(float(0.0))
            # Induced RMS-RMS placeholder: use mean RMS of rows
            induced_norms.append(float(torch.mean(rms(p, dim=1)).cpu().numpy()))

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].set_title(f"Frobenius norm of update per layer ({label})")
    axs[0].set_xlabel("Hidden Size of layer")
    axs[0].bar(np.arange(len(frob_norms)), frob_norms)
    if p_shapes:
        axs[0].set_xticks(np.arange(len(frob_norms)))
        axs[0].set_xticklabels([str(s) for s in p_shapes], rotation=45, ha="right")

    axs[1].set_title(f"Spectral norm of update per layer ({label})")
    axs[1].set_xlabel("Hidden Size of layer")
    axs[1].bar(np.arange(len(spectral_norms)), spectral_norms)
    if p_shapes:
        axs[1].set_xticks(np.arange(len(spectral_norms)))
        axs[1].set_xticklabels([str(s) for s in p_shapes], rotation=45, ha="right")

    axs[2].set_title(f"Induced norm of update per layer ({label})")
    axs[2].set_xlabel("Hidden Size of layer")
    axs[2].bar(np.arange(len(induced_norms)), induced_norms)
    if p_shapes:
        axs[2].set_xticks(np.arange(len(induced_norms)))
        axs[2].set_xticklabels([str(s) for s in p_shapes], rotation=45, ha="right")

    plt.show()


def train_with_lr(
    hiddens: List[int] = [64, 64, 64],
    optimizer: Callable = None,
    lr: float = 0.01,
    *,
    train_images: Optional[Tensor] = None,
    train_labels: Optional[Tensor] = None,
    valid_images: Optional[Tensor] = None,
    valid_labels: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    steps: int = 100,
):
    """
    Train for a number of steps and return mean of last validation losses (same
    behavior used in the notebook sweep).
    """
    if optimizer is None:
        raise ValueError("optimizer class must be provided")
    if train_images is None or train_labels is None:
        raise ValueError("train_images and train_labels must be provided")
    if valid_images is None or valid_labels is None:
        raise ValueError("valid_images and valid_labels must be provided")

    device = get_device() if device is None else device

    torch.manual_seed(4)
    np.random.seed(4)

    model = __import__("models").models.MLP(hidden_sizes=hiddens) if False else None
    # We import models lazily to avoid circular imports when used as a module.
    from q_mup_coding_refactor.src.models import MLP

    model = MLP(hidden_sizes=hiddens).to(device)
    criterion = nn.CrossEntropyLoss()
    optim_inst = optimizer(model.parameters(), lr=lr)
    losses: List[float] = []

    for i in range(steps):
        batch_idx = np.random.randint(0, len(train_images), size=64)

        images_batch = train_images[batch_idx]
        labels_batch = train_labels[batch_idx]
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        optim_inst.zero_grad()
        outputs, _ = model(images_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optim_inst.step()

        with torch.no_grad():
            outputs_valid, _ = model(valid_images.to(device))
            valid_losses = criterion(outputs_valid, valid_labels.to(device))
            losses.append(valid_losses.item())

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
