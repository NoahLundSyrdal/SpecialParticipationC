from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ---------- utilities ----------

def detach_logits(output: Any) -> Tensor:
    """
    Supports models that return either:
      - logits (Tensor), or
      - (logits, activations) where activations are ignored by the engine.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


def set_seed(seed: int, deterministic: bool = False) -> None:
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ---------- config & callbacks ----------

@dataclass
class TrainerConfig:
    epochs: int = 10
    lr: float = 1e-3                # used only if you construct a torch optimizer here
    log_every: int = 50
    grad_clip: Optional[float] = None
    seed: Optional[int] = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Callback:
    """Lightweight callback interface."""
    def on_epoch_start(self, trainer: "Trainer", epoch: int): ...
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, float]): ...
    def on_log(self, trainer: "Trainer", step: int, logs: Dict[str, float]): ...


class CSVLogger(Callback):
    def __init__(self, path: str = "training.csv"):
        self.path = path
        self._wrote_header = False

    def on_log(self, trainer: "Trainer", step: int, logs: Dict[str, float]):
        import csv, os
        row = {"step": step, **logs}
        write_header = not self._wrote_header and not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        self._wrote_header = True


# ---------- trainer ----------

class Trainer:
    """
    Minimal trainer: supports any nn.Module, any optimizer (torch or your SimpleAdam),
    CE/MSE/etc. Works when model returns logits or (logits, activations).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,                  # torch.optim.Optimizer or a SimpleAdam-like object
        loss_fn: nn.Module,
        cfg: TrainerConfig = TrainerConfig(),
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.cbs: List[Callback] = callbacks or []

        if cfg.seed is not None:
            set_seed(cfg.seed)

        self.model.to(cfg.device)

    # ---- core steps ----

    def training_step(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        self.model.train()
        x, y = batch["x"].to(self.cfg.device), batch["y"].to(self.cfg.device)

        self.optimizer.zero_grad()
        out = self.model(x)
        logits = detach_logits(out)
        loss = self.loss_fn(logits, y)
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        return loss.detach(), {"train_loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for batch in loader:
            x, y = batch["x"].to(self.cfg.device), batch["y"].to(self.cfg.device)
            logits = detach_logits(self.model(x))
            loss = self.loss_fn(logits, y)
            loss_sum += float(loss.detach().cpu()) * y.size(0)
            if logits.ndim == 2 and logits.size(1) > 1:
                preds = logits.argmax(dim=1)
            else:
                preds = (logits > 0).long().view_as(y)
            correct += int((preds == y).sum().item())
            total += int(y.size(0))
        return {
            "val_loss": loss_sum / max(total, 1),
            "val_acc": (correct / max(total, 1)) if total else 0.0,
        }

    # ---- fit loop ----

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        step = 0
        for epoch in range(self.cfg.epochs):
            for cb in self.cbs: cb.on_epoch_start(self, epoch)

            for batch in train_loader:
                loss, logs = self.training_step(batch)
                if step % self.cfg.log_every == 0:
                    for cb in self.cbs:
                        cb.on_log(self, step, {"epoch": epoch, **logs})
                step += 1

            epoch_logs = {}
            if val_loader is not None:
                epoch_logs = self.evaluate(val_loader)
            for cb in self.cbs: cb.on_epoch_end(self, epoch, epoch_logs)

    # ---- helpers ----

    @staticmethod
    def make_loader(x: Tensor, y: Tensor, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        ds: Dataset[Tuple[Tensor, Tensor]] = TensorDataset(x, y)
        # Wrap to dicts the engine expects: {"x": x, "y": y}
        class _DictDS(Dataset):
            def __init__(self, base): self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                xi, yi = self.base[i]
                return {"x": xi, "y": yi}
        return DataLoader(_DictDS(ds), batch_size=batch_size, shuffle=shuffle)
