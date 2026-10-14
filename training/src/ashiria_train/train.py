"""Step 5: train SignNet on the built dataset."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import Config, path
from .data import Cache
from .features import FEATURE_DIM, SPEC_VERSION
from .model import SignNet


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SignDataset(Dataset):
    def __init__(self, cache: Cache, indices: np.ndarray, aug: dict | None = None, seed: int = 0):
        self.cache, self.indices, self.aug = cache, np.asarray(indices), aug
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        return torch.from_numpy(self.cache.features(i, self.rng, self.aug)), int(self.cache.y[i])


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn) -> tuple[float, float]:
    model.eval()
    loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss / max(n, 1), correct / max(n, 1)


def train(cfg: Config, name: str | None = None) -> Path:
    t = cfg.train
    torch.manual_seed(t.seed)
    cache = Cache(cfg)
    tr, va = np.flatnonzero(cache.split == "train"), np.flatnonzero(cache.split == "val")
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("Train or validation split is empty. Run `ashiria build` and check its summary.")

    plain = np.stack([cache.features(i) for i in tr]).reshape(-1, FEATURE_DIM)
    model_args = {"in_dim": FEATURE_DIM, "num_classes": len(cache.classes), **dict(t.model)}
    model = SignNet(**model_args)
    model.set_normalization(plain.mean(0), plain.std(0))
    device = pick_device(t.device)
    model.to(device)
    print(f"{len(tr)} train / {len(va)} val clips, {len(cache.classes)} labels, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device {device}")

    train_dl = DataLoader(SignDataset(cache, tr, dict(t.augment), seed=t.seed), batch_size=t.batch_size,
                          shuffle=True, drop_last=len(tr) > t.batch_size)
    val_dl = DataLoader(SignDataset(cache, va), batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    total, warm = t.epochs * len(train_dl), t.warmup_epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)

    run = path(cfg, "runs") / (name or time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    best_acc, best_loss, stale = -1.0, float("inf"), 0

    with open(run / "metrics.csv", "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])
        for epoch in range(1, t.epochs + 1):
            start = time.time()
            model.train()
            loss_sum = correct = n = 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                n += len(y)
            val_loss, val_acc = evaluate_loader(model, val_dl, device, loss_fn)
            log.writerow([epoch, round(loss_sum / n, 4), round(correct / n, 4), round(val_loss, 4), round(val_acc, 4),
                          f"{sched.get_last_lr()[0]:.2e}", round(time.time() - start, 1)])
            fh.flush()

            improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
            if improved:
                best_acc, best_loss, stale = val_acc, val_loss, 0
                torch.save({
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_args": model_args,
                    "classes": cache.classes,
                    "seq_len": cache.seq_len,
                    "spec_version": SPEC_VERSION,
                    "epoch": epoch,
                    "val_acc": val_acc,
                }, run / "best.pt")
            else:
                stale += 1
            print(f"epoch {epoch:3d}  train {correct / n:.3f}  val {val_acc:.3f}  best {best_acc:.3f}"
                  + ("  *" if improved else ""))
            if stale >= t.patience:
                print(f"No improvement for {t.patience} epochs, stopping.")
                break
    print(f"Best validation accuracy {best_acc:.3f}. Run saved to {run}")
    return run
