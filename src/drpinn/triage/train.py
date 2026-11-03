"""Training with checkpoint/resume (survives Colab disconnects), model selection on validation QWK.

Files in the checkpoint folder:
  last.pt   model + optimiser state after the latest finished epoch (to resume)
  best.pt   weights of the epoch with the highest validation QWK (used for testing)
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from .. import __version__
from .data import augment, to_input
from .metrics import N_GRADES, nll, quadratic_weighted_kappa
from .model import build_model


@dataclass
class TriageConfig:
    arch: str = "mobilenet_v3_large"
    pretrained: bool = True
    image_size: int = 512
    epochs: int = 20
    batch: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    class_weighting: str = "sqrt"   # "sqrt" (inverse square root of frequency) or "none"
    warmup_epochs: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def class_weights(grades: NDArray, mode: str = "sqrt", n: int = N_GRADES) -> NDArray:
    if mode == "none":
        return np.ones(n)
    if mode != "sqrt":
        raise ValueError(f"unknown class weighting {mode!r}")
    counts = np.bincount(np.asarray(grades, int), minlength=n).astype(float)
    w = np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0.0)
    return w / (w[counts > 0] * counts[counts > 0]).sum() * counts.sum()  # mean weight per image = 1


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, images: NDArray[np.uint8], device="cpu", batch: int = 64) -> NDArray:
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            out.append(model(to_input(images[i:i + batch], device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_GRADES))


def load_triage(path: str | Path, device="cpu") -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"triage model not found: {path} — train it first (notebook step 15)")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = TriageConfig(**ck["config"])
    model = build_model(cfg.arch, N_GRADES, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def train_triage(train_x: NDArray[np.uint8], train_y: NDArray, val_x: NDArray[np.uint8], val_y: NDArray,
                 cfg: TriageConfig, ckpt_dir: str | Path, device="cpu", stop_after_epochs: int | None = None,
                 meta: dict | None = None, log=print) -> dict:
    """Train or resume. Returns the training state (history, best epoch, complete flag)."""
    ckpt_dir = Path(ckpt_dir)
    last, best = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.arch, N_GRADES, cfg.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_x) / cfg.batch))
    total, warm = cfg.epochs * steps_per_epoch, max(1, int(cfg.warmup_epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weights = torch.tensor(class_weights(train_y, cfg.class_weighting), dtype=torch.float32, device=device)
    state = {"epoch": 0, "history": [], "best_epoch": 0, "best_val_qwk": -np.inf, "complete": False}
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck.get("config") != cfg.to_dict():
            raise ValueError(f"{last} was trained with different settings; delete the folder or change --ckpt-dir")
        model.load_state_dict(ck["model"])
        state = ck["state"]
        if state["complete"]:
            return state
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        log(f"  resuming from epoch {state['epoch']}/{cfg.epochs}")
    meta = {**(meta or {}), "drpinn_version": __version__, "torch_version": torch.__version__, "device": str(device),
            "train_images": int(len(train_x)), "val_images": int(len(val_x))}
    y_t = torch.as_tensor(np.asarray(train_y, int))
    start = state["epoch"]
    while state["epoch"] < cfg.epochs:
        if stop_after_epochs is not None and state["epoch"] - start >= stop_after_epochs:
            return state
        ep = state["epoch"]
        g = torch.Generator().manual_seed(cfg.seed * 10007 + ep)
        perm = torch.randperm(len(train_x), generator=g).numpy()
        model.train()
        t0, losses = time.perf_counter(), []
        for i in range(0, len(perm), cfg.batch):
            idx = np.sort(perm[i:i + cfg.batch])
            if idx.size < 2:  # BatchNorm needs > 1 image
                continue
            x = augment(train_x[idx], g, device)
            y = y_t[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, weight=weights, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        val_logits = predict_logits(model, val_x, device)
        val_pred = val_logits.argmax(1)
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_loss": nll(val_logits, val_y),
               "val_qwk": quadratic_weighted_kappa(val_y, val_pred), "val_accuracy": float((val_pred == val_y).mean()),
               "seconds": round(time.perf_counter() - t0, 1)}
        state["history"].append(rec)
        state["epoch"] = ep + 1
        if (np.isfinite(rec["val_qwk"]) and rec["val_qwk"] > state["best_val_qwk"]) or state["best_epoch"] == 0:
            state["best_val_qwk"], state["best_epoch"] = rec["val_qwk"], ep + 1
            _atomic_save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": ep + 1,
                          "val": rec, "meta": meta}, best)
        state["complete"] = state["epoch"] >= cfg.epochs
        log(f"  epoch {ep + 1}/{cfg.epochs}  loss {rec['train_loss']:.3f}  val QWK {rec['val_qwk']:.3f}  "
            f"val acc {rec['val_accuracy']:.3f}  ({rec['seconds']:.0f} s)")
        _atomic_save({"model": model.state_dict(), "config": cfg.to_dict(), "state": state, "meta": meta,
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, last)
    return state
