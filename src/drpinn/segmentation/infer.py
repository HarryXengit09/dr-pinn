"""Whole-image prediction by overlapping tiles with smooth blending."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def _window(size: int) -> NDArray[np.float32]:
    """2D blending weight: high in the tile centre, low (not zero) at the edges."""
    w = np.hanning(size + 2)[1:-1].astype(np.float32)
    w = np.maximum(w, 1e-3)
    return np.outer(w, w)


@torch.no_grad()
def predict_image(model: torch.nn.Module, image: NDArray[np.float32], fov: NDArray[np.bool_] | None = None,
                  tile: int = 256, overlap: float = 0.5, batch: int = 8,
                  device: str | torch.device = "cpu") -> NDArray[np.float32]:
    """Sigmoid probabilities (C_out, H, W) for a preprocessed image (C_in, H, W)."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    m = getattr(model, "multiple", 16)
    tile = max(m, (tile // m) * m)
    c, h, w = image.shape
    step = max(m, int(tile * (1 - overlap)) // m * m)
    ph = max(tile, int(np.ceil((h - tile) / step)) * step + tile) if h > tile else tile
    pw = max(tile, int(np.ceil((w - tile) / step)) * step + tile) if w > tile else tile
    padded = np.pad(image, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect" if min(h, w) > 1 else "constant")
    weight = _window(tile)
    model.eval()
    acc = None
    norm = np.zeros((ph, pw), np.float32)
    coords = [(y, x) for y in range(0, ph - tile + 1, step) for x in range(0, pw - tile + 1, step)]
    for s in range(0, len(coords), batch):
        chunk = coords[s:s + batch]
        xb = torch.from_numpy(np.stack([padded[:, y:y + tile, x:x + tile] for y, x in chunk])).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            prob = torch.sigmoid(model(xb).float()).cpu().numpy()
        if acc is None:
            acc = np.zeros((prob.shape[1], ph, pw), np.float32)
        for (y, x), p in zip(chunk, prob):
            acc[:, y:y + tile, x:x + tile] += p * weight
            norm[y:y + tile, x:x + tile] += weight
    out = (acc / norm)[:, :h, :w]
    if fov is not None:
        out[:, ~np.asarray(fov, bool)] = 0.0
    return out
