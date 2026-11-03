"""Fundus image preprocessing for segmentation (numpy only).

Input channels (4): R, G, B each standardised by its mean and standard deviation
inside the field of view (FOV), plus contrast-limited adaptive histogram
equalisation (CLAHE) of the green channel, where vessels have most contrast.
Pixels outside the FOV are set to 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

IN_CHANNELS = 4


def estimate_fov(rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Camera aperture from brightness: largest bright region, holes filled."""
    lum = np.asarray(rgb, np.float32).mean(axis=2)
    thr = max(15.0, 0.15 * float(np.percentile(lum, 99)))
    m = ndimage.binary_opening(lum > thr, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.ones(lum.shape, bool)
    big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return ndimage.binary_fill_holes(big)


def preprocess(rgb: NDArray[np.uint8], fov: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
    """RGB uint8 (H, W, 3) -> float32 (4, H, W)."""
    from skimage.exposure import equalize_adapthist

    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected an RGB image (H, W, 3), got shape {a.shape}")
    a = a[..., :3].astype(np.float32)
    fov = estimate_fov(a.astype(np.uint8)) if fov is None else np.asarray(fov, bool)
    if fov.shape != a.shape[:2]:
        raise ValueError("FOV mask must match the image size")
    if fov.sum() < 100:
        raise ValueError("field of view is (almost) empty")
    out = np.zeros((IN_CHANNELS, *a.shape[:2]), np.float32)
    for c in range(3):
        v = a[..., c][fov]
        out[c] = (a[..., c] - v.mean()) / (v.std() + 1e-6)
    clahe = equalize_adapthist(a[..., 1] / 255.0, clip_limit=0.01).astype(np.float32)
    v = clahe[fov]
    out[3] = (clahe - v.mean()) / (v.std() + 1e-6)
    out[:, ~fov] = 0.0
    return out
