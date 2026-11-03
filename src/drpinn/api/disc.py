"""Optic-disc estimate for photos that come without a disc annotation (API / dashboard).

Heuristic, stated as an assumption: the disc is where the smoothed image is
brightest AND (when a vessel map is available) where vessels are densest,
inside the slightly eroded field of view. Its diameter is assumed to be
DISC_TO_FOV x the field-of-view diameter. 0.16 is the median measured on the
81 IDRiD photos (0.14, the textbook value for a 1.8 mm disc in a 45-50 degree
photo, was used until 0.8.1). On IDRiD the estimated centre landed inside the
expert disc in 98-99 % of photos (scripts/m9_check_disc.py); a user can always
give the disc position instead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

DISC_TO_FOV = 0.16  # median measured on the 81 IDRiD photos (0.161; scripts/m9_check_disc.py)


def _z(a: NDArray, m: NDArray[np.bool_]) -> NDArray:
    v = a[m]
    sd = float(v.std()) or 1.0
    return (a - float(v.mean())) / sd


def estimate_disc(rgb: NDArray[np.uint8], fov: NDArray[np.bool_], vessel_prob: NDArray | None = None,
                  disc_to_fov: float = DISC_TO_FOV) -> tuple[tuple[float, float], float]:
    """((row, col), diameter px) of the likely optic disc."""
    fov = np.asarray(fov, bool)
    if fov.sum() < 100:
        raise ValueError("no field of view found: cannot place the optic disc")
    fov_d = 2 * np.sqrt(fov.sum() / np.pi)
    inner = ndimage.binary_erosion(fov, iterations=max(1, int(0.03 * fov_d)))
    if inner.sum() < 50:
        inner = fov
    sigma = max(1.0, fov_d / 25)
    lum = np.asarray(rgb, np.float32)[..., :3].mean(axis=2)
    lum = np.where(fov, lum, float(lum[fov].mean()))  # no dark rim pulling the smoothing
    score = _z(ndimage.gaussian_filter(lum, sigma), inner)
    if vessel_prob is not None:
        v = np.where(fov, np.asarray(vessel_prob, np.float32), 0)
        score = score + _z(ndimage.gaussian_filter(v, 1.5 * sigma), inner)
    score = np.where(inner, score, -np.inf)
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    return (float(r), float(c)), float(disc_to_fov * fov_d)


def disc_mask(shape: tuple[int, int], centre_rc: tuple[float, float], diameter: float) -> NDArray[np.bool_]:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - centre_rc[0]) ** 2 + (xx - centre_rc[1]) ** 2 <= (diameter / 2) ** 2
