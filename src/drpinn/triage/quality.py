"""Image quality checks for fundus photographs (heuristic, for research use).

Measured on the field of view (FOV), after cropping it and resizing so it is
QUALITY_SIZE px across, which makes the numbers comparable between cameras:
- fov_fraction: share of the original photo covered by the FOV;
- brightness: mean luminance inside the FOV (0-255);
- contrast: standard deviation / mean of luminance inside the FOV;
- clipped: share of FOV pixels that are nearly black or nearly white;
- sharpness: 100 x std(Laplacian of the smoothed green channel) / mean(green),
  inside the eroded FOV. Blur lowers it.

Flags for brightness, contrast, clipping and sharpness are RELATIVE TO THE
TRAINING PHOTOS: a value outside their 1st-99th percentile range is flagged.
The FOV fraction is only checked against an absolute minimum (15 %), because
it depends on how a dataset was cropped, not on image quality. This is a pipeline sanity check, not a
validated gradability assessment: none of the datasets used here has
gradability labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..segmentation.preprocess import estimate_fov

QUALITY_SIZE = 512
METRICS = ("fov_fraction", "brightness", "contrast", "clipped", "sharpness")
MIN_FOV_FRACTION = 0.15
NO_FUNDUS = "no fundus found"
# metric -> (flag when too low, flag when too high); None = not checked on that side.
# fov_fraction is NOT compared with the training photos: it depends on how a dataset was cropped
# (the APTOS working copy is cropped to the FOV, IDRiD originals are not), not on quality.
# Only the absolute minimum (MIN_FOV_FRACTION) applies. Changed in 0.7.2 after the first real run
# flagged every IDRiD photo as "small field of view".
_FLAGS = {"brightness": ("too dark", "too bright"),
          "contrast": ("low contrast", None),
          "clipped": (None, "over/under-exposed areas"),
          "sharpness": ("blurry", None)}


def _resize(a: NDArray, size: int, mask: bool = False) -> NDArray:
    from PIL import Image

    h, w = a.shape[:2]
    s = size / max(h, w)
    shape = (max(1, round(w * s)), max(1, round(h * s)))
    if mask:
        return np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize(shape, Image.NEAREST)) > 127
    return np.asarray(Image.fromarray(a).resize(shape, Image.BILINEAR))


def quality_metrics(rgb: NDArray[np.uint8]) -> dict[str, float]:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected an RGB image (H, W, 3), got shape {a.shape}")
    a = np.ascontiguousarray(a[..., :3]).astype(np.uint8)
    out = dict.fromkeys(METRICS, float("nan"))
    if float(np.percentile(a.astype(np.float32).mean(axis=2), 99)) < 15:  # nothing bright enough to be a fundus
        out["fov_fraction"] = 0.0
        return out
    fov = estimate_fov(a)
    out["fov_fraction"] = float(fov.mean())
    if fov.sum() < 100:
        return out
    rows, cols = np.flatnonzero(fov.any(1)), np.flatnonzero(fov.any(0))
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    img, m = _resize(a[crop], QUALITY_SIZE), _resize(fov[crop], QUALITY_SIZE, mask=True)
    lum = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    v = lum[m]
    out["brightness"] = float(v.mean())
    out["contrast"] = float(v.std() / max(v.mean(), 1.0))
    out["clipped"] = float(((v < 5) | (v > 250)).mean())
    inner = ndimage.binary_erosion(m, iterations=8)
    if inner.sum() > 100:
        g = img[..., 1].astype(np.float32)
        lap = ndimage.laplace(ndimage.gaussian_filter(g, 1.0))
        out["sharpness"] = float(100 * lap[inner].std() / max(g[inner].mean(), 1.0))
    return out


@dataclass
class QualityReference:
    """Acceptable range per metric, from the training photos (percentiles)."""

    low: dict
    high: dict
    n: int = 0

    @classmethod
    def fit(cls, rows: list[dict], lo: float = 1.0, hi: float = 99.0) -> "QualityReference":
        low, high = {}, {}
        for k in METRICS:
            v = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))], float)
            low[k] = float(np.percentile(v, lo)) if v.size else -np.inf
            high[k] = float(np.percentile(v, hi)) if v.size else np.inf
        return cls(low, high, len(rows))

    @classmethod
    def absolute_only(cls) -> "QualityReference":
        return cls(dict.fromkeys(METRICS, -np.inf), dict.fromkeys(METRICS, np.inf), 0)

    def flags(self, m: dict) -> list[str]:
        if m["fov_fraction"] < MIN_FOV_FRACTION or not np.isfinite(m["brightness"]):
            return [NO_FUNDUS]
        out = []
        for k, (low_flag, high_flag) in _FLAGS.items():
            x = m.get(k, np.nan)
            if not np.isfinite(x):
                continue
            if low_flag and x < self.low[k]:
                out.append(low_flag)
            if high_flag and x > self.high[k]:
                out.append(high_flag)
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QualityReference":
        return cls(dict(d["low"]), dict(d["high"]), int(d.get("n", 0)))
