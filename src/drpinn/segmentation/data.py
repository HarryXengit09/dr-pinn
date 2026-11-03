"""Training samples (DRIVE vessels + optional RITE artery/vein labels) and a patch sampler."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..datasets.drive import find_drive_samples, load_binary, load_rgb
from ..datasets.rite import decode_av, find_rite
from .preprocess import preprocess


@dataclass
class SegSample:
    image_id: str                         # DRIVE/RITE photo number, e.g. "21"
    split: str                            # "train" (DRIVE training 21-40) or "test" (1-20)
    image: NDArray[np.float32]            # preprocessed (4, H, W)
    rgb: NDArray[np.uint8]
    fov: NDArray[np.bool_]
    vessel: NDArray[np.bool_]
    vessel_source: str                    # "DRIVE 1st observer" / "RITE test (DRIVE 2nd observer)"
    av_labels: NDArray[np.uint8] | None = None   # 0 bg, 1 artery, 2 vein, 3 overlap, 4 uncertain

    @property
    def has_av(self) -> bool:
        return self.av_labels is not None

    def targets(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """(target, weight), each (3, H, W): vessel / artery / vein."""
        y = np.zeros((3, *self.fov.shape), np.float32)
        w = np.zeros_like(y)
        y[0] = self.vessel
        w[0] = self.fov
        if self.av_labels is not None:
            lab = self.av_labels
            y[1] = (lab == 1) | (lab == 3)
            y[2] = (lab == 2) | (lab == 3)
            known = ((lab >= 1) & (lab <= 3)) & self.fov
            w[1] = w[2] = known
        return y, w


def _fov_from_rgb(rgb):
    from .preprocess import estimate_fov
    return estimate_fov(rgb)


def load_drive_rite(drive_dir: str | Path | None, rite_dir: str | Path | None = None) -> list[SegSample]:
    """DRIVE training photos (expert vessel masks) plus RITE A/V labels where present.

    RITE test photos (1-20) are added as split 'test' with RITE's vessel
    reference (DRIVE's second observer) and A/V labels: a held-out test set.
    """
    samples: dict[str, SegSample] = {}
    if drive_dir is not None and Path(drive_dir).exists():
        for s in find_drive_samples(drive_dir):
            if s.image_path is None:
                continue
            rgb = load_rgb(s.image_path)
            fov = load_binary(s.fov_path) if s.fov_path is not None else _fov_from_rgb(rgb)
            samples[str(int(s.image_id))] = SegSample(
                str(int(s.image_id)), "train" if int(s.image_id) >= 21 else "test", preprocess(rgb, fov), rgb,
                fov, load_binary(s.manual_path) & fov, "DRIVE 1st observer")
    if rite_dir is not None and Path(rite_dir).exists():
        for r in find_rite(rite_dir).records:
            iid = str(int(r.image_id))
            av = decode_av(load_rgb(r.extra["av"]))
            if iid in samples:
                if av.shape != samples[iid].fov.shape:
                    raise ValueError(f"RITE A/V map {iid} does not match the DRIVE image size")
                samples[iid].av_labels = av
            elif int(iid) <= 20 and r.extra.get("vessel"):
                rgb = load_rgb(r.path)
                fov = _fov_from_rgb(rgb)
                samples[iid] = SegSample(iid, "test", preprocess(rgb, fov), rgb, fov,
                                         load_binary(r.extra["vessel"]) & fov,
                                         "RITE test (DRIVE 2nd observer)", av)
    return [samples[k] for k in sorted(samples, key=int)]


def augment(x: NDArray, y: NDArray, w: NDArray, rng: np.random.Generator):
    k = int(rng.integers(4))
    x, y, w = (np.rot90(a, k, axes=(1, 2)) for a in (x, y, w))
    if rng.random() < 0.5:
        x, y, w = x[:, :, ::-1], y[:, :, ::-1], w[:, :, ::-1]
    x = x * rng.uniform(0.85, 1.15, (x.shape[0], 1, 1)) + rng.uniform(-0.15, 0.15, (x.shape[0], 1, 1))
    return (np.ascontiguousarray(a, dtype=np.float32) for a in (x, y, w))


class PatchSampler:
    """Random training patches; a share of them centred on vessel pixels."""

    def __init__(self, samples: list[SegSample], patch: int = 96, vessel_fraction: float = 0.5,
                 seed: int = 0, use_av: bool = True):
        if not samples:
            raise ValueError("no training samples")
        self.samples, self.patch, self.vessel_fraction = samples, patch, vessel_fraction
        self.rng = np.random.default_rng(seed)
        self.targets = []
        for s in samples:
            y, w = s.targets()
            if not use_av:
                w[1:] = 0
            self.targets.append((y, w))
        self.fov_idx = [np.flatnonzero(s.fov.ravel()) for s in samples]
        self.vessel_idx = [np.flatnonzero(s.vessel.ravel()) for s in samples]

    def batch(self, n: int) -> tuple[NDArray, NDArray, NDArray]:
        xs, ys, ws = [], [], []
        p, half = self.patch, self.patch // 2
        for _ in range(n):
            i = int(self.rng.integers(len(self.samples)))
            s = self.samples[i]
            h, w_ = s.fov.shape
            pool = self.vessel_idx[i] if (self.rng.random() < self.vessel_fraction and self.vessel_idx[i].size) \
                else self.fov_idx[i]
            r, c = divmod(int(pool[self.rng.integers(pool.size)]), w_)
            r0 = int(np.clip(r - half, 0, max(0, h - p)))
            c0 = int(np.clip(c - half, 0, max(0, w_ - p)))
            y, wt = self.targets[i]
            sl = (slice(None), slice(r0, r0 + p), slice(c0, c0 + p))
            x, yy, ww = augment(s.image[sl], y[sl], wt[sl], self.rng)
            if x.shape[1:] != (p, p):  # image smaller than the patch: pad
                pad = ((0, 0), (0, p - x.shape[1]), (0, p - x.shape[2]))
                x, yy, ww = np.pad(x, pad), np.pad(yy, pad), np.pad(ww, pad)
            xs.append(x)
            ys.append(yy)
            ws.append(ww)
        return np.stack(xs), np.stack(ys), np.stack(ws)


def photo_number(image_id: str) -> int:
    m = re.match(r"\d+", str(image_id))
    if not m:
        raise ValueError(f"not a DRIVE/RITE photo id: {image_id!r}")
    return int(m.group())
