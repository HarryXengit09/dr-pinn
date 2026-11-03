"""Images for the triage model: square FOV crops cached at training size, GPU augmentation, synthetic graded eyes.

Every photo (APTOS working copy or IDRiD original) is read at <= 1024 px (the
size of the APTOS working copy, so quality metrics are comparable), cropped to
its field of view, padded to a square with black and resized to `size` px, then
cached as a lossless PNG. Quality metrics (quality.py) are computed in the same
pass. The cache is rebuilt only for missing files, so an
interrupted run resumes.

Training keeps the cached uint8 images in RAM and augments whole batches on
the device: random flips, rotation by any angle (a fundus has no "up"),
brightness and contrast jitter, then ImageNet normalisation.
"""

from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from ..datasets.signatures import _crop_to_fov
from .quality import quality_metrics

WORK_SIDE = 1024  # photos are read at most this size (the APTOS working copy is already 1024 px)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def square_fundus(rgb: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """FOV crop -> black-padded square -> size x size (LANCZOS)."""
    from PIL import Image

    a = np.asarray(rgb)[..., :3]
    crop = _crop_to_fov(a.astype(np.float32)).astype(np.uint8)
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 3), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return np.asarray(Image.fromarray(sq).resize((size, size), Image.LANCZOS))


def _key_file(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key.replace('/', '__')}.png"


def read_working(path: str | Path, side: int = WORK_SIDE) -> NDArray[np.uint8]:
    """RGB with the longest side at most `side` px (JPEGs are decoded at reduced size: fast for 12 MP photos)."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        if max(w, h) > side:
            s = side / max(w, h)
            im.draft("RGB", (math.ceil(w * s), math.ceil(h * s)))  # never below the requested size
        im = im.convert("RGB")
        if max(im.size) > side:
            im.thumbnail((side, side), Image.LANCZOS)
        return np.asarray(im)


def limit_side(rgb: NDArray[np.uint8], side: int = WORK_SIDE) -> NDArray[np.uint8]:
    """Downscale an in-memory photo so its longest side is at most `side` px (as read_working does for files)."""
    from PIL import Image

    a = np.asarray(rgb)[..., :3]
    if max(a.shape[:2]) <= side:
        return np.ascontiguousarray(a)
    im = Image.fromarray(np.ascontiguousarray(a))
    im.thumbnail((side, side), Image.LANCZOS)
    return np.asarray(im)


def _cache_one(job: tuple[str, str, int, str]) -> tuple[str, dict | None, str]:
    from PIL import Image

    key, src, size, out = job
    try:
        rgb = read_working(src)
        q = quality_metrics(rgb)
        tmp = Path(out).with_suffix(".part")
        Image.fromarray(square_fundus(rgb, size)).save(tmp, "PNG")
        tmp.replace(out)
        return key, q, ""
    except Exception as exc:  # report and carry on
        return key, None, f"{type(exc).__name__}: {exc}"


def build_cache(items: list[tuple[str, str | Path]], size: int, cache_dir: str | Path, workers: int = 1,
                progress_every: int = 0) -> tuple[dict[str, dict], dict[str, str]]:
    """Cache every (key, path); returns (quality metrics per key, errors per key)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    qfile = cache_dir / "quality.json"
    quality = json.loads(qfile.read_text()) if qfile.exists() else {}
    jobs = [(k, str(p), size, str(_key_file(cache_dir, k))) for k, p in items
            if not (_key_file(cache_dir, k).exists() and k in quality)]
    errors: dict[str, str] = {}
    t0 = time.perf_counter()

    def done(res, i):
        k, q, err = res
        if err:
            errors[k] = err
        else:
            quality[k] = q
        if progress_every and (i % progress_every == 0 or i == len(jobs)):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            print(f"  cached {i}/{len(jobs)} new images ({rate:.1f}/s)", flush=True)
            qfile.write_text(json.dumps(quality))

    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            for i, res in enumerate(ex.map(_cache_one, jobs, chunksize=8), 1):
                done(res, i)
    else:
        for i, job in enumerate(jobs, 1):
            done(_cache_one(job), i)
    qfile.write_text(json.dumps(quality))
    return {k: quality[k] for k, _ in items if k in quality}, errors


def load_cached(keys: list[str], cache_dir: str | Path) -> NDArray[np.uint8]:
    from PIL import Image

    cache_dir = Path(cache_dir)
    first = np.asarray(Image.open(_key_file(cache_dir, keys[0])).convert("RGB"))
    out = np.empty((len(keys),) + first.shape, np.uint8)
    out[0] = first
    for i, k in enumerate(keys[1:], 1):
        out[i] = np.asarray(Image.open(_key_file(cache_dir, k)).convert("RGB"))
    return out


def _normalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def to_input(batch: NDArray[np.uint8] | torch.Tensor, device="cpu") -> torch.Tensor:
    """uint8 (B, H, W, 3) -> normalised float (B, 3, H, W), no augmentation."""
    x = torch.from_numpy(np.array(batch, np.uint8)).to(device).permute(0, 3, 1, 2).float() / 255
    return _normalise(x)


def augment(batch: NDArray[np.uint8], generator: torch.Generator, device="cpu", jitter: float = 0.15) -> torch.Tensor:
    """Random flip, rotation (any angle), brightness and contrast jitter; returns normalised (B, 3, H, W)."""
    x = torch.as_tensor(np.asarray(batch)).to(device).permute(0, 3, 1, 2).float() / 255
    b = x.shape[0]

    def u(lo, hi):
        return (torch.rand(b, generator=generator) * (hi - lo) + lo).to(device)

    theta = u(0, 2 * math.pi)
    flip = torch.where(torch.rand(b, generator=generator) < 0.5, -1.0, 1.0).to(device)
    cos, sin = torch.cos(theta), torch.sin(theta)
    mat = torch.stack([torch.stack([cos * flip, -sin, torch.zeros_like(cos)], 1),
                       torch.stack([sin * flip, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    inside = x.amax(1, keepdim=True) > 0.02
    mean = (x * inside).sum((1, 2, 3), keepdim=True) / (3 * inside.sum((1, 2, 3), keepdim=True).clamp_min(1))
    c = u(1 - jitter, 1 + jitter).view(b, 1, 1, 1)
    br = u(1 - jitter, 1 + jitter).view(b, 1, 1, 1)
    x = ((x - mean) * c + mean) * br
    x = torch.where(inside, x.clamp(0, 1), torch.zeros_like(x))
    return _normalise(x)


# --- synthetic graded eyes (software tests and controls only) -----------------------

LESIONS_PER_GRADE = ((0, 0), (3, 0), (8, 2), (16, 5), (28, 9))  # (red dots, yellow exudates)
GRADE_SHARES = (0.49, 0.10, 0.27, 0.05, 0.09)  # roughly APTOS 2019


@lru_cache(maxsize=32)
def _base(size: int, k: int) -> NDArray[np.uint8]:
    from ..datasets.synthetic import fake_fundus_av

    return fake_fundus_av(size, 1000 + k)[0]


def synthetic_graded_eye(grade: int, size: int, seed: int, lesion_grade: int | None = None,
                         tint: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> NDArray[np.uint8]:
    """A fake fundus with lesions whose number grows with `lesion_grade` (default = grade).

    Passing a different `lesion_grade` breaks the link between image and label (null control).
    """
    rng = np.random.default_rng(seed)
    img = np.rot90(_base(size, int(rng.integers(16))), int(rng.integers(4))).astype(np.float32).copy()
    g = grade if lesion_grade is None else lesion_grade
    n_red, n_yellow = LESIONS_PER_GRADE[g]
    yy, xx = np.mgrid[:size, :size]
    c = size / 2
    for n, colour, rad in ((n_red, (95, 15, 10), 0.012), (n_yellow, (235, 200, 70), 0.010)):
        for _ in range(n):
            while True:
                r, cc = rng.uniform(0.12, 0.88, 2) * size
                if (r - c) ** 2 + (cc - c) ** 2 < (0.40 * size) ** 2:
                    break
            rr = rad * size * rng.uniform(0.8, 1.4)
            m = (yy - r) ** 2 + (xx - cc) ** 2 <= rr ** 2
            img[m] = colour
    img *= np.asarray(tint, np.float32)
    img += rng.normal(0, 3, img.shape)
    fov = (yy - c) ** 2 + (xx - c) ** 2 <= (0.47 * size) ** 2
    return np.where(fov[..., None], np.clip(img, 0, 255), 0).astype(np.uint8)


def synthetic_grades(n: int, seed: int) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    return rng.choice(len(GRADE_SHARES), size=n, p=GRADE_SHARES)
