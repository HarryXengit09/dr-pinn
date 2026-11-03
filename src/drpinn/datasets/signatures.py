"""Image integrity and near-duplicate detection.

Each image is decoded once (corrupt files are caught here), cropped to its
bright field of view, reduced to a 64 x 64 green-channel thumbnail (vessels
have the most contrast in green) and band-pass filtered, ignoring a band just
inside the field-of-view edge (that circular outline is the same in every
photo). Two images are near-duplicates when the correlation of these
signatures is >= DEFAULT_THRESHOLD.

Calibration (synthetic fundus images, 60 different eyes): different eyes
reached at most 0.76; copies rescaled x0.7 to x4, brightened 8 % and JPEG
re-saved (quality 75-92) scored at least 0.987. Cropped, rotated or flipped
copies are not targeted. Pairs between REVIEW_THRESHOLD and DEFAULT_THRESHOLD
are listed for inspection by eye; real photos may behave differently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

SIG_SIZE = 64
BAND = (0.7, 3.0)
DEFAULT_THRESHOLD = 0.92
REVIEW_THRESHOLD = 0.80


@dataclass
class ImageCheck:
    ok: bool
    width: int = 0
    height: int = 0
    mode: str = ""
    error: str = ""


def _crop_to_fov(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Square crop around the field of view, located from its centroid and area
    (robust to stray bright pixels and JPEG colour bleed at the border)."""
    lum = rgb.mean(axis=2)
    thr = max(15.0, 0.15 * float(np.percentile(lum, 99)))
    fov = ndimage.binary_opening(lum > thr, iterations=2)
    lab, n = ndimage.label(fov)
    if n == 0:
        return rgb
    big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    if big.sum() < 0.05 * big.size:
        return rgb
    cy, cx = ndimage.center_of_mass(big)
    r = np.sqrt(big.sum() / np.pi)
    h, w = lum.shape
    r0, r1 = int(max(0, round(cy - r))), int(min(h, round(cy + r)))
    c0, c1 = int(max(0, round(cx - r))), int(min(w, round(cx + r)))
    return rgb[r0:r1, c0:c1]


def image_signature(path: str | Path, size: int = SIG_SIZE) -> tuple[ImageCheck, NDArray[np.float32] | None]:
    from PIL import Image

    try:
        with Image.open(path) as im:
            w, h, mode = im.size[0], im.size[1], im.mode
            if im.format == "JPEG":
                im.draft("RGB", (256, 256))  # fast reduced decode, never below 256 px
            rgb = im.convert("RGB")
            rgb.thumbnail((256, 256))
            a = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:  # corrupt, truncated or unsupported file
        return ImageCheck(False, error=f"{type(exc).__name__}: {exc}"), None
    a = _crop_to_fov(a)[..., 1]
    t = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS),
                   np.float32)
    inside = ndimage.binary_erosion(t > 8, iterations=max(2, size // 12))
    if inside.sum() < 0.2 * t.size:
        inside = np.ones_like(inside)
    bp = ndimage.gaussian_filter(t, BAND[0]) - ndimage.gaussian_filter(t, BAND[1])
    vals = bp[inside]
    std = vals.std()
    vec = np.zeros(bp.shape, np.float32)
    if std > 1e-6:
        vec[inside] = (vals - vals.mean()) / std / np.sqrt(vals.size)
    vec = vec.ravel()
    return ImageCheck(True, w, h, mode), vec.astype(np.float32)


class SignatureCache:
    """Signatures keyed by path + size + modification time, stored as .npz + .json."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta: dict[str, dict] = {}
        self.vecs: dict[str, NDArray[np.float32]] = {}
        js, npz = self.path.with_suffix(".json"), self.path.with_suffix(".npz")
        if js.exists() and npz.exists():
            try:
                self.meta = json.loads(js.read_text())
                with np.load(npz) as z:
                    self.vecs = {k: z[k] for k in z.files}
            except Exception:
                self.meta, self.vecs = {}, {}

    @staticmethod
    def _key(p: Path) -> str:
        st = p.stat()
        return f"{p}|{st.st_size}|{st.st_mtime_ns}"

    def get(self, p: Path) -> tuple[ImageCheck, NDArray[np.float32] | None]:
        key = self._key(p)
        if key in self.meta:
            m = dict(self.meta[key])
            vid = m.pop("vec", "")
            vec = self.vecs.get(vid)
            return ImageCheck(**m), (None if vec is None else vec.astype(np.float32))
        chk, vec = image_signature(p)
        vid = f"v{len(self.meta)}"
        self.meta[key] = {**chk.__dict__, "vec": vid if vec is not None else ""}
        if vec is not None:
            self.vecs[vid] = vec.astype(np.float16)
        return chk, vec

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.with_suffix(".json").write_text(json.dumps(self.meta))
        np.savez_compressed(self.path.with_suffix(".npz"), **self.vecs)


def near_duplicate_pairs(vecs: NDArray[np.float32], threshold: float = DEFAULT_THRESHOLD,
                         chunk: int = 1024) -> list[tuple[int, int, float]]:
    """All pairs i < j whose signature correlation is >= threshold."""
    v = np.asarray(vecs, dtype=np.float32)
    out = []
    for s in range(0, len(v), chunk):
        sim = v[s:s + chunk] @ v.T
        ii, jj = np.nonzero(sim >= threshold)
        for i, j in zip(ii + s, jj):
            if i < j:
                out.append((int(i), int(j), float(sim[i - s, j])))
    return out


def group_duplicates(n: int, pairs: list[tuple[int, int, float]]) -> NDArray[np.int64]:
    """Union-find: group id per item (items with no duplicate get their own group)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid.astype(np.int64)
