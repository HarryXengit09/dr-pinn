"""DRIVE (Digital Retinal Images for Vessel Extraction) file discovery and loading.

Facts checked against drive.grand-challenge.org (Oct 2026): 40 images split
20 training / 20 test; manual vessel annotations are released for the
training images only (test annotations are withheld for the challenge); every
image has a field-of-view mask. Cite Staal et al., IEEE TMI 23(4):501-509, 2004.

Layout-agnostic: works with the official zip (training/images, 1st_manual,
mask) and with copies that renamed folders or converted GIF/TIF to PNG,
matching files by their leading image number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_EXTS = {".gif", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
_ID = re.compile(r"^(\d+)_")


@dataclass(frozen=True)
class DriveSample:
    image_id: str
    manual_path: Path
    fov_path: Path | None
    image_path: Path | None


def _image_id(p: Path) -> str | None:
    m = _ID.match(p.name)
    return m.group(1) if m else None


def find_drive_samples(root: str | Path) -> list[DriveSample]:
    """All images under `root` that have a manual vessel annotation (*manual1*)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DRIVE folder not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _EXTS
             and not p.name.startswith(".")]
    manual, fov, image = {}, {}, {}
    for p in files:
        iid = _image_id(p)
        if iid is None:
            continue
        name = p.name.lower()
        if "manual1" in name:
            manual.setdefault(iid, p)
        elif "manual" in name:
            continue  # e.g. 2nd_manual: other observer, not used here
        elif "mask" in name:
            fov.setdefault(iid, p)
        else:
            image.setdefault(iid, p)
    return [DriveSample(iid, manual[iid], fov.get(iid), image.get(iid))
            for iid in sorted(manual, key=lambda s: int(s))]


def load_binary(path: str | Path) -> NDArray[np.bool_]:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return arr > 127


def load_rgb(path: str | Path) -> NDArray[np.uint8]:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
