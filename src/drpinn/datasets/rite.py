"""RITE (Retinal Images vessel Tree Extraction): artery/vein labels for the 40 DRIVE images.

Checked against eye.medicine.uiowa.edu/rite-dataset (Oct 2026): 40 image sets
(20 training, 20 test, as DRIVE); fundus photos TIF, vessel and A/V reference
standards PNG. A/V colours: arteries red, veins blue, artery-vein overlap
green, uncertain white. The test-set vessel reference is DRIVE's 2nd observer.
Access via a download-request form. Cite: Hu, Abramoff & Garvin, MICCAI 2013,
16(Pt 2):436-443.

Exact folder names inside the RITE download are not documented on that page;
files are matched by their leading image number and by folder names
containing 'av', 'vessel' or 'image', and every A/V map is checked by colour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .records import ImageRecord, split_from_path

AV_LABELS = {0: "background", 1: "artery", 2: "vein", 3: "overlap", 4: "uncertain"}
_ID = re.compile(r"^(\d+)_")


@dataclass
class RiteIndex:
    records: list[ImageRecord] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def decode_av(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """RITE colour A/V map -> labels 0 background, 1 artery, 2 vein, 3 overlap, 4 uncertain."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("A/V map must be an RGB image")
    r, g, b = (a[..., k] > 127 for k in range(3))
    out = np.zeros(a.shape[:2], np.uint8)
    out[r & ~g & ~b] = 1
    out[b & ~r & ~g] = 2
    out[g & ~r & ~b] = 3
    out[r & g & b] = 4
    return out


def find_rite(root: str | Path) -> RiteIndex:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"RITE folder not found: {root}")
    av, vessel, image = {}, {}, {}
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        m = _ID.match(p.name)
        if not m:
            continue
        rel = p.relative_to(root)
        folders = " ".join(x.lower() for x in rel.parts[:-1])
        key = (m.group(1), split_from_path(rel))
        if re.search(r"\bav\b|artery|a-v|a_v", folders):
            av[key] = p
        elif "vessel" in folders or "manual" in p.name.lower():
            vessel[key] = p
        elif "image" in folders or p.suffix.lower() in {".tif", ".tiff"}:
            image[key] = p
    idx = RiteIndex()
    for key in sorted(av, key=lambda k: (k[1], int(k[0]))):
        if key not in image:
            idx.problems.append(f"A/V map without fundus image: {av[key].name}")
            continue
        idx.records.append(ImageRecord("RITE", "av", key[0], image[key], key[1],
                                       extra={"av": str(av[key]), "vessel": str(vessel.get(key, ""))}))
    return idx
