"""APTOS 2019 Blindness Detection (Kaggle) discovery and loading.

Checked (Oct 2026): images from Aravind Eye Hospital (India), graded on the
0-4 ICDR scale; strong class imbalance. Download = 10.21 GB (train_images.zip
8.60 GB, test_images.zip 1.61 GB, train.csv, test.csv); only train.csv carries
grades. No licence is stated outside the Kaggle competition rules — read and
accept them on the competition page before downloading.

Because of its size the notebook keeps only a cropped, resized working copy
in Google Drive (scripts/m4_make_working_copy.py). This loader accepts either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .records import ImageRecord, find_column, read_csv_rows

_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class AptosIndex:
    records: list[ImageRecord] = field(default_factory=list)
    unlabeled_images: int = 0
    problems: list[str] = field(default_factory=list)


def find_aptos(root: str | Path) -> AptosIndex:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"APTOS folder not found: {root}")
    labels = None
    for csv_path in sorted(root.rglob("*.csv")):
        rows = read_csv_rows(csv_path)
        cols = list(rows[0]) if rows else []
        if find_column(cols, "id") and find_column(cols, "diagnosis"):
            labels = (csv_path, rows, find_column(cols, "id"), find_column(cols, "diagnosis"))
            break
    if labels is None:
        raise FileNotFoundError(f"no CSV with 'id_code' and 'diagnosis' columns under {root}")
    csv_path, rows, idc, dxc = labels

    images = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXT and not p.name.startswith("."):
            images.setdefault(p.stem, p)
    idx = AptosIndex()
    labeled = set()
    for r in rows:
        iid = r[idc]
        try:
            grade = int(float(r[dxc]))
        except (TypeError, ValueError):
            idx.problems.append(f"unreadable diagnosis for {iid!r}")
            continue
        if not 0 <= grade <= 4:
            idx.problems.append(f"diagnosis {grade} out of range 0-4 for {iid}")
            continue
        p = images.get(iid)
        if p is None:
            idx.problems.append(f"label without image: {iid}")
            continue
        labeled.add(iid)
        idx.records.append(ImageRecord("APTOS2019", "grading", iid, p, "", grade))
    idx.unlabeled_images = len(set(images) - labeled)
    return idx
