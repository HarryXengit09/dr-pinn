"""IDRiD (Indian Diabetic Retinopathy Image Dataset) discovery and loading.

Checked against idrid.grand-challenge.org/Data and IEEE DataPort (Oct 2026):
- 516 images, Kowa VX-10 alpha, 50 degree field, 4288 x 2848 px, JPG. Licence CC BY 4.0.
- A. Segmentation: 81 images with TIF masks for microaneurysms (MA, 81 images),
  haemorrhages (HE, 80), hard exudates (EX, 81), soft exudates (SE, 40) and the
  optic disc (OD, 81). A missing lesion mask means that lesion was not annotated.
- B. Disease Grading: 516 images (413 train / 103 test), CSV with ICDR DR grade
  and risk of macular oedema.
- C. Localization: the same 516 images; CSVs with optic-disc and fovea centres.
Cite: Porwal et al., IEEE DataPort (2018), doi:10.21227/H25W98.

Folder names are matched loosely (any layout works); the segmentation set is
told apart from the grading set by folder name, else by 2- vs 3-digit ids.
Localization images duplicate the grading images and are not indexed again.

The grading TRAINING and TESTING sets reuse the same file names (both start at
IDRiD_001), so images and label rows are matched by (split, name), and grading
records get unique ids "IDRiD_001-train" / "IDRiD_001-test". (Before 0.7.2
they were matched by name only, which paired train labels with test photos.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .records import ImageRecord, find_column, read_csv_rows, split_from_path

LESIONS = ("MA", "HE", "EX", "SE")
_MASK = re.compile(r"^(IDRiD_\d+)_(MA|HE|EX|SE|OD)\.tiff?$", re.I)
_IMG = re.compile(r"^(IDRiD_(\d+))\.(jpe?g|png|tiff?)$", re.I)


@dataclass
class IdridIndex:
    grading: list[ImageRecord] = field(default_factory=list)
    segmentation: list[ImageRecord] = field(default_factory=list)
    od_centre: dict[str, tuple[float, float]] = field(default_factory=dict)     # grading id -> (x, y) px
    fovea_centre: dict[str, tuple[float, float]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def grading_id(name: str, split: str) -> str:
    """Unique id of a grading photo: train and test reuse file names (IDRiD_001 exists in both)."""
    return f"{name}-{split}" if split else name


def _part(path: Path, digits: int) -> str:
    joined = " ".join(p.lower() for p in path.parts)
    if "segmentation" in joined:
        return "segmentation"
    if "grading" in joined:
        return "grading"
    if "locali" in joined:
        return "localization"
    return "segmentation" if digits == 2 else "grading"


def find_idrid(root: str | Path) -> IdridIndex:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"IDRiD folder not found: {root}")
    idx = IdridIndex()
    images: dict[tuple[str, str, str], Path] = {}  # (part, split, name) -> file
    masks: dict[str, dict[str, Path]] = {}
    grade_rows: list[tuple[Path, dict]] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        name = p.name
        if p.suffix.lower() == ".csv":
            rows = read_csv_rows(p)
            cols = list(rows[0]) if rows else []
            if find_column(cols, "retinopathy"):
                grade_rows.extend((p, r) for r in rows)
            elif cols and ("fovea" in name.lower() or "od" in name.lower() or "optic" in name.lower()):
                target = idx.fovea_centre if "fovea" in name.lower() else idx.od_centre
                xc = find_column(cols, "x")
                yc = find_column(cols, "y")
                ic = cols[0]
                split = split_from_path(p.relative_to(root))
                for r in rows:
                    try:
                        target[grading_id(r[ic], split)] = (float(r[xc]), float(r[yc]))
                    except (KeyError, TypeError, ValueError):
                        continue
            continue
        m = _MASK.match(name)
        if m:
            masks.setdefault(m.group(1), {})[m.group(2).upper()] = p
            continue
        m = _IMG.match(name)
        if m:
            part = _part(p.relative_to(root), len(m.group(2)))
            if part != "localization":
                key = (part, split_from_path(p.relative_to(root)), m.group(1))
                if key in images:
                    idx.problems.append(f"two files for {key[2]} ({key[1] or 'no split'}): kept {images[key].name}")
                else:
                    images[key] = p

    # grading part: labels from CSV, images from folders, matched by (split, name)
    by_name: dict[str, dict[str, Path]] = {}
    for (part, split, name), img in images.items():
        if part == "grading":
            by_name.setdefault(name, {})[split] = img
    seen = set()
    for csv_path, r in grade_rows:
        cols = list(r)
        name = r.get(find_column(cols, "image") or cols[0], "")
        gcol, dcol = find_column(cols, "retinopathy"), find_column(cols, "edema") or find_column(cols, "oedema")
        try:
            grade = int(float(r[gcol]))
            dme = int(float(r[dcol])) if dcol and r.get(dcol, "") != "" else None
        except (TypeError, ValueError):
            idx.problems.append(f"unreadable grade for {name!r} in {csv_path.name}")
            continue
        if not 0 <= grade <= 4:
            idx.problems.append(f"DR grade {grade} out of range 0-4 for {name}")
            continue
        split_csv = split_from_path(csv_path.relative_to(root))
        cands = by_name.get(name, {})
        if split_csv in cands:
            split, img = split_csv, cands[split_csv]
        elif "" in cands:  # image folder has no split name: take the label file's
            split, img = split_csv, cands[""]
        elif not split_csv and len(cands) == 1:
            split, img = next(iter(cands.items()))
        else:
            where = f" in the {split_csv} set" if split_csv else ""
            idx.problems.append(f"grading label without image{where}: {name}"
                                + (f" (found only in {sorted(cands)})" if cands else ""))
            continue
        if (split, name) in seen:
            idx.problems.append(f"two label rows for {name} ({split or 'no split'}): kept the first")
            continue
        seen.add((split, name))
        idx.grading.append(ImageRecord("IDRiD", "grading", grading_id(name, split), img, split, grade, dme))
    unlabeled = [f"{n} ({s or 'no split'})" for n, d in by_name.items() for s in d if (s, n) not in seen]
    if unlabeled:
        idx.problems.append(f"{len(unlabeled)} grading images without a label row (e.g. {sorted(unlabeled)[0]})")

    for (part, _split, name), img in sorted(images.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        if part != "segmentation":
            continue
        mk = masks.get(name, {})
        if "OD" not in mk:
            idx.problems.append(f"{name}: no optic-disc mask")
        idx.segmentation.append(ImageRecord(
            "IDRiD", "segmentation", name, img, split_from_path(img.relative_to(root)),
            extra={"masks": {k: str(v) for k, v in sorted(mk.items())}}))
    orphan = sorted(set(masks) - {n for (p, _s, n) in images if p == "segmentation"})
    if orphan:
        idx.problems.append(f"{len(orphan)} lesion-mask sets without an image (e.g. {orphan[0]})")
    return idx


def load_binary_mask(path: str | Path) -> NDArray[np.bool_]:
    """Binary mask from an annotation image in any common encoding.

    Handles 1-bit images, grey images storing 0/1 or 0/255 (8 or 16 bit), palette
    images and colour images (e.g. lesion = red). A pixel is set when its strongest
    colour channel is above half of the image's maximum; alpha is ignored.
    """
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("P", "PA"):
            im = im.convert("RGB")
        a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., :3].max(axis=2) if a.shape[2] >= 3 else a[..., 0]
    a = a.astype(np.float64)
    top = float(a.max()) if a.size else 0.0
    return a > top / 2 if top > 0 else np.zeros(a.shape, bool)


def describe_image_file(path: str | Path) -> str:
    """Mode, size and value range of an image file (for diagnosing unexpected masks)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            a = np.asarray(im)
            return f"{Path(path).name}: mode {im.mode}, {im.size[0]}x{im.size[1]}, values {a.min()}-{a.max()}"
    except Exception as exc:  # report, do not crash the diagnosis
        return f"{Path(path).name}: unreadable ({type(exc).__name__}: {exc})"


def disc_diameter_px(od_mask: NDArray[np.bool_]) -> dict[str, float]:
    """Optic-disc size from its mask: vertical, horizontal and equal-area diameters (px)."""
    from scipy import ndimage

    m = np.asarray(od_mask, bool)
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("optic-disc mask is empty")
    largest = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    rows, cols = np.nonzero(largest)
    return {"vertical": float(rows.max() - rows.min() + 1),
            "horizontal": float(cols.max() - cols.min() + 1),
            "equal_area": float(2.0 * np.sqrt(largest.sum() / np.pi))}
