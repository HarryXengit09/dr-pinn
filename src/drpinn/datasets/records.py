"""Common image record used by every dataset loader and by the split manifest."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# ICDR scale used by IDRiD and APTOS: 0 none, 1 mild, 2 moderate, 3 severe NPDR, 4 PDR.
DR_GRADE_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
# Project definition (docs/PLAN.md, A15): referable DR = ICDR grade >= 2.
# Macular oedema is NOT part of it, because APTOS has no DME label.
REFERABLE_MIN_GRADE = 2


@dataclass
class ImageRecord:
    dataset: str                 # "IDRiD", "APTOS2019", "DRIVE", "RITE"
    part: str                    # e.g. "grading", "segmentation", "vessels", "av"
    image_id: str
    path: Path                   # absolute path to the fundus image
    official_split: str          # "train", "test" or "" when the source defines none
    dr_grade: int | None = None
    dme_grade: int | None = None
    extra: dict = field(default_factory=dict)  # e.g. paths of lesion / vessel / A-V masks

    @property
    def referable(self) -> bool | None:
        return None if self.dr_grade is None else self.dr_grade >= REFERABLE_MIN_GRADE


def split_from_path(path: Path) -> str:
    """'train' / 'test' from folder or file names like 'a. Training Set', 'Testing Labels'."""
    parts = [p.lower() for p in path.parts]
    for p in reversed(parts):
        if "train" in p:
            return "train"
        if "test" in p:
            return "test"
    return ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV rows with stripped header names and values; skips fully empty rows."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader, [])]
        rows = []
        for raw in reader:
            if not any(c.strip() for c in raw):
                continue
            rows.append({h: (raw[i].strip() if i < len(raw) else "") for i, h in enumerate(header) if h})
    return rows


def find_column(columns: list[str], *keywords: str) -> str | None:
    for c in columns:
        lc = c.lower()
        if all(k in lc for k in keywords):
            return c
    return None
