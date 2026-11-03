"""Leakage-free split assignment across all datasets.

Rules (applied to duplicate GROUPS, never to single images):
1. A group containing any image from an official test set goes entirely to
   'test', in every dataset where it appears (it can never be trained on).
2. Groups whose members disagree on DR grade are flagged as label conflicts
   and excluded by default.
3. Otherwise the group's primary dataset decides:
   - IDRiD segmentation: keep the official train/test split.
   - IDRiD grading (official train): train 85 % / val 15 %, stratified by grade.
   - APTOS 2019 (no labelled official test): train 70 / val 15 / test 15, stratified.
   - DRIVE and RITE: 5 cross-validation folds (split 'cv'), shared through groups.
Allocation is seeded and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from .records import ImageRecord

PRIMARY_ORDER = (("IDRiD", "segmentation"), ("IDRiD", "grading"), ("APTOS2019", "grading"),
                 ("DRIVE", "vessels"), ("RITE", "av"))
FRACTIONS = {("IDRiD", "grading"): {"train": 0.85, "val": 0.15},
             ("APTOS2019", "grading"): {"train": 0.70, "val": 0.15, "test": 0.15}}
CV_DATASETS = {"DRIVE", "RITE"}


def drive_fold(image_id: str | int, seed: int = 0, n_folds: int = 5) -> int:
    """Fixed CV fold of a DRIVE/RITE photograph (they share the same 40 photos).

    Depends only on the photo number, so adding or removing datasets never
    reshuffles folds. Photos 21-40 (annotated training set) and 1-20 are each
    spread evenly: 4 per fold with 5 folds.
    """
    n = int(image_id)
    if not 1 <= n <= 40:
        raise ValueError(f"DRIVE/RITE photo number must be 1-40, got {image_id!r}")
    half = list(range(21, 41)) if n >= 21 else list(range(1, 21))
    order = list(np.random.default_rng([seed, 20260930]).permutation(half))
    return order.index(n) % n_folds


def _allocate(items: list[tuple[int, object]], fractions: dict[str, float], rng) -> dict[int, str]:
    """Stratified systematic allocation.

    Groups are ordered by stratum (grade), shuffled within each stratum, then
    walked once; each goes to the split furthest below its target share of the
    running total. Totals match the fractions and every stratum is spread
    across splits in proportion, even when a stratum is small.
    """
    by_stratum: dict[object, list[int]] = defaultdict(list)
    for gid, stratum in items:
        by_stratum[stratum].append(gid)
    order = []
    for stratum in sorted(by_stratum, key=str):
        gids = sorted(by_stratum[stratum])
        rng.shuffle(gids)
        order.extend(gids)
    names = list(fractions)
    counts = dict.fromkeys(names, 0)
    out: dict[int, str] = {}
    for k, gid in enumerate(order, start=1):
        split = max(names, key=lambda s: (fractions[s] * k - counts[s], fractions[s]))
        counts[split] += 1
        out[gid] = split
    return out


def assign_splits(records: Sequence[ImageRecord], groups: Sequence[int], seed: int = 0,
                  n_folds: int = 5, keep_conflicts: bool = False) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[int(g)].append(i)

    split_of: dict[int, str] = {}
    conflict: set[int] = set()
    pending: dict[tuple[str, str], list[tuple[int, object]]] = defaultdict(list)
    cv_groups: list[int] = []
    moved_to_test = 0
    for gid in sorted(members):
        recs = [records[i] for i in members[gid]]
        grades = {r.dr_grade for r in recs if r.dr_grade is not None}
        if len(grades) > 1:
            conflict.add(gid)
        if any(r.official_split == "test" and r.dataset in ("IDRiD",) for r in recs):
            split_of[gid] = "test"
            moved_to_test += sum(r.official_split != "test" for r in recs)
            continue
        primary = min(recs, key=lambda r: PRIMARY_ORDER.index((r.dataset, r.part))
                      if (r.dataset, r.part) in PRIMARY_ORDER else len(PRIMARY_ORDER))
        key = (primary.dataset, primary.part)
        if primary.dataset in CV_DATASETS:
            cv_groups.append(gid)
        elif key == ("IDRiD", "segmentation"):
            split_of[gid] = primary.official_split or "train"
        elif key in FRACTIONS:
            pending[key].append((gid, max(grades) if grades else None))
        else:
            split_of[gid] = "train"
    for key, items in sorted(pending.items()):
        split_of.update(_allocate(items, FRACTIONS[key], rng))
    fold_of = {}
    for gid in cv_groups:
        photo = next((r.image_id for i in members[gid] if (r := records[i]).dataset in CV_DATASETS), None)
        split_of[gid], fold_of[gid] = "cv", drive_fold(photo, seed, n_folds)

    rows = []
    for i, r in enumerate(records):
        gid = int(groups[i])
        if gid in conflict and not keep_conflicts:
            continue
        rows.append({"dataset": r.dataset, "part": r.part, "image_id": r.image_id,
                     "official_split": r.official_split, "split": split_of[gid],
                     "fold": fold_of.get(gid, ""), "dr_grade": "" if r.dr_grade is None else r.dr_grade,
                     "dme_grade": "" if r.dme_grade is None else r.dme_grade,
                     "referable": "" if r.referable is None else int(r.referable),
                     "dup_group": gid, "label_conflict": int(gid in conflict),
                     "_path": r.path, "_extra": r.extra})
    info = {"groups": len(members), "duplicate_groups": sum(len(m) > 1 for m in members.values()),
            "label_conflict_groups": len(conflict),
            "excluded_for_conflict": 0 if keep_conflicts else sum(len(members[g]) for g in conflict),
            "moved_to_test_by_duplicates": moved_to_test}
    return rows, info


def check_no_leakage(rows: Sequence[dict]) -> list[str]:
    """Every duplicate group must sit in exactly one split (and one fold)."""
    seen: dict[int, set] = defaultdict(set)
    for r in rows:
        seen[r["dup_group"]].add((r["split"], r["fold"]))
    return [f"group {g} spans {sorted(v)}" for g, v in seen.items() if len(v) > 1]


def split_summary(rows: Sequence[dict]) -> dict:
    out: dict = {}
    for (ds, part), rs in sorted(_by(rows, lambda r: (r["dataset"], r["part"])).items()):
        out[f"{ds}/{part}"] = {
            s: dict(sorted(Counter(str(r["dr_grade"]) for r in rr).items()))
            for s, rr in sorted(_by(rs, lambda r: r["split"]).items())}
    return out


def manifest_fingerprint(rows: Sequence[dict]) -> str:
    keys = ("dataset", "part", "image_id", "split", "fold", "dr_grade")
    payload = json.dumps(sorted([[str(r[k]) for k in keys] for r in rows]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _by(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[key(r)].append(r)
    return d
