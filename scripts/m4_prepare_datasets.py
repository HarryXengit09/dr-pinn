"""M4: verify the datasets, find duplicates, and write leakage-free splits.

Looks under --data-dir for DRIVE/, RITE/, IDRiD/ and a folder starting with
APTOS; any that are missing are reported and skipped. For every image found:
decode it (corrupt files are excluded), compute a duplicate signature, group
near-duplicates across ALL datasets, assign splits by group (see
src/drpinn/datasets/splits.py) and check that no group crosses splits.

Outputs
  <data-dir>/splits/manifest.csv          one row per usable image (paths relative to data-dir)
  <out>/integrity_report.json             counts, problems, duplicates, split summary, fingerprint
  <out>/duplicate_pairs.csv               duplicates and borderline pairs to inspect by eye

    python scripts/m4_prepare_datasets.py --data-dir data --out reports/m4_data
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn.datasets.aptos import find_aptos  # noqa: E402
from drpinn.datasets.drive import find_drive_samples  # noqa: E402
from drpinn.datasets.idrid import find_idrid  # noqa: E402
from drpinn.datasets.records import ImageRecord, split_from_path  # noqa: E402
from drpinn.datasets.rite import find_rite  # noqa: E402
from drpinn.datasets.signatures import (  # noqa: E402
    DEFAULT_THRESHOLD, REVIEW_THRESHOLD, SignatureCache, group_duplicates, near_duplicate_pairs,
)
from drpinn.datasets.splits import assign_splits, check_no_leakage, manifest_fingerprint, split_summary  # noqa: E402

COLUMNS = ["dataset", "part", "image_id", "rel_path", "official_split", "split", "fold", "dr_grade",
           "dme_grade", "referable", "dup_group", "label_conflict", "extra"]


def discover(data_dir: Path) -> tuple[list[ImageRecord], dict]:
    records: list[ImageRecord] = []
    found: dict[str, dict] = {}

    d = data_dir / "DRIVE"
    if d.exists():
        samples = find_drive_samples(d)
        recs = [ImageRecord("DRIVE", "vessels", s.image_id, s.image_path, split_from_path(s.manual_path.relative_to(d)),
                            extra={"manual": str(s.manual_path), "fov": str(s.fov_path or "")})
                for s in samples if s.image_path is not None]
        found["DRIVE"] = {"images_with_vessel_masks": len(recs),
                          "problems": [f"{len(samples) - len(recs)} masks without a fundus image"]
                          if len(samples) != len(recs) else []}
        records += recs
    d = data_dir / "RITE"
    if d.exists():
        idx = find_rite(d)
        found["RITE"] = {"images_with_av_labels": len(idx.records), "problems": idx.problems}
        records += idx.records
    d = data_dir / "IDRiD"
    if d.exists():
        idx = find_idrid(d)
        found["IDRiD"] = {"grading_images": len(idx.grading), "segmentation_images": len(idx.segmentation),
                          "od_centres": len(idx.od_centre), "fovea_centres": len(idx.fovea_centre),
                          "lesion_masks": dict(Counter(k for r in idx.segmentation for k in r.extra["masks"])),
                          "problems": idx.problems}
        records += idx.grading + idx.segmentation
    aptos_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.upper().startswith("APTOS")) \
        if data_dir.exists() else []
    if aptos_dirs:
        idx = find_aptos(aptos_dirs[0])
        found["APTOS2019"] = {"folder": aptos_dirs[0].name, "labelled_images": len(idx.records),
                              "images_without_label": idx.unlabeled_images, "problems": idx.problems}
        records += idx.records
    for name in ("DRIVE", "RITE", "IDRiD", "APTOS2019"):
        found.setdefault(name, {"status": "not found - see data/README.md"})
    return records, found


def _compare_aptos(previous: dict[str, str], rows: list[dict]) -> dict | None:
    """Did any APTOS photo change split since the previous manifest? (A trained triage model needs the same split.)"""
    if not previous:
        return None
    new = {r["image_id"]: r["split"] for r in rows if r["dataset"] == "APTOS2019"}
    common = set(previous) & set(new)
    return {"compared": len(common), "changed_split": sum(previous[k] != new[k] for k in common),
            "only_in_previous": len(set(previous) - set(new)), "only_in_new": len(set(new) - set(previous))}


def _rel(p, base: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(p)


def run(data_dir: Path, out_dir: Path, seed: int = 0, threshold: float = DEFAULT_THRESHOLD,
        keep_conflicts: bool = False) -> dict:
    t0 = time.perf_counter()
    records, found = discover(data_dir)
    if not records:
        raise FileNotFoundError(f"no supported dataset found under {data_dir} (see data/README.md)")

    cache = SignatureCache(out_dir / "signature_cache")
    ok_recs, vecs, corrupt, sizes = [], [], [], {}
    for r in records:
        chk, vec = cache.get(r.path)
        if not chk.ok:
            corrupt.append({"dataset": r.dataset, "image_id": r.image_id, "error": chk.error})
            continue
        ok_recs.append(r)
        vecs.append(vec.astype(np.float32))
        sizes.setdefault(r.dataset, []).append((chk.width, chk.height))
    cache.save()

    pairs = near_duplicate_pairs(np.array(vecs), REVIEW_THRESHOLD)
    dup_pairs = [p for p in pairs if p[2] >= threshold]
    groups = group_duplicates(len(ok_recs), dup_pairs)
    rows, info = assign_splits(ok_recs, groups, seed=seed, keep_conflicts=keep_conflicts)
    leaks = check_no_leakage(rows)
    if leaks:
        raise RuntimeError(f"split leakage detected: {leaks[:3]}")

    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    if (split_dir / "manifest.csv").exists():  # keep the old split, and report whether APTOS moved
        with open(split_dir / "manifest.csv", newline="") as f:
            previous = {r["image_id"]: r["split"] for r in csv.DictReader(f) if r["dataset"] == "APTOS2019"}
        shutil.copy2(split_dir / "manifest.csv", split_dir / "manifest_previous.csv")
    with open(split_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            extra = {k: (_rel(v, data_dir) if isinstance(v, str) and v else v) for k, v in r["_extra"].items()}
            if "masks" in extra:
                extra["masks"] = {k: _rel(v, data_dir) for k, v in r["_extra"]["masks"].items()}
            w.writerow({**{k: r[k] for k in COLUMNS if k in r}, "rel_path": _rel(r["_path"], data_dir),
                        "extra": json.dumps(extra)})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "duplicate_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "similarity", "a_dataset", "a_part", "a_id", "a_grade", "b_dataset", "b_part", "b_id",
                    "b_grade", "a_rel_path", "b_rel_path"])
        for i, j, s in sorted(pairs, key=lambda p: -p[2]):
            a, b = ok_recs[i], ok_recs[j]
            w.writerow(["duplicate" if s >= threshold else "review", f"{s:.4f}", a.dataset, a.part, a.image_id,
                        a.dr_grade, b.dataset, b.part, b.image_id, b.dr_grade, _rel(a.path, data_dir),
                        _rel(b.path, data_dir)])

    def kind(i, j):
        a, b = ok_recs[i], ok_recs[j]
        return "same dataset" if (a.dataset, a.part) == (b.dataset, b.part) else f"{a.dataset}/{a.part} vs {b.dataset}/{b.part}"

    size_stats = {ds: {"width_min_median_max": [int(np.min(v[:, 0])), int(np.median(v[:, 0])), int(np.max(v[:, 0]))],
                       "height_min_median_max": [int(np.min(v[:, 1])), int(np.median(v[:, 1])), int(np.max(v[:, 1]))]}
                  for ds, v in ((k, np.array(x)) for k, x in sizes.items())}
    report = {
        "data_dir": str(data_dir), "seed": seed, "duplicate_threshold": threshold,
        "review_threshold": REVIEW_THRESHOLD, "datasets": found, "images_checked": len(records),
        "corrupt_images": corrupt, "image_sizes": size_stats,
        "duplicate_pairs": len(dup_pairs), "duplicate_pairs_by_kind": dict(Counter(kind(i, j) for i, j, _ in dup_pairs)),
        "review_pairs": len(pairs) - len(dup_pairs), "grouping": info,
        "manifest_rows": len(rows), "manifest_fingerprint": manifest_fingerprint(rows),
        "splits": split_summary(rows), "leakage_check": "passed", "seconds": round(time.perf_counter() - t0, 1),
        "aptos_split_vs_previous_manifest": _compare_aptos(previous, rows),
    }
    (out_dir / "integrity_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(root / "data"))
    ap.add_argument("--out", default=str(root / "reports" / "m4_data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--keep-conflicts", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="first write FAKE datasets into --data-dir (software testing only)")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    if a.synthetic:
        from drpinn.datasets.synthetic import make_fake_datasets
        make_fake_datasets(data_dir)
        print(f"SYNTHETIC datasets written to {data_dir} — software test only, not results")
    r = run(data_dir, Path(a.out), a.seed, a.threshold, a.keep_conflicts)
    print(f"checked {r['images_checked']} images in {r['seconds']} s")
    for name, info in r["datasets"].items():
        print(f"  {name:<10} {json.dumps({k: v for k, v in info.items() if k != 'problems'})}")
        for p in info.get("problems", [])[:5]:
            print(f"             problem: {p}")
    print(f"  corrupt images: {len(r['corrupt_images'])}")
    print(f"  duplicate pairs (>= {r['duplicate_threshold']}): {r['duplicate_pairs']} {r['duplicate_pairs_by_kind']}")
    print(f"  pairs to inspect by eye ({r['review_threshold']}-{r['duplicate_threshold']}): {r['review_pairs']}")
    print(f"  grouping: {r['grouping']}")
    print(f"  manifest: {r['manifest_rows']} rows, fingerprint {r['manifest_fingerprint']}, leakage check {r['leakage_check']}")
    c = r.get("aptos_split_vs_previous_manifest")
    if c:
        print(f"  APTOS vs previous manifest: {c['changed_split']} of {c['compared']} photos changed split"
              + (" (a triage model trained on the old split must be retrained)" if c["changed_split"] else " (unchanged)"))
    for key, splits in r["splits"].items():
        print(f"  {key:<24} " + "  ".join(f"{s}: {sum(c.values())}" for s, c in splits.items()))


if __name__ == "__main__":
    main()
