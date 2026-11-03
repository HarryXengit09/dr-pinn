"""M9: how often does the API's optic-disc estimate land on the expert-marked disc? (IDRiD, CPU, ~2 min)

For the 81 IDRiD segmentation photos (expert optic-disc masks), at the same
working scale as M7, the estimate (src/drpinn/api/disc.py) is compared with
the mask:
  hit            estimated centre inside the expert disc
  centre error   distance to the expert disc centre, in expert disc diameters
  size ratio     expert disc diameter / field-of-view diameter (checks the 0.14 assumption)
With --probs (the M7 probability cache, reports/m7_full/probs) the vessel map is
used as well, as in the API when the vessel model is present.

    python scripts/m9_check_disc.py --idrid-dir data/IDRiD --probs reports/m7_full/probs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpinn.api.disc import DISC_TO_FOV, estimate_disc  # noqa: E402
from drpinn.regional.pipeline import M7Config, disc_from_mask  # noqa: E402


def _m7():
    spec = importlib.util.spec_from_file_location("m7", ROOT / "scripts" / "m7_lesion_association.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(idrid_dir: Path, out_dir: Path, probs_dir: Path | None = None, eyes=None) -> dict:
    eyes = eyes if eyes is not None else _m7().idrid_eyes(idrid_dir, M7Config())
    rows = []
    for image_id, split, rgb, fov, od, _ in eyes:
        (tr, tc), td = disc_from_mask(od)
        fov_d = 2 * np.sqrt(fov.sum() / np.pi)
        row = {"image_id": image_id, "split": split, "true_disc_to_fov": td / fov_d}
        variants = {"brightness": None}
        p = probs_dir / f"{image_id}.npz" if probs_dir else None
        if p is not None and p.exists():
            with np.load(p) as z:
                if z["prob"].shape[1:] == fov.shape:
                    variants["brightness + vessels"] = z["prob"][0].astype(np.float32)
        for name, prob in variants.items():
            (r, c), _ = estimate_disc(rgb, fov, prob)
            row[f"{name}: hit"] = bool(od[int(round(r)), int(round(c))])
            row[f"{name}: error_dd"] = float(np.hypot(r - tr, c - tc) / td)
        rows.append(row)
        print(f"  {image_id}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                         for k, v in row.items() if k not in ("image_id", "split")), flush=True)
    summary = {"images": len(rows), "assumed_disc_to_fov": DISC_TO_FOV,
               "true_disc_to_fov_median": float(np.median([r["true_disc_to_fov"] for r in rows])),
               "true_disc_to_fov_iqr": [float(np.percentile([r["true_disc_to_fov"] for r in rows], q)) for q in (25, 75)]}
    for name in ("brightness", "brightness + vessels"):
        hits = [r[f"{name}: hit"] for r in rows if f"{name}: hit" in r]
        if hits:
            err = [r[f"{name}: error_dd"] for r in rows if f"{name}: error_dd" in r]
            summary[name] = {"images": len(hits), "hit_rate": float(np.mean(hits)),
                             "median_error_disc_diameters": float(np.median(err)),
                             "worst_error_disc_diameters": float(np.max(err))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "disc_check.json").write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idrid-dir", default=str(ROOT / "data" / "IDRiD"))
    ap.add_argument("--probs", default=str(ROOT / "reports" / "m7_full" / "probs"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "m9_disc"))
    a = ap.parse_args()
    s = run(Path(a.idrid_dir), Path(a.out), Path(a.probs))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
