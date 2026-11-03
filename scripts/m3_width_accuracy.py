"""M3 width-accuracy study: how well is vessel radius recovered from a mask?

SYNTHETIC straight tubes of known radius at many angles and sub-pixel offsets.
Compares the distance-transform estimate (median EDT - 0.5 px, the PDF's
approach) with the area-based estimate on binary masks and on soft
(probability-like) masks. Answers the M2 target: radius error <= ~5 %.

    python scripts/m3_width_accuracy.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn.vessel_graph.extract import extract_vessel_graph  # noqa: E402
from drpinn.vessel_graph.synthetic_masks import straight_tube  # noqa: E402


def run(radii=np.arange(1.0, 8.01, 0.5), angles=np.arange(0, 90, 5), offsets: int = 3,
        seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    rows = []
    for r in radii:
        errs = {"edt_binary": [], "area_binary": [], "area_soft": []}
        for ang in angles:
            for _ in range(offsets):
                img = straight_tube(shape=(140, 200), length=140.0, radius=float(r), angle_deg=float(ang),
                                    offset=tuple(rng.uniform(-0.5, 0.5, 2)))
                gb = extract_vessel_graph(img.binary)
                gs = extract_vessel_graph(img.binary, probability=img.soft)
                if gb.n_edges != 1 or gs.n_edges != 1:
                    continue  # topology failure is counted separately below
                errs["edt_binary"].append(gb.radius_edt_px[0] / r - 1)
                errs["area_binary"].append(gb.radius_px[0] / r - 1)
                errs["area_soft"].append(gs.radius_px[0] / r - 1)
        n_ok = len(errs["area_soft"])
        row = {"radius_px": float(r), "n": n_ok, "n_topology_failures": len(angles) * offsets - n_ok}
        for k, v in errs.items():
            v = np.array(v)
            row[f"{k}_median_abs"] = float(np.median(np.abs(v))) if v.size else float("nan")
            row[f"{k}_bias"] = float(np.mean(v)) if v.size else float("nan")
        rows.append(row)
    return {"note": "SYNTHETIC straight tubes; relative radius error", "runtime_s": time.perf_counter() - t0,
            "results": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                          / "reports/experiments/m3_width_accuracy.json"))
    a = ap.parse_args()
    res = run()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"SYNTHETIC straight tubes — median |radius error| (bias in brackets), {res['runtime_s']:.0f} s\n")
    print(f"{'r (px)':>7} {'EDT binary':>17} {'area binary':>17} {'area soft':>17} {'fail':>5}")
    for r in res["results"]:
        cells = [f"{100 * r[f'{k}_median_abs']:5.1f}% ({100 * r[f'{k}_bias']:+5.1f})"
                 for k in ("edt_binary", "area_binary", "area_soft")]
        print(f"{r['radius_px']:>7.1f} {cells[0]:>17} {cells[1]:>17} {cells[2]:>17} {r['n_topology_failures']:>5}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
