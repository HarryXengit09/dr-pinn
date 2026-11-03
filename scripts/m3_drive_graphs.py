"""M3 on real data: extract vessel graphs from DRIVE manual annotations.

Inputs are the expert vessel masks (ground truth), so this measures the graph
step alone, not segmentation. Outputs per image: graph statistics (CSV), a
summary (JSON) and overlay figures. No physics is run here: real graphs still
mix arteries and veins (see docs/PLAN.md, assumption A3).

    python scripts/m3_drive_graphs.py --drive-dir data/DRIVE --overlays 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn.datasets import find_drive_samples, load_binary, load_rgb  # noqa: E402
from drpinn.vessel_graph import extract_vessel_graph  # noqa: E402


def overlay(sample, mask, g, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [2.2, 1]})
    if sample.image_path is not None:
        ax.imshow(load_rgb(sample.image_path))
    else:
        ax.imshow(mask, cmap="gray")
    segs = [p[:, ::-1] for p in g.paths_rc]
    lc = LineCollection(segs, array=g.radius_px, cmap="viridis", linewidths=1.4)
    ax.add_collection(lc)
    deg = g.degree()
    for sel, style, label in (
        ((deg >= 3) & ~g.node_possible_crossing, dict(marker="o", ms=3, color="orange", ls=""), "junction"),
        (g.node_possible_crossing, dict(marker="o", ms=7, mfc="none", mec="red", ls=""), "possible crossing"),
        (g.node_truncated, dict(marker="x", ms=4, color="white", ls=""), "cut by FOV border"),
    ):
        if sel.any():
            ax.plot(g.node_rc[sel, 1], g.node_rc[sel, 0], label=label, **style)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    fig.colorbar(lc, ax=ax, shrink=0.7, label="radius (px)")
    ax.set_title(f"DRIVE {sample.image_id}: {g.n_edges} segments, "
                 f"{g.stats['crossings_resolved']} crossings split", fontsize=10)
    axh.hist(g.radius_px[g.radius_reliable], bins=30, color="tab:blue", alpha=0.8, label="area method")
    axh.hist(g.radius_edt_px[g.radius_reliable], bins=30, histtype="step", color="k", label="distance transform")
    axh.set_xlabel("segment radius (px)")
    axh.set_ylabel("segments")
    axh.legend(fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def run(drive_dir: Path, out_dir: Path, n_overlays: int = 3, max_images: int | None = None) -> dict:
    samples = find_drive_samples(drive_dir)
    if not samples:
        raise FileNotFoundError(f"no '*manual1*' vessel annotations found under {drive_dir}")
    samples = samples[:max_images] if max_images else samples
    rows, failures = [], []
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        try:
            mask = load_binary(s.manual_path)
            fov = load_binary(s.fov_path) if s.fov_path is not None else None
            g = extract_vessel_graph(mask, fov=fov)
        except Exception as exc:  # report and continue; never silently skip
            failures.append({"image_id": s.image_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        n_comp = g.to_network(1.0).connected_components()[0]
        r = g.radius_px[g.radius_reliable]
        rows.append({
            "image_id": s.image_id, "height": mask.shape[0], "width": mask.shape[1],
            "fov_found": s.fov_path is not None, "vessel_fraction": float(mask.mean()),
            "segments": g.n_edges, "junctions": g.stats["junctions"], "tips": g.stats["tips"],
            "truncated_tips": g.stats["truncated_tips"], "crossings_resolved": g.stats["crossings_resolved"],
            "possible_crossings": g.stats["possible_crossings"], "components": int(n_comp),
            "fragments_removed": g.stats["components_removed"], "unreliable_width": g.stats["unreliable_width"],
            "radius_px_p10": float(np.percentile(r, 10)) if r.size else float("nan"),
            "radius_px_median": float(np.median(r)) if r.size else float("nan"),
            "radius_px_p90": float(np.percentile(r, 90)) if r.size else float("nan"),
            "edt_minus_area_median_px": float(np.median(g.radius_edt_px[g.radius_reliable] - r)) if r.size else float("nan"),
            "total_length_px": float(g.length_px.sum()), "seconds": time.perf_counter() - t0,
        })
        if i < n_overlays:
            overlay(s, mask, g, out_dir / "overlays" / f"drive_{s.image_id}_graph.png")

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / "drive_graph_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    keys = ["segments", "junctions", "tips", "truncated_tips", "crossings_resolved",
            "possible_crossings", "components", "radius_px_median", "seconds"]
    summary = {
        "note": "graphs from DRIVE manual (expert) masks; tests graph extraction only",
        "images": len(rows), "failures": failures,
        "median": {k: float(np.median([r[k] for r in rows])) for k in keys} if rows else {},
    }
    (out_dir / "drive_graph_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-dir", default=str(root / "data" / "DRIVE"))
    ap.add_argument("--out", default=str(root / "reports" / "m3_drive"))
    ap.add_argument("--overlays", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=None)
    a = ap.parse_args()
    s = run(Path(a.drive_dir), Path(a.out), a.overlays, a.max_images)
    print(f"DRIVE manual masks: {s['images']} graphs extracted, {len(s['failures'])} failures")
    for k, v in s["median"].items():
        print(f"  median {k:<20} {v:.2f}")
    for f in s["failures"]:
        print("  FAILED", f)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
