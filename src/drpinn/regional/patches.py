"""Regional (patch) aggregation of segment scores, lesion labels and baseline features.

The image is divided into square patches. A patch is analysed when its centre
is inside the field of view and at least one vessel segment passes through it
(stress scores are only defined where there are vessels). Every score is
oriented so that HIGHER means "predicted more likely to contain a lesion".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..vessel_graph.extract import VesselGraph
from ..vessel_graph.validation import densify


@dataclass(frozen=True)
class EyeGeometry:
    disc_rc: tuple[float, float]
    disc_diameter_px: float
    fovea_rc: tuple[float, float]
    fovea_estimated: bool


def estimate_fovea(disc_rc, disc_diameter_px: float, fov: NDArray[np.bool_]) -> tuple[float, float]:
    """ASSUMPTION: fovea ~2.5 disc diameters from the disc centre, towards the field-of-view centre, same row.

    IDRiD's segmentation images have optic-disc masks but no fovea position;
    this rough anatomical estimate is only used for the distance-to-fovea baseline.
    """
    rows, cols = np.nonzero(fov)
    centre_c = cols.mean() if cols.size else fov.shape[1] / 2
    direction = 1.0 if centre_c >= disc_rc[1] else -1.0
    return float(disc_rc[0]), float(disc_rc[1] + direction * 2.5 * disc_diameter_px)


def segment_patches(graph: VesselGraph | None, patch: int, shape) -> list[NDArray[np.int64]]:
    """Flat patch indices each segment passes through."""
    if graph is None:
        return []
    n_c = int(np.ceil(shape[1] / patch))
    out = []
    for p in graph.paths_rc:
        pts = densify(p, 1.0)
        r = np.clip(pts[:, 0].astype(int) // patch, 0, int(np.ceil(shape[0] / patch)) - 1)
        c = np.clip(pts[:, 1].astype(int) // patch, 0, n_c - 1)
        out.append(np.unique(r * n_c + c))
    return out


def _patch_max(n_patches: int, seg_patches: list, seg_scores: NDArray) -> NDArray:
    out = np.full(n_patches, -np.inf)
    for idx, s in zip(seg_patches, seg_scores):
        if np.isfinite(s):
            out[idx] = np.maximum(out[idx], s)
    return out


def _pool(mask: NDArray, patch: int, how: str = "any") -> NDArray:
    h, w = mask.shape
    ph, pw = int(np.ceil(h / patch)), int(np.ceil(w / patch))
    pad = np.zeros((ph * patch, pw * patch), mask.dtype)
    pad[:h, :w] = mask
    blocks = pad.reshape(ph, patch, pw, patch)
    return (blocks.any(axis=(1, 3)) if how == "any" else blocks.mean(axis=(1, 3))).ravel()


def patch_table(image_id: str, shape, fov, patch: int, eye: EyeGeometry, trees: dict, vessel_mask,
                lesion_masks: dict[str, NDArray] | None = None, geometry: dict | None = None) -> list[dict]:
    """One row per analysed patch.

    trees: {"artery": (graph, TreeRisk), "vein": (graph, TreeRisk)} - either may be missing.
    geometry: optional per-tree dict of per-segment geometry features (tortuosity, radius_um, murray_deviation).
    """
    h, w = shape
    ph, pw = int(np.ceil(h / patch)), int(np.ceil(w / patch))
    n = ph * pw
    rr, cc = np.divmod(np.arange(n), pw)
    cy, cx = (rr + 0.5) * patch, (cc + 0.5) * patch
    inside = np.zeros(n, bool)
    ok = (cy < h) & (cx < w)
    inside[ok] = np.asarray(fov, bool)[cy[ok].astype(int), cx[ok].astype(int)]

    scores = {k: np.full(n, -np.inf) for k in ("high_shear_mc", "low_shear_mc", "high_shear_deterministic",
                                               "low_shear_deterministic", "tortuosity", "thinness",
                                               "murray_deviation")}
    has_vessel = np.zeros(n, bool)
    n_segments = np.zeros(n, int)
    for tree, (graph, risk) in trees.items():
        if graph is None or risk is None:
            continue
        sp = segment_patches(graph, patch, shape)
        solved = risk.solved
        for idx, ok_seg in zip(sp, solved):
            if ok_seg:
                has_vessel[idx] = True
                n_segments[idx] += 1
        upd = {"high_shear_mc": risk.p_high, "low_shear_mc": risk.p_low,
               "high_shear_deterministic": risk.percentile_deterministic,
               "low_shear_deterministic": 1 - risk.percentile_deterministic}
        g = (geometry or {}).get(tree)
        if g is not None:
            upd.update({"tortuosity": g["tortuosity"], "thinness": -g["radius_um"],
                        "murray_deviation": np.abs(g["murray_deviation"])})
        for k, v in upd.items():
            v = np.where(solved, v, np.nan)
            scores[k] = np.maximum(scores[k], _patch_max(n, sp, v))

    density = _pool(np.asarray(vessel_mask, float), patch, "mean")
    dd = eye.disc_diameter_px
    d_fovea = np.hypot(cy - eye.fovea_rc[0], cx - eye.fovea_rc[1]) / dd
    d_disc = np.hypot(cy - eye.disc_rc[0], cx - eye.disc_rc[1]) / dd
    labels = {}
    if lesion_masks:
        pooled = {k: _pool(np.asarray(m, bool), patch, "any") for k, m in lesion_masks.items()}
        z = np.zeros(n, bool)
        labels = {"red_lesion": pooled.get("MA", z) | pooled.get("HE", z),
                  "exudate": pooled.get("EX", z) | pooled.get("SE", z)}
        labels["any_lesion"] = labels["red_lesion"] | labels["exudate"]
        for k, v in pooled.items():
            labels[k] = v
    sel = np.flatnonzero(inside & has_vessel)
    rows = []
    for i in sel:
        row = {"image_id": image_id, "patch": int(i), "row": int(rr[i]), "col": int(cc[i]),
               "segments": int(n_segments[i]), "vessel_density": float(density[i]),
               "near_fovea": float(-d_fovea[i]), "near_disc": float(-d_disc[i]),
               "dist_fovea_dd": float(d_fovea[i]), "dist_disc_dd": float(d_disc[i])}
        for k, v in scores.items():
            row[k] = float(v[i]) if np.isfinite(v[i]) else float("nan")
        for k, v in labels.items():
            row[k] = int(v[i])
        rows.append(row)
    return rows
