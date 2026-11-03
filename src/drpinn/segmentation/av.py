"""Separate arterial and venous vessel graphs (fixes assumption A3, docs/PLAN.md).

Arteries and veins cross; in a single vessel mask each crossing becomes a
false junction that joins the two trees. With multi-label artery/vein maps,
each tree is extracted on its own; crossing pixels belong to both maps, so
both trees stay continuous and neither sees the other's crossings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..vessel_graph.extract import ExtractionConfig, VesselGraph, extract_vessel_graph


@dataclass(frozen=True, eq=False)
class AVGraphs:
    combined: VesselGraph
    artery: VesselGraph | None
    vein: VesselGraph | None

    def summary(self) -> dict[str, float]:
        out = {}
        for name, g in (("combined", self.combined), ("artery", self.artery), ("vein", self.vein)):
            if g is None:
                continue
            out[f"{name}_segments"] = g.n_edges
            out[f"{name}_components"] = int(g.to_network(1.0).connected_components()[0])
            out[f"{name}_crossings_split"] = g.stats["crossings_resolved"]
            out[f"{name}_possible_crossings"] = g.stats["possible_crossings"]
        return out


def _graph(mask: NDArray[np.bool_], prob: NDArray | None, fov, config) -> VesselGraph | None:
    if mask.sum() < (config or ExtractionConfig()).min_component_area_px:
        return None
    soft = None
    if prob is not None:
        near = ndimage.binary_dilation(mask, iterations=2)
        soft = np.where(near, np.clip(prob, 0, 1), 0.0)
    try:
        return extract_vessel_graph(mask, fov=fov, probability=soft, config=config)
    except ValueError:  # nothing left after cleaning
        return None


def av_graphs(vessel_mask: NDArray[np.bool_], artery_mask: NDArray[np.bool_], vein_mask: NDArray[np.bool_],
              fov: NDArray[np.bool_] | None = None, vessel_prob: NDArray | None = None,
              config: ExtractionConfig | None = None) -> AVGraphs:
    """Combined, arterial and venous graphs. Masks must be boolean and the same size."""
    v = np.asarray(vessel_mask, bool)
    a = np.asarray(artery_mask, bool) & v
    w = np.asarray(vein_mask, bool) & v
    combined = extract_vessel_graph(v, fov=fov, probability=vessel_prob, config=config)
    return AVGraphs(combined, _graph(a, vessel_prob, fov, config), _graph(w, vessel_prob, fov, config))


def masks_from_probabilities(prob: NDArray, threshold: float = 0.5) -> tuple[NDArray, NDArray, NDArray]:
    """(vessel, artery, vein) boolean masks from a (3, H, W) probability map."""
    vessel = prob[0] >= threshold
    return vessel, vessel & (prob[1] >= threshold), vessel & (prob[2] >= threshold)
