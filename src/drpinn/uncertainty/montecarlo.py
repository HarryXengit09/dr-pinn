"""Monte Carlo propagation of the unknowns through the vessel-network physics (M7).

For one vessel graph (an arterial or a venous tree from M5) and the optic-disc
position, every sample draws the quantities a fundus photo cannot measure -
the ones Milestone 2 showed actually reorder wall-shear-stress (WSS) rankings:

  * outlet model: terminal resistance of k unseen Murray generations, k ~ U(levels)
  * per-outlet reference pressure jitter (heterogeneous capillary beds)
  * per-segment radius error, lognormal with sigma = measured segmentation error
  * viscosity model: Newtonian or Fahraeus-Lindqvist (Pries 1992) with Hd ~ U(range)
  * physical pixel size, from an ASSUMED optic-disc diameter ~ U(range)

Boundary conditions (stated assumptions, not measurements): in every connected
piece of the graph the end nearest the optic-disc centre is the disc end; all
other ends drain to (arteries) or are fed from (veins) the unseen capillary bed.
Absolute pressures are illustrative; only within-tree rankings are reported.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..hemodynamics import (BoundaryConditions, NewtonianViscosity, PriesInVitroViscosity, VesselNetwork,
                            murray_terminal_resistance, solve_with_viscosity_model)
from ..hemodynamics.units import mmhg_to_pa
from ..vessel_graph.extract import VesselGraph
from .ranking import percentile_rank, spearman, top_fraction_mask


@dataclass(frozen=True)
class MCConfig:
    n_samples: int = 50
    radius_sigma: float = 0.10                 # per-segment relative radius error (set from M5 CV)
    levels: tuple[float, float] = (2.0, 10.0)  # unseen Murray generations behind each end
    length_to_radius: float = 20.0
    outlet_jitter_mmhg: float = 3.0
    p_pries: float = 0.5                       # probability a sample uses Fahraeus-Lindqvist
    hd_range: tuple[float, float] = (0.30, 0.50)
    disc_um_range: tuple[float, float] = (1500.0, 2100.0)  # ASSUMED optic-disc diameter
    top_fraction: float = 0.10
    seed: int = 0
    # illustrative pressures (mmHg); rankings depend on them only through the jitter
    artery_disc_mmhg: float = 40.0
    artery_bed_mmhg: float = 25.0
    vein_disc_mmhg: float = 15.0
    vein_bed_mmhg: float = 25.0


@dataclass
class TreeRisk:
    """Per-segment results for one tree; NaN for segments in pieces without an outflow end."""
    tree: str
    wss_deterministic: NDArray[np.float64]        # Pa, Newtonian, no noise, k = mid-range
    percentile_deterministic: NDArray[np.float64]
    p_high: NDArray[np.float64]                   # P(segment in the tree's top 10 % WSS)
    p_low: NDArray[np.float64]                    # P(segment in the tree's bottom 10 % WSS)
    median_percentile: NDArray[np.float64]
    rank_stability: float                         # mean Spearman, samples vs deterministic
    disc_nodes: list[int] = field(default_factory=list)
    solved: NDArray[np.bool_] | None = None


def disc_ends(graph: VesselGraph, disc_rc: tuple[float, float]) -> tuple[list[int], list[int], NDArray]:
    """Per connected piece: the node nearest the disc centre (prefer vessel ends) and the other ends."""
    net = graph.to_network(1.0)
    _, comp = net.connected_components()
    deg = graph.degree()
    dist = np.linalg.norm(graph.node_rc - np.asarray(disc_rc, float), axis=1)
    inlets, outlets = [], []
    for c in np.unique(comp):
        nodes = np.flatnonzero(comp == c)
        tips = nodes[deg[nodes] == 1]
        pool = tips if tips.size else nodes
        inlet = int(pool[np.argmin(dist[pool])])
        others = [int(t) for t in tips if t != inlet]
        if others:  # a piece with no other end has no outflow and is skipped
            inlets.append(inlet)
            outlets.extend(others)
    return inlets, outlets, comp


def _solve(net: VesselNetwork, inlets, outlets, tree: str, cfg: MCConfig, k: float, jitter: NDArray,
           model) -> NDArray:
    if tree == "artery":
        p_disc, p_bed = cfg.artery_disc_mmhg, cfg.artery_bed_mmhg
    else:
        p_disc, p_bed = cfg.vein_disc_mmhg, cfg.vein_bed_mmhg
    mu_ref = 3.5e-3
    robin = murray_terminal_resistance(net, outlets, mu_ref, k, cfg.length_to_radius, mmhg_to_pa(p_bed))
    robin = {o: (r, pref + mmhg_to_pa(float(j))) for (o, (r, pref)), j in zip(robin.items(), jitter)}
    bc = BoundaryConditions(pressure_pa={i: mmhg_to_pa(p_disc) for i in inlets}, terminal_resistance=robin)
    return solve_with_viscosity_model(net, bc, model).solution.wss_pa


def tree_risk(graph: VesselGraph | None, disc_rc: tuple[float, float], disc_diameter_px: float, tree: str,
              cfg: MCConfig = MCConfig()) -> TreeRisk | None:
    """Monte Carlo WSS rankings for one arterial or venous graph."""
    if graph is None or graph.n_edges == 0:
        return None
    if tree not in ("artery", "vein"):
        raise ValueError("tree must be 'artery' or 'vein'")
    if not (math.isfinite(disc_diameter_px) and disc_diameter_px > 0):
        raise ValueError("optic-disc diameter (px) must be > 0")
    inlets, outlets, comp = disc_ends(graph, disc_rc)
    n_e = graph.n_edges
    if not inlets:
        return None
    keep_comp = set(comp[inlets])
    solved = np.array([comp[u] in keep_comp for u, _ in graph.edges])
    # Pieces without an outflow end (closed loops, single-ended fragments) cannot carry flow under these
    # boundary conditions: they are left out of the solve and stay NaN.
    keep_nodes = np.flatnonzero(np.isin(comp, list(keep_comp)))
    new_id = np.full(graph.n_nodes, -1)
    new_id[keep_nodes] = np.arange(keep_nodes.size)
    sub_in, sub_out = [int(new_id[i]) for i in inlets], [int(new_id[o]) for o in outlets]

    def restrict(net: VesselNetwork) -> VesselNetwork:
        return VesselNetwork(keep_nodes.size, new_id[net.edges[solved]], net.length_m[solved], net.radius_m[solved],
                             net.node_xy[keep_nodes])

    rng = np.random.default_rng([cfg.seed, 0 if tree == "artery" else 1])
    base_net = restrict(graph.to_network(float(np.mean(cfg.disc_um_range)) / disc_diameter_px))
    k_mid = float(np.mean(cfg.levels))

    def ranked(wss):
        out = np.full(n_e, np.nan)
        out[solved] = wss
        return out

    det = ranked(_solve(base_net, sub_in, sub_out, tree, cfg, k_mid, np.zeros(len(outlets)), NewtonianViscosity()))
    pct_det = np.full(n_e, np.nan)
    pct_det[solved] = percentile_rank(det[solved])
    high = np.zeros(n_e)
    low = np.zeros(n_e)
    pcts, rhos = [], []
    for _ in range(cfg.n_samples):
        px_um = rng.uniform(*cfg.disc_um_range) / disc_diameter_px
        net = graph.to_network(px_um)
        noise = rng.lognormal(0.0, cfg.radius_sigma, n_e) if cfg.radius_sigma > 0 else np.ones(n_e)
        with warnings.catch_warnings():  # rare noisy radii can trip the units sanity check; units are metres here
            warnings.filterwarnings("ignore", message="segment radius > 1 mm", category=UserWarning)
            net = restrict(net.with_radius(np.maximum(net.radius_m * noise, 0.5e-6)))
        model = (PriesInVitroViscosity(discharge_hematocrit=rng.uniform(*cfg.hd_range))
                 if rng.random() < cfg.p_pries else NewtonianViscosity())
        jitter = rng.uniform(-cfg.outlet_jitter_mmhg, cfg.outlet_jitter_mmhg, len(outlets))
        w = ranked(_solve(net, sub_in, sub_out, tree, cfg, rng.uniform(*cfg.levels), jitter, model))
        v = w[solved]
        high[solved] += top_fraction_mask(v, cfg.top_fraction)
        low[solved] += top_fraction_mask(-v, cfg.top_fraction)
        pcts.append(percentile_rank(v))
        if v.size > 2:
            rhos.append(spearman(det[solved], v))
    n = max(cfg.n_samples, 1)
    p_high = np.where(solved, high / n, np.nan)
    p_low = np.where(solved, low / n, np.nan)
    med = np.full(n_e, np.nan)
    if pcts:
        med[solved] = np.median(np.stack(pcts), axis=0)
    finite = [r for r in rhos if np.isfinite(r)]
    return TreeRisk(tree, det, pct_det, p_high, p_low, med, float(np.mean(finite)) if finite else float("nan"),
                    inlets, solved)
