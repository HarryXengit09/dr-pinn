"""M2 rank-sensitivity study: which uncertain inputs actually reorder WSS rankings?

SYNTHETIC trees and ILLUSTRATIVE perturbation sizes. This tells us which
uncertainties the real study (M7) must sample; the magnitudes must later be
replaced by measured segmentation error (M5) and literature ranges.

Baseline: Newtonian 3.5 mPa s, inlet 40 mmHg, every outlet 15 mmHg.
Metrics vs baseline, per sample: tie-aware Spearman of segment WSS, and the
share of the baseline top-10% segments still in the top 10%.

    python scripts/m2_rank_sensitivity.py --trees 20 --samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn.hemodynamics import (  # noqa: E402
    BoundaryConditions,
    CarreauYasudaViscosity,
    NewtonianViscosity,
    PriesInVitroViscosity,
    SyntheticNetwork,
    murray_terminal_resistance,
    random_tree,
    solve_network,
    solve_with_viscosity_model,
)
from drpinn.hemodynamics.units import mmhg_to_pa  # noqa: E402
from drpinn.uncertainty import spearman, top_set_overlap  # noqa: E402

MU = 3.5e-3
P_IN, P_OUT = mmhg_to_pa(40.0), mmhg_to_pa(15.0)
LAMBDA = 20.0  # nominal length/radius ratio of the synthetic trees


def _bc(syn: SyntheticNetwork, p_in: float = P_IN, p_out: float = P_OUT) -> BoundaryConditions:
    return BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, p_in, syn.outlet_nodes, p_out)


def _wss(syn, net=None, bc=None, mu=MU):
    return solve_network(net or syn.network, bc or _bc(syn), mu).wss_pa


def _with_radius(syn, r):
    return syn.network.with_radius(np.maximum(r, 1e-6))


def _outlet_hetero(syn, rng):
    p = {int(o): mmhg_to_pa(rng.uniform(10, 20)) for o in syn.outlet_nodes}
    return _wss(syn, bc=BoundaryConditions(pressure_pa={int(syn.inlet_nodes[0]): P_IN, **p}))


def _terminal_resistance(syn, rng):
    robin = murray_terminal_resistance(syn.network, syn.outlet_nodes, MU,
                                       rng.uniform(2, 10), LAMBDA, P_OUT)
    return _wss(syn, bc=BoundaryConditions(pressure_pa={int(syn.inlet_nodes[0]): P_IN},
                                           terminal_resistance=robin))


def _lognormal_radius(sigma):
    def f(syn, rng):
        r = syn.network.radius_m * rng.lognormal(0.0, sigma, syn.network.n_edges)
        return _wss(syn, net=_with_radius(syn, r))
    return f


def _length_noise(syn, rng):
    length = syn.network.length_m * rng.lognormal(0.0, 0.1, syn.network.n_edges)
    net = type(syn.network)(syn.network.n_nodes, syn.network.edges, length, syn.network.radius_m)
    return _wss(syn, net=net)


def _model(model):
    return lambda syn, net=None: solve_with_viscosity_model(net or syn.network, _bc(syn), model)


PERTURBATIONS: list[tuple[str, str, str, Callable, bool]] = [
    # (name, group, description, function(syn, rng) -> wss, deterministic)
    ("control_blood_pressure", "control (proved invariant)", "inlet 30–60 mmHg",
     lambda s, g: _wss(s, bc=_bc(s, mmhg_to_pa(g.uniform(30, 60)))), False),
    ("control_viscosity", "control (proved invariant)", "uniform 3.0–4.5 mPa s",
     lambda s, g: _wss(s, mu=g.uniform(3.0e-3, 4.5e-3)), False),
    ("control_image_scale", "control (proved invariant)", "pixel size x0.8–1.2",
     lambda s, g: _wss(s, net=s.network.scaled(g.uniform(0.8, 1.2))), False),
    ("control_radius_x_bias", "control (proved invariant)", "all radii x0.9–1.1",
     lambda s, g: _wss(s, net=_with_radius(s, s.network.radius_m * g.uniform(0.9, 1.1))), False),
    ("outlet_pressure_heterogeneity", "boundary conditions", "each outlet 10–20 mmHg",
     _outlet_hetero, False),
    ("terminal_resistance", "boundary conditions", "Robin, 2–10 unseen Murray levels",
     _terminal_resistance, False),
    ("radius_noise_5pct", "geometry error", "per-segment radius, lognormal 5%",
     _lognormal_radius(0.05), False),
    ("radius_noise_10pct", "geometry error", "per-segment radius, lognormal 10%",
     _lognormal_radius(0.10), False),
    ("radius_noise_20pct", "geometry error", "per-segment radius, lognormal 20%",
     _lognormal_radius(0.20), False),
    ("radius_additive_bias", "geometry error", "all radii +/- up to 3 um",
     lambda s, g: _wss(s, net=_with_radius(s, s.network.radius_m + g.uniform(-3e-6, 3e-6))), False),
    ("length_noise_10pct", "geometry error", "per-segment length, lognormal 10%",
     _length_noise, False),
    ("pries_viscosity", "rheology", "Fåhræus–Lindqvist, Hd 0.45",
     lambda s, g: _model(PriesInVitroViscosity())(s).solution.wss_pa, True),
    ("pries_uncertain_hd_and_scale", "rheology", "Fåhræus–Lindqvist, Hd 0.30–0.50, scale x0.8–1.2",
     lambda s, g: _model(PriesInVitroViscosity(discharge_hematocrit=g.uniform(0.3, 0.5)))(
         s, s.network.scaled(g.uniform(0.8, 1.2))).solution.wss_pa, False),
    ("carreau_yasuda", "rheology", "shear-thinning (Cho & Kensey)",
     lambda s, g: _model(CarreauYasudaViscosity())(s).solution.wss_pa, True),
]


def run(trees: int = 20, samples: int = 20, generations: int = 8, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    stats: dict[str, dict[str, list[float]]] = {p[0]: {"spearman": [], "top10": []} for p in PERTURBATIONS}
    n_segments = []
    for t in range(trees):
        syn = random_tree(generations, seed=seed + t)
        n_segments.append(syn.network.n_edges)
        base = solve_with_viscosity_model(syn.network, _bc(syn), NewtonianViscosity(MU)).solution.wss_pa
        for k, (name, _, _, fn, deterministic) in enumerate(PERTURBATIONS):
            rng = np.random.default_rng([seed, t, k])  # stable across runs and machines
            for _ in range(1 if deterministic else samples):
                w = fn(syn, rng)
                stats[name]["spearman"].append(spearman(base, w))
                stats[name]["top10"].append(top_set_overlap(base, w, 0.10))

    rows = []
    for name, group, desc, _, _ in PERTURBATIONS:
        sp, tp = np.array(stats[name]["spearman"]), np.array(stats[name]["top10"])
        rows.append({"perturbation": name, "group": group, "description": desc, "n": int(sp.size),
                     "spearman_mean": float(sp.mean()), "spearman_p05": float(np.percentile(sp, 5)),
                     "top10_overlap_mean": float(tp.mean()), "top10_overlap_p05": float(np.percentile(tp, 5))})
    return {
        "note": "SYNTHETIC trees, illustrative perturbation sizes; not patient data",
        "config": {"trees": trees, "samples": samples, "generations": generations, "seed": seed,
                   "segments_per_tree": int(np.median(n_segments)), "baseline":
                   "Newtonian 3.5 mPa s, inlet 40 mmHg, outlets 15 mmHg"},
        "runtime_s": time.perf_counter() - t0,
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=20)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                          / "reports/experiments/m2_rank_sensitivity.json"))
    a = ap.parse_args()
    res = run(a.trees, a.samples, a.generations, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print(f"SYNTHETIC — {a.trees} trees x ~{res['config']['segments_per_tree']} segments, "
          f"{a.samples} samples each ({res['runtime_s']:.1f} s)\n")
    print(f"{'perturbation':<31}{'Spearman':>10}{'p05':>8}{'top-10%':>9}{'p05':>7}")
    group = None
    for r in res["results"]:
        if r["group"] != group:
            group = r["group"]
            print(f"-- {group}")
        print(f"{r['perturbation']:<31}{r['spearman_mean']:>10.4f}{r['spearman_p05']:>8.4f}"
              f"{r['top10_overlap_mean']:>9.2f}{r['top10_overlap_p05']:>7.2f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
