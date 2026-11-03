"""Measure classical 1D solver latency on THIS machine (Experiment 7, part 1).

SYNTHETIC networks, illustrative boundary pressures. Results describe software
speed only. Re-run on your own hardware before quoting any number.

    python scripts/benchmark_solver.py --repeats 20 --mc-samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn.hemodynamics import BoundaryConditions, random_tree, solve_network  # noqa: E402
from drpinn.hemodynamics.units import mmhg_to_pa  # noqa: E402

MU = 3.5e-3  # Pa s, illustrative
P_IN, P_OUT = mmhg_to_pa(40.0), mmhg_to_pa(15.0)  # illustrative, not calibrated


def time_solve(syn, repeats: int) -> list[float]:
    bc = BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, P_IN, syn.outlet_nodes, P_OUT)
    solve_network(syn.network, bc, MU)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sol = solve_network(syn.network, bc, MU)
        times.append(time.perf_counter() - t0)
    assert sol.max_rel_conservation_error < 1e-9
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--mc-samples", type=int, default=1000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                          / "reports/benchmarks/solver_latency.json"))
    args = ap.parse_args()

    rows = []
    for gens in range(4, 15):
        syn = random_tree(gens, seed=gens, n_loops=max(1, 2**gens // 50))
        t = np.array(time_solve(syn, args.repeats)) * 1e3
        rows.append({"segments": syn.network.n_edges, "nodes": syn.network.n_nodes,
                     "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))})

    # Monte Carlo with per-segment radius noise (the kind of uncertainty that
    # actually changes rankings), on a ~500-segment network.
    syn = random_tree(9, seed=9, n_loops=10)
    bc = BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, P_IN, syn.outlet_nodes, P_OUT)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(args.mc_samples):
        r = syn.network.radius_m * rng.lognormal(0.0, 0.1, syn.network.n_edges)
        solve_network(syn.network.with_radius(r), bc, MU)
    mc_total = time.perf_counter() - t0

    result = {
        "note": "synthetic networks; measured on the machine listed below only",
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
        },
        "single_solve": rows,
        "monte_carlo": {"segments": syn.network.n_edges, "samples": args.mc_samples,
                        "total_s": mc_total, "per_sample_ms": 1e3 * mc_total / args.mc_samples},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"{'segments':>9} {'median ms':>10} {'p95 ms':>8}")
    for r in rows:
        print(f"{r['segments']:>9} {r['median_ms']:>10.2f} {r['p95_ms']:>8.2f}")
    mc = result["monte_carlo"]
    print(f"Monte Carlo: {mc['samples']} samples x {mc['segments']} segments = "
          f"{mc['total_s']:.2f} s ({mc['per_sample_ms']:.2f} ms/sample)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
