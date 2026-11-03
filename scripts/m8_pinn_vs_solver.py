"""M8: does a PINN add value over the classical 1D solver? (PDF §9.4, plan assumption A11)

Same networks, same boundary conditions, same viscosity for every method.
Reference: the direct sparse solve (hemodynamics.solver).
Methods:
  Jacobi(T), CG(T)    classical iterative solvers with T steps
  PI-GNN(T)           physics-informed GNN, trained ONCE on random synthetic
                      trees with the physics residual only (no solver labels),
                      then applied to every test network in one forward pass
  nodal PINN          per network: pressures as free parameters, L-BFGS on the physics loss
  coordinate PINN     per network: classic PINN, MLP of node (x, y) with Fourier features;
                      Fourier scale picked on separate TUNING networks (never the test set)
Metrics per network: max pressure error (fraction of the inlet-outlet drop),
median relative WSS error, WSS rank agreement (tie-aware Spearman, top-10 %
overlap), worst mass-conservation error, final physics loss, wall time.
Networks: random synthetic trees (6-9 generations, loops, two outlet models)
and, if available, graphs from DRIVE expert masks (largest connected piece;
inlet = widest end, Dirichlet 40/15 mmHg, pixel size 10 um - illustrative;
numerical comparison only, not physiology).

    python scripts/m8_pinn_vs_solver.py --drive-dir data/DRIVE
    python scripts/m8_pinn_vs_solver.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from drpinn.hemodynamics import (  # noqa: E402
    BoundaryConditions, VesselNetwork, evaluate_pressure_field, murray_terminal_resistance, random_tree, solve_network,
)
from drpinn.hemodynamics.units import mmhg_to_pa  # noqa: E402
from drpinn.pinn.gnn import gnn_solve, train_gnn  # noqa: E402
from drpinn.pinn.graph_physics import GraphProblem  # noqa: E402
from drpinn.pinn.methods import conjugate_gradient, coordinate_pinn, jacobi, nodal_pinn  # noqa: E402
from drpinn.uncertainty import spearman, top_set_overlap  # noqa: E402

MU = 3.5e-3
P_IN, P_OUT = mmhg_to_pa(40.0), mmhg_to_pa(15.0)
TEST_SEED_BASE = 2_000_000_000  # training draws seeds < 1e9, so test networks are never seen in training


def synthetic_problem(rng: np.random.Generator, generations: int | None = None, seed: int | None = None) -> GraphProblem:
    syn = random_tree(int(generations or rng.integers(5, 9)), seed=int(seed if seed is not None else rng.integers(10**9)),
                      n_loops=int(rng.integers(0, 6)), radius_jitter=float(rng.uniform(0.05, 0.25)))
    net = syn.network
    tips = [int(o) for o in syn.outlet_nodes if net.degree()[o] == 1]
    if rng.random() < 0.5:  # outlets at slightly different pressures
        bc = BoundaryConditions(pressure_pa={int(syn.inlet_nodes[0]): P_IN,
                                             **{t: P_OUT + float(rng.uniform(-300, 300)) for t in tips}})
    else:                   # outlets drain through unseen Murray levels
        bc = BoundaryConditions(pressure_pa={int(syn.inlet_nodes[0]): P_IN},
                                terminal_resistance=murray_terminal_resistance(net, tips, MU, float(rng.uniform(2, 10)), 20, P_OUT))
    return GraphProblem.build(net, bc, MU)


def drive_problems(drive_dir: Path, max_graphs: int, pixel_um: float = 10.0) -> list[tuple[str, GraphProblem]]:
    from drpinn.datasets import find_drive_samples, load_binary
    from drpinn.vessel_graph import extract_vessel_graph

    out = []
    for s in find_drive_samples(drive_dir)[:max_graphs]:
        g = extract_vessel_graph(load_binary(s.manual_path), fov=load_binary(s.fov_path) if s.fov_path else None)
        net = g.to_network(pixel_um)
        _, comp = net.connected_components()
        keep_nodes = np.flatnonzero(comp == np.bincount(comp).argmax())
        remap = -np.ones(net.n_nodes, int)
        remap[keep_nodes] = np.arange(keep_nodes.size)
        ek = np.flatnonzero(np.isin(net.edges[:, 0], keep_nodes))
        sub = VesselNetwork(keep_nodes.size, remap[net.edges[ek]], net.length_m[ek], net.radius_m[ek],
                            net.node_xy[keep_nodes])
        tips = [int(t) for t in sub.terminal_nodes()]
        if len(tips) < 2:
            continue
        r_tip = {t: sub.radius_m[np.flatnonzero((sub.edges == t).any(axis=1))[0]] for t in tips}
        inlet = max(tips, key=r_tip.get)
        bc = BoundaryConditions.inlet_outlet_pressure([inlet], P_IN, [t for t in tips if t != inlet], P_OUT)
        out.append((f"DRIVE {s.image_id}", GraphProblem.build(sub, bc, MU)))
    return out


def score(problem: GraphProblem, ref_sol, p_scaled: np.ndarray) -> dict:
    sol = evaluate_pressure_field(problem.network, problem.bc, problem.viscosity, problem.to_pa(p_scaled))
    ref_wss, wss = ref_sol.wss_pa, sol.wss_pa
    ref_p = (ref_sol.pressure_pa - problem.p_low) / problem.p_span
    rel = np.abs(wss - ref_wss) / np.maximum(ref_wss, 1e-12 * ref_wss.max())
    return {"pressure_max_error": float(np.abs(p_scaled - ref_p).max()),
            "wss_median_rel_error": float(np.median(rel)),
            "wss_spearman": spearman(ref_wss, wss), "wss_top10_overlap": top_set_overlap(ref_wss, wss, 0.10),
            "conservation_error": float(sol.max_rel_conservation_error)}


def tune_sigma(tuning: list[GraphProblem], sigmas, adam: int, lbfgs: int) -> tuple[float, dict]:
    table = {}
    for sigma in sigmas:
        errs = []
        for pr in tuning:
            ref = solve_network(pr.network, pr.bc, pr.viscosity)
            r = coordinate_pinn(pr, adam_steps=adam, lbfgs_steps=lbfgs, sigma=sigma)
            errs.append(score(pr, ref, r.p_scaled)["pressure_max_error"])
        table[str(sigma)] = float(np.median(errs))
        print(f"  tuning coordinate PINN: Fourier scale {sigma}: median pressure error {table[str(sigma)]:.2e}", flush=True)
    best = float(min(sigmas, key=lambda s: table[str(s)]))
    return best, table


def run(out_dir: Path, n_test: int = 20, drive_dir: Path | None = None, max_drive: int = 10, steps: int = 30,
        gnn_iters: int = 600, pinn_adam: int = 3000, pinn_lbfgs: int = 500, nodal_lbfgs: int = 2000,
        sigmas=(3.0, 10.0, 30.0), seed: int = 0) -> dict:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(seed)
    t_all = time.perf_counter()

    print("1/4 training the physics-informed GNN on random synthetic trees (physics loss only) …", flush=True)
    gnn, gnn_info = train_gnn(synthetic_problem, n_iter=gnn_iters, batch=8, steps=steps, seed=seed,
                              log_every=max(1, gnn_iters // 6))

    print("2/4 picking the coordinate-PINN Fourier scale on tuning networks (not test networks) …", flush=True)
    tune_rng = np.random.default_rng(seed + 777)
    tuning = [synthetic_problem(tune_rng, generations=7, seed=TEST_SEED_BASE - 1 - k) for k in range(2)]
    sigma, sigma_table = (sigmas[0], {}) if len(sigmas) == 1 else tune_sigma(tuning, sigmas, pinn_adam, pinn_lbfgs)

    print("3/4 building test networks …", flush=True)
    test_rng = np.random.default_rng(seed + 1)
    problems = [(f"synthetic {k}", synthetic_problem(test_rng, generations=int(test_rng.integers(6, 10)),
                                                    seed=TEST_SEED_BASE + k)) for k in range(n_test)]
    if drive_dir is not None and Path(drive_dir).exists():
        problems += drive_problems(Path(drive_dir), max_drive)

    print(f"4/4 comparing methods on {len(problems)} networks …", flush=True)
    rows = []
    for name, pr in problems:
        t0 = time.perf_counter()
        ref = solve_network(pr.network, pr.bc, pr.viscosity)
        t_direct = time.perf_counter() - t0
        base = {"network": name, "set": name.split()[0], "nodes": pr.n, "segments": pr.network.n_edges}
        rows.append({**base, "method": "direct solve (reference)", "seconds": t_direct, "final_loss": 0.0,
                     **score(pr, ref, (ref.pressure_pa - pr.p_low) / pr.p_span)})
        for res in (jacobi(pr, steps), conjugate_gradient(pr, steps), gnn_solve(gnn, pr),
                    nodal_pinn(pr, adam_steps=0, lbfgs_steps=nodal_lbfgs),
                    coordinate_pinn(pr, adam_steps=pinn_adam, lbfgs_steps=pinn_lbfgs, sigma=sigma)):
            rows.append({**base, "method": res.method, "seconds": res.seconds, "final_loss": res.final_loss,
                         **score(pr, ref, res.p_scaled)})
        print(f"  {name:12s} {pr.n:4d} nodes  " + "  ".join(
            f"{r['method'].split('(')[0].strip()}: {r['pressure_max_error']:.1e}" for r in rows[-5:]), flush=True)

    summary = {}
    for st in sorted({r["set"] for r in rows}):
        for m in dict.fromkeys(r["method"] for r in rows):
            sel = [r for r in rows if r["set"] == st and r["method"] == m]
            if not sel:
                continue
            col = lambda k: np.array([r[k] for r in sel], float)  # noqa: E731
            summary.setdefault(st, {})[m] = {
                "networks": len(sel),
                "pressure_error_median": float(np.median(col("pressure_max_error"))),
                "pressure_error_worst": float(col("pressure_max_error").max()),
                "wss_median_rel_error": float(np.median(col("wss_median_rel_error"))),
                "wss_spearman_median": float(np.nanmedian(col("wss_spearman"))),
                "wss_top10_overlap_median": float(np.median(col("wss_top10_overlap"))),
                "conservation_error_worst": float(col("conservation_error").max()),
                "seconds_median": float(np.median(col("seconds")))}
    res = {"note": "numerical comparison on identical networks and boundary conditions; synthetic + DRIVE-geometry graphs",
           "settings": {"steps": steps, "gnn_iters": gnn_iters, "pinn_adam": pinn_adam, "pinn_lbfgs": pinn_lbfgs,
                        "nodal_lbfgs": nodal_lbfgs, "coordinate_sigma": sigma, "sigma_tuning": sigma_table, "seed": seed},
           "gnn_training": {k: v for k, v in gnn_info.items() if k != "history"}, "gnn_history": gnn_info["history"],
           "environment": {"python": platform.python_version(), "torch": torch.__version__,
                           "threads": torch.get_num_threads()},
           "summary": summary, "runtime_s": time.perf_counter() - t_all}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m8_results.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / "m8_per_network.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    torch.save({"model": gnn.state_dict(), "info": {k: v for k, v in gnn_info.items() if k != "history"}},
               out_dir / "pi_gnn.pt")
    plot(rows, out_dir / "m8_accuracy_vs_time.png")
    return res


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(r["method"] for r in rows))
    colors = dict(zip(methods, ["black", "#9e9e9e", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        err = np.maximum([r["pressure_max_error"] for r in sel], 1e-16)
        a1.scatter([r["seconds"] for r in sel], err, s=18, alpha=0.75, color=colors[m], label=m)
        if m != "direct solve (reference)":
            a2.scatter(np.maximum([r["final_loss"] for r in sel], 1e-30), err, s=18, alpha=0.75, color=colors[m], label=m)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("max pressure error (fraction of inlet-outlet drop)")
    a1.set_xlabel("wall time per network (s); PI-GNN excludes its one-time training")
    a1.set_title("Accuracy vs time", fontsize=10)
    a2.set_xlabel("final physics loss (mean squared normalised residual)")
    a2.set_title("Same physics loss, up to ~100x different pressure error", fontsize=10)
    a1.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root / "reports" / "m8"))
    ap.add_argument("--drive-dir", default=str(root / "data" / "DRIVE"))
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--max-drive", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gnn-iters", type=int, default=600)
    ap.add_argument("--quick", action="store_true", help="fewer networks and training steps (~5-10 min on Colab CPU)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    kw = dict(n_test=a.n_test, max_drive=a.max_drive, steps=a.steps, gnn_iters=a.gnn_iters)
    if a.quick:
        kw.update(n_test=8, max_drive=4, gnn_iters=200, pinn_adam=1500, pinn_lbfgs=300, nodal_lbfgs=1000,
                  sigmas=(10.0,))
    r = run(Path(a.out), drive_dir=Path(a.drive_dir), seed=a.seed, **kw)
    for st, methods in r["summary"].items():
        print(f"\n{st} networks (n={next(iter(methods.values()))['networks']}): median [worst] pressure error, "
              f"WSS top-10% overlap, median time")
        for m, s in methods.items():
            print(f"  {m:26s} {s['pressure_error_median']:.1e} [{s['pressure_error_worst']:.1e}]  "
                  f"overlap {s['wss_top10_overlap_median']:.2f}  {s['seconds_median'] * 1000:9.1f} ms")
    print(f"\nPI-GNN training: {r['gnn_training']['seconds']:.0f} s once; total runtime {r['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
