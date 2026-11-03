"""Synthetic demo: what the 1D model says about a narrowed segment.

SYNTHETIC NETWORK, ILLUSTRATIVE PARAMETERS — software/physics demonstration,
not a patient result.

    python scripts/demo_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn import DISCLAIMER  # noqa: E402
from drpinn.hemodynamics import BoundaryConditions, murray_tree, solve_network  # noqa: E402
from drpinn.hemodynamics.units import mmhg_to_pa, m3_s_to_ul_per_min  # noqa: E402

MU = 3.5e-3  # Pa s, illustrative Newtonian value


def bc_for(syn, p_in_mmhg, p_out_mmhg=15.0):
    return BoundaryConditions.inlet_outlet_pressure(
        syn.inlet_nodes, mmhg_to_pa(p_in_mmhg), syn.outlet_nodes, mmhg_to_pa(p_out_mmhg))


def main() -> None:
    print("SYNTHETIC DEMO —", DISCLAIMER, "\n")
    syn = murray_tree(6, root_radius_m=60e-6, length_to_radius=20)
    base = solve_network(syn.network, bc_for(syn, 40.0), MU)
    print(f"Murray tree: {syn.network.n_edges} segments, inflow "
          f"{m3_s_to_ul_per_min(base.total_inflow_m3_s):.2f} uL/min, WSS spread "
          f"{np.ptp(base.wss_pa):.1e} Pa (uniform, as Murray's law predicts)")

    e = int(np.flatnonzero(syn.edge_generation == 2)[0])
    sibling = int(np.flatnonzero((syn.parent_edge == syn.parent_edge[e]))[1])
    radius = syn.network.radius_m.copy()
    radius[e] *= 0.5
    syn_s = type(syn)(syn.network.with_radius(radius), syn.inlet_nodes, syn.outlet_nodes,
                      syn.edge_generation, syn.parent_edge, syn.description + " + stenosis")
    sten = solve_network(syn_s.network, bc_for(syn_s, 40.0), MU)
    down = syn.descendants(e)
    ratio = sten.wss_pa / base.wss_pa
    print(f"\n50% radius narrowing of segment {e} (generation 2):")
    print(f"  narrowed segment WSS  x{ratio[e]:.2f}   (high shear)")
    print(f"  its {down.size} downstream segments  x{ratio[down].mean():.2f}   (low shear)")
    print(f"  sibling segment {sibling}       x{ratio[sibling]:.2f}")
    print(f"  conservation error {sten.max_rel_conservation_error:.1e}")

    # A symmetric tree has many exactly tied WSS values; round before ranking so
    # floating-point noise does not reorder ties (the ranking module must do the same).
    def tie_safe(w):
        return np.round(w / w.max(), 9)

    print("\nBlood-pressure sweep (two-pressure Newtonian model, P_out = 15 mmHg):")
    for p_in in (30.0, 50.0, 70.0):
        w = solve_network(syn_s.network, bc_for(syn_s, p_in), MU).wss_pa
        k = (p_in - 15.0) / (40.0 - 15.0)
        dev = np.abs(w / (k * sten.wss_pa) - 1.0).max()
        rho = spearmanr(tie_safe(w), tie_safe(sten.wss_pa)).statistic
        print(f"  P_in={p_in:.0f} mmHg  max WSS {w.max():6.2f} Pa  "
              f"deviation from pure rescaling {dev:.1e}  Spearman vs 40 mmHg {rho:.6f}")
    print("  -> absolute WSS rescales with pressure; the ranking cannot change.")


if __name__ == "__main__":
    main()
