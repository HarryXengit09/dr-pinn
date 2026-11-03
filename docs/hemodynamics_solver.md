# The classical 1D network solver (Milestone 1)

## Idea in one paragraph
Each visible vessel segment is treated as a straight rigid pipe. In a narrow pipe with slow, steady flow, the flow rate is proportional to the pressure difference (Hagen–Poiseuille), exactly like current through a resistor is proportional to voltage (Ohm’s law). Junctions conserve flow, like Kirchhoff’s current law. So the whole vessel tree is an electrical resistor network: fix pressures (voltages) at inlets and outlets, solve one sparse linear system for every junction pressure, then read off flow and wall shear stress per segment.

## Equations (SI units: Pa, m, Pa·s, m³/s)

| Quantity | Formula | Units |
|---|---|---|
| Conductance | G = π r⁴ / (8 μ L) | m³/(Pa·s) |
| Flow (u→v positive) | Q = G (p_u − p_v) | m³/s |
| Pressure drop | ΔP = 8 μ L Q / (π r⁴) | Pa |
| Wall shear stress | τ = 4 μ \|Q\| / (π r³) = r \|ΔP\| / (2L) | Pa |
| Mean velocity | v = Q / (π r²) | m/s |
| Wall shear rate | γ̇ = 4 \|Q\| / (π r³) = τ / μ | 1/s |

Unit check for G·ΔP: m⁴ / (Pa·s·m) × Pa = m³/s ✓. For τ: Pa·s × m³/s / m³ = Pa ✓.

**Conservation.** With the signed incidence matrix A (one row per segment, +1 at u, −1 at v), Q = diag(G) A p, and the flow balance at every node is Aᵀ Q = b, i.e. **(Aᵀ diag(G) A) p = b** — a weighted graph Laplacian. `b` is zero at ordinary junctions and carries boundary inflows.

**Boundary conditions.**
- *Dirichlet* — fixed pressure (inlet at the optic disc, or an assumed outlet pressure).
- *Neumann* — fixed inflow.
- *Robin* — terminal resistance R to a reference pressure: inflow = (p_ref − p)/R. Models the unseen capillary bed downstream of a visible endpoint.

Every connected piece of the graph needs at least one Dirichlet or Robin node, otherwise pressure is defined only up to a constant; the solver refuses such input.

**Numerics.** Conductances are ~10⁻¹³ m³/(Pa·s), so the matrix is divided by the median conductance before solving (keeps it O(1); pressures stay in Pa). Solved with SciPy’s sparse direct solver. Reported diagnostics: relative linear residual, max conservation error relative to total inflow, max Dirichlet error.

## Two results the tests prove

**1. Murray’s law ⇒ uniform wall shear stress.** In a symmetric tree where r_parent³ = 2 r_child³, flow halves and r³ halves at each split, so τ ∝ Q/r³ is identical everywhere. A healthy, Murray-like tree therefore has a flat WSS ranking; WSS hot-spots come from **deviations** in geometry (narrowing, asymmetric branching). Narrowing one segment by 50 % (demo) gives ×1.9 WSS in that segment and ×0.24 in the 14 segments downstream — high and low shear at once (assumption A6).

**2. The ranking-invariance theorem.** If viscosity is uniform and Newtonian, all inlets share one pressure P_in and all outlets one pressure P_out, then p = P_out + (P_in − P_out)·φ, where φ depends only on the *ratios* of conductances. Hence for every segment
τ_e = (P_in − P_out) · r_e |Δφ_e| / (2 L_e).
Consequences:
- Changing blood pressure multiplies **every** τ by the same factor → ranking unchanged.
- Changing μ, rescaling the image (pixel size), or a uniform ×c radius bias → ranking unchanged.
- So a Monte Carlo study over only these parameters reports perfect rank stability **by construction**, not as evidence (PDF §7.3, RQ3).
- Rankings *do* change with heterogeneous outlet conditions, per-segment radius errors, additive segmentation bias, diameter-dependent viscosity and shear-thinning — these must be the uncertainty inputs (M2, M7).

## Using it

```python
from drpinn.hemodynamics import BoundaryConditions, VesselNetwork, solve_network
from drpinn.hemodynamics.units import mmhg_to_pa, um_to_m

net = VesselNetwork(
    n_nodes=4,
    edges=[[0, 1], [1, 2], [1, 3]],                       # inlet → junction → two outlets
    length_m=[um_to_m(1200), um_to_m(900), um_to_m(1500)],
    radius_m=[um_to_m(55), um_to_m(42), um_to_m(38)],
)
bc = BoundaryConditions.inlet_outlet_pressure([0], mmhg_to_pa(40), [2, 3], mmhg_to_pa(15))
sol = solve_network(net, bc, viscosity_pa_s=3.5e-3)       # illustrative values
print(sol.wss_pa, sol.summary())
```
`evaluate_pressure_field(net, bc, mu, pressures)` scores any pressure field (e.g. a PINN prediction) with the same conservation and boundary checks.

## Limitations of the 1D pipe model (carried into every report)
Rigid straight cylinders; constant radius per segment; no curvature, no entrance or junction losses; steady flow; Newtonian within a segment (non-Newtonian effects only through per-segment apparent viscosity, M2); only visible vessels; no autoregulation. Absolute values depend on uncalibrated boundary pressures — only within-eye relative rankings are reported.

## Milestone 2 additions

**Viscosity models** (`hemodynamics/rheology.py`). Each returns a per-segment apparent viscosity for the same Poiseuille solver.
- *Newtonian* — one constant value (baseline).
- *Fåhræus–Lindqvist* — Pries, Neuhaus & Gaehtgens, Am J Physiol 263:H1770 (1992), in-vitro Eqs. 3–5 as reproduced in Secomb & Pries, C R Physique 14:470 (2013). Viscosity relative to plasma depends on absolute diameter D (µm) and discharge hematocrit H_D; fitted to high-shear tube data. Because it uses absolute diameter, image scale now matters (test: `test_pries_makes_image_scale_matter`).
- *Carreau–Yasuda* — μ = μ∞ + (μ0 − μ∞)[1 + (λγ̇)^a]^((n−1)/a), Cho & Kensey parameters (μ0 0.056 Pa·s, μ∞ 0.0035 Pa·s, λ 3.313 s, n 0.3568, a 2), evaluated at the Poiseuille wall shear rate. Solved by damped fixed-point iteration in log-viscosity; non-convergence is warned and reported.

A uniform viscosity factor (e.g. plasma viscosity) cancels from WSS rankings; only between-segment variation matters.

**Force balance.** In a single pipe τ = r ΔP / (2L) for *any* rheology — viscosity changes the flow, not this balance. In a network, rheology changes WSS only by redistributing pressure drops between segments.

**Terminal resistance.** One unseen Murray generation below a segment of radius r (length/radius ratio λ) has resistance 8μλ/(πr³) — the same for every generation, because 2^k parallel segments each have resistance 8μλ·2^k/(πr³). An outlet draining k unseen generations gets R = k·8μλ/(πr³). Test: a tree truncated by k generations with this Robin BC reproduces the full tree's flows to 1e-10.

**Geometry-only features** (`vessel_graph/features.py`): radius, length, length/radius, tortuosity (from centrelines; NaN if unavailable), branch order, path length from inlet, parent radius ratio, Murray deviation (r_p³ − Σr_c³)/r_p³, sibling radius ratio, branch angle, terminal flag. No flow quantities — this is the RQ1 baseline.

**Sensitivity study** (`scripts/m2_rank_sensitivity.py`): results and consequences in `docs/PLAN.md` §6.
