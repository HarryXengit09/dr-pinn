"""Classical steady-state 1D network solver (Hagen-Poiseuille + Kirchhoff).

This is the baseline and verification reference for any PINN / GNN surrogate.

Per segment e = (u, v), treated as a rigid straight pipe with constant radius
and a Newtonian viscosity mu_e inside the segment:

    conductance      G_e   = pi r_e^4 / (8 mu_e L_e)             [m^3 / (Pa s)]
    volumetric flow  Q_e   = G_e (p_u - p_v)                     [m^3 / s], + means u -> v
    pressure drop    dP_e  = p_u - p_v = 8 mu_e L_e Q_e / (pi r_e^4)   [Pa]
    wall shear       tau_e = 4 mu_e |Q_e| / (pi r_e^3) = r_e |dP_e| / (2 L_e)  [Pa]
    mean velocity    v_e   = Q_e / (pi r_e^2)                    [m / s]

Mass conservation at each node i with no boundary condition (A = signed
incidence matrix):  sum_e A_ei Q_e = 0.  Substituting Q = G A p gives the
weighted graph-Laplacian system (A^T diag(G) A) p = b.

Boundary conditions (each node may carry at most one):
    Dirichlet  fixed pressure p_i                         (e.g. inlet at the disc)
    Neumann    fixed external inflow q_i into the network (+ = into the network)
    Robin      terminal resistance R_i to a reference pressure p_ref:
               inflow = (p_ref - p_i) / R_i               (lumped downstream bed)

Every connected component needs at least one Dirichlet or Robin node, otherwise
pressure is only defined up to a constant and the system is singular.

Steady state is a modelling choice: one photograph carries no time information.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .network import VesselNetwork


def poiseuille_conductance(radius_m: ArrayLike, length_m: ArrayLike,
                           viscosity_pa_s: ArrayLike) -> NDArray[np.float64]:
    """Hydraulic conductance pi r^4 / (8 mu L) in m^3 / (Pa s)."""
    r = np.asarray(radius_m, dtype=np.float64)
    length = np.asarray(length_m, dtype=np.float64)
    mu = np.asarray(viscosity_pa_s, dtype=np.float64)
    return math.pi * r**4 / (8.0 * mu * length)


def wall_shear_stress(flow_m3_s: ArrayLike, radius_m: ArrayLike,
                      viscosity_pa_s: ArrayLike) -> NDArray[np.float64]:
    """Poiseuille wall shear stress 4 mu |Q| / (pi r^3) in Pa."""
    q = np.abs(np.asarray(flow_m3_s, dtype=np.float64))
    r = np.asarray(radius_m, dtype=np.float64)
    return 4.0 * np.asarray(viscosity_pa_s, dtype=np.float64) * q / (math.pi * r**3)


@dataclass(frozen=True)
class BoundaryConditions:
    """Boundary conditions keyed by node index (SI units)."""

    pressure_pa: Mapping[int, float] = field(default_factory=dict)
    inflow_m3_s: Mapping[int, float] = field(default_factory=dict)
    terminal_resistance: Mapping[int, tuple[float, float]] = field(default_factory=dict)
    """node -> (R in Pa s / m^3, reference pressure in Pa)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "pressure_pa",
                           {int(k): float(v) for k, v in self.pressure_pa.items()})
        object.__setattr__(self, "inflow_m3_s",
                           {int(k): float(v) for k, v in self.inflow_m3_s.items()})
        object.__setattr__(self, "terminal_resistance",
                           {int(k): (float(r), float(p))
                            for k, (r, p) in self.terminal_resistance.items()})

    @classmethod
    def inlet_outlet_pressure(cls, inlet_nodes: Iterable[int], inlet_pressure_pa: float,
                              outlet_nodes: Iterable[int], outlet_pressure_pa: float
                              ) -> "BoundaryConditions":
        """One pressure for all inlets and one for all outlets."""
        bc = {int(i): inlet_pressure_pa for i in inlet_nodes}
        for o in outlet_nodes:
            if int(o) in bc:
                raise ValueError(f"node {o} is both an inlet and an outlet")
            bc[int(o)] = outlet_pressure_pa
        return cls(pressure_pa=bc)

    def anchor_nodes(self) -> set[int]:
        return set(self.pressure_pa) | set(self.terminal_resistance)

    def all_nodes(self) -> set[int]:
        return self.anchor_nodes() | set(self.inflow_m3_s)

    def validate(self, n_nodes: int) -> None:
        groups = (self.pressure_pa, self.inflow_m3_s, self.terminal_resistance)
        seen: set[int] = set()
        for group in groups:
            overlap = seen & set(group)
            if overlap:
                raise ValueError(f"nodes with more than one boundary condition: {sorted(overlap)}")
            seen |= set(group)
        bad = [i for i in seen if not 0 <= i < n_nodes]
        if bad:
            raise ValueError(f"boundary nodes outside [0, {n_nodes}): {bad}")
        values = list(self.pressure_pa.values()) + list(self.inflow_m3_s.values())
        values += [v for pair in self.terminal_resistance.values() for v in pair]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("boundary values must be finite")
        if any(r <= 0 for r, _ in self.terminal_resistance.values()):
            raise ValueError("terminal resistances must be > 0")
        if not self.anchor_nodes():
            raise ValueError("at least one pressure or terminal-resistance node is required")


@dataclass(frozen=True, eq=False)
class NetworkSolution:
    """Solver output. Edge arrays follow network.edges order; all SI units."""

    network: VesselNetwork
    boundary: BoundaryConditions
    viscosity_pa_s: NDArray[np.float64]
    conductance: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    flow_m3_s: NDArray[np.float64]
    pressure_drop_pa: NDArray[np.float64]
    wss_pa: NDArray[np.float64]
    mean_velocity_m_s: NDArray[np.float64]
    wall_shear_rate_s: NDArray[np.float64]
    node_injection_m3_s: NDArray[np.float64]
    conservation_residual_m3_s: NDArray[np.float64]
    total_inflow_m3_s: float
    max_rel_conservation_error: float
    max_dirichlet_error_pa: float
    linear_residual: float  # NaN when the pressures did not come from the linear solve

    def summary(self) -> dict[str, float]:
        from .units import m3_s_to_ul_per_min, pa_to_mmhg

        return {
            "n_nodes": self.network.n_nodes,
            "n_segments": self.network.n_edges,
            "total_inflow_ul_per_min": m3_s_to_ul_per_min(self.total_inflow_m3_s),
            "pressure_min_mmhg": pa_to_mmhg(self.pressure_pa.min()),
            "pressure_max_mmhg": pa_to_mmhg(self.pressure_pa.max()),
            "wss_min_pa": float(self.wss_pa.min()),
            "wss_median_pa": float(np.median(self.wss_pa)),
            "wss_max_pa": float(self.wss_pa.max()),
            "max_rel_conservation_error": self.max_rel_conservation_error,
            "max_dirichlet_error_pa": self.max_dirichlet_error_pa,
            "linear_residual": self.linear_residual,
        }


def _edge_viscosity(viscosity_pa_s: float | ArrayLike, n_edges: int) -> NDArray[np.float64]:
    mu = np.asarray(viscosity_pa_s, dtype=np.float64)
    mu = np.full(n_edges, float(mu)) if mu.ndim == 0 else mu.reshape(-1).copy()
    if mu.shape != (n_edges,):
        raise ValueError("viscosity must be a scalar or one value per edge")
    if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("viscosity must be finite and > 0 Pa s")
    return mu


def _check_components_anchored(network: VesselNetwork, bc: BoundaryConditions) -> None:
    n_comp, labels = network.connected_components()
    anchored = {int(labels[i]) for i in bc.anchor_nodes()}
    missing = sorted(set(range(n_comp)) - anchored)
    if missing:
        example = int(np.flatnonzero(labels == missing[0])[0])
        raise ValueError(
            f"{len(missing)} connected component(s) have no pressure or terminal-resistance "
            f"node (e.g. the component containing node {example}); pressure there is undefined"
        )


def solve_network(network: VesselNetwork, bc: BoundaryConditions,
                  viscosity_pa_s: float | ArrayLike) -> NetworkSolution:
    """Solve steady Poiseuille flow on a vessel network.

    Args:
        network: geometry in metres.
        bc: boundary conditions in SI units.
        viscosity_pa_s: scalar or per-edge effective viscosity (Pa s). Non-Newtonian
            laws enter by supplying per-edge apparent viscosities.

    Raises:
        ValueError: invalid inputs or an unanchored component.
        np.linalg.LinAlgError: the linear solve produced non-finite values.
    """
    n = network.n_nodes
    mu = _edge_viscosity(viscosity_pa_s, network.n_edges)
    bc.validate(n)
    _check_components_anchored(network, bc)

    g = poiseuille_conductance(network.radius_m, network.length_m, mu)
    a = network.incidence_matrix()
    # Conductances are ~1e-13 m^3/(Pa s); divide by a reference so the matrix
    # is O(1) and the right-hand side stays in pascals.
    g_ref = float(np.median(g))
    k = (a.T @ sparse.diags(g / g_ref) @ a).tocsr()

    rhs = np.zeros(n)
    extra_diag = np.zeros(n)
    for i, q in bc.inflow_m3_s.items():
        rhs[i] += q / g_ref
    for i, (r_term, p_ref) in bc.terminal_resistance.items():
        g_term = 1.0 / (r_term * g_ref)
        extra_diag[i] += g_term
        rhs[i] += g_term * p_ref
    k = (k + sparse.diags(extra_diag)).tocsr()

    p = np.zeros(n)
    fixed = np.array(sorted(bc.pressure_pa), dtype=np.int64)
    p[fixed] = [bc.pressure_pa[i] for i in fixed]
    free = np.setdiff1d(np.arange(n), fixed)
    linear_residual = 0.0
    if free.size:
        k_free = k[free]
        k_ff = k_free[:, free].tocsc()
        rhs_f = rhs[free] - k_free[:, fixed] @ p[fixed]
        p_f = np.atleast_1d(spsolve(k_ff, rhs_f))
        if not np.all(np.isfinite(p_f)):
            raise np.linalg.LinAlgError("linear solve returned non-finite pressures")
        p[free] = p_f
        lhs = k_ff @ p_f
        denom = max(np.linalg.norm(rhs_f), np.linalg.norm(lhs))
        linear_residual = float(np.linalg.norm(lhs - rhs_f) / denom) if denom > 0 else 0.0

    return evaluate_pressure_field(network, bc, mu, p, linear_residual=linear_residual)


def evaluate_pressure_field(network: VesselNetwork, bc: BoundaryConditions,
                            viscosity_pa_s: float | ArrayLike, pressure_pa: ArrayLike,
                            linear_residual: float = math.nan) -> NetworkSolution:
    """Derive flows, WSS and physics-violation metrics from ANY nodal pressure field.

    Used by the classical solver and, later, to score PINN/GNN predictions
    against the same conservation and boundary-condition checks.
    """
    n = network.n_nodes
    mu = _edge_viscosity(viscosity_pa_s, network.n_edges)
    bc.validate(n)
    p = np.array(pressure_pa, dtype=np.float64, copy=True).reshape(-1)
    if p.shape != (n,) or not np.all(np.isfinite(p)):
        raise ValueError("pressure_pa must hold one finite value per node")

    g = poiseuille_conductance(network.radius_m, network.length_m, mu)
    a = network.incidence_matrix()
    dp = a @ p
    q = g * dp
    injection = a.T @ q  # net flow leaving node i through its segments = external inflow
    expected = np.zeros(n)
    for i, qi in bc.inflow_m3_s.items():
        expected[i] = qi
    for i, (r_term, p_ref) in bc.terminal_resistance.items():
        expected[i] = (p_ref - p[i]) / r_term
    fixed = np.array(sorted(bc.pressure_pa), dtype=np.int64)
    residual = injection - expected
    residual[fixed] = 0.0  # Dirichlet reactions are whatever the network demands
    dirichlet_error = max((abs(p[i] - v) for i, v in bc.pressure_pa.items()), default=0.0)

    bc_nodes = np.array(sorted(bc.all_nodes()), dtype=np.int64)
    boundary_in = np.where(np.isin(np.arange(n), fixed), injection, expected)[bc_nodes]
    total_inflow = float(boundary_in[boundary_in > 0].sum())
    max_abs_res = float(np.abs(residual).max())
    if total_inflow > 0:
        rel_err = max_abs_res / total_inflow
    else:
        rel_err = 0.0 if max_abs_res == 0 else math.inf

    return NetworkSolution(
        network=network,
        boundary=bc,
        viscosity_pa_s=mu,
        conductance=g,
        pressure_pa=p,
        flow_m3_s=q,
        pressure_drop_pa=dp,
        wss_pa=wall_shear_stress(q, network.radius_m, mu),
        mean_velocity_m_s=q / (math.pi * network.radius_m**2),
        wall_shear_rate_s=4.0 * np.abs(q) / (math.pi * network.radius_m**3),
        node_injection_m3_s=injection,
        conservation_residual_m3_s=residual,
        total_inflow_m3_s=total_inflow,
        max_rel_conservation_error=rel_err,
        max_dirichlet_error_pa=float(dirichlet_error),
        linear_residual=linear_residual,
    )
