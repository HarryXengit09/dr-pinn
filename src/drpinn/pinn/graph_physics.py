"""The 1D network physics as a differentiable residual (PyTorch).

Unknowns: nodal pressures. For every node without a fixed pressure,
mass conservation (Kirchhoff) with Poiseuille edge flows reads

    sum_j g_ij (p_i - p_j) + gt_i (p_i - pref_i) - q_i = 0

(g = segment conductance, gt/pref = terminal resistance to a reference
pressure, q = prescribed inflow). Dividing by the diagonal
D_i = sum_j g_ij + gt_i gives the Jacobi-normalised residual

    r_i = p_i - (sum_j g_ij p_j + gt_i pref_i + q_i) / D_i,

which is dimensionless after pressures are scaled to [0, 1] between the
lowest and highest boundary pressure. r = 0 exactly at the classical
solution. Fixed-pressure (Dirichlet) nodes are imposed exactly ("hard"
boundary conditions) by every method, so the loss only contains physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..hemodynamics.network import VesselNetwork
from ..hemodynamics.solver import BoundaryConditions, poiseuille_conductance


@dataclass
class GraphProblem:
    n: int
    src: torch.Tensor          # (2E,) directed edge sources (each segment both ways)
    dst: torch.Tensor          # (2E,)
    w: torch.Tensor            # (2E,) g_ij / D_dst : weight of p_src in node dst's mean
    fixed: torch.Tensor        # (N,) bool, Dirichlet nodes
    fixed_value: torch.Tensor  # (N,) scaled pressure at fixed nodes (0 elsewhere)
    bias: torch.Tensor         # (N,) (gt*pref + q) / D, scaled
    robin_frac: torch.Tensor   # (N,) gt / D (share of the node's conductance going to the bed)
    coords: torch.Tensor       # (N, 2) node positions scaled to [-1, 1]
    g: torch.Tensor            # (2E,) conductances / median conductance (both directions)
    diag: torch.Tensor         # (N,) D_i = sum_j g_ij + gt_i
    p_low: float               # Pa
    p_span: float              # Pa
    network: VesselNetwork
    bc: BoundaryConditions
    viscosity: np.ndarray

    @classmethod
    def build(cls, network: VesselNetwork, bc: BoundaryConditions, viscosity_pa_s,
              dtype=torch.float64) -> "GraphProblem":
        n = network.n_nodes
        bc.validate(n)
        mu = np.broadcast_to(np.asarray(viscosity_pa_s, float), (network.n_edges,)).copy()
        g = poiseuille_conductance(network.radius_m, network.length_m, mu)
        g_ref = float(np.median(g))
        g = g / g_ref
        u, v = network.edges[:, 0], network.edges[:, 1]
        diag = np.bincount(u, g, n) + np.bincount(v, g, n)
        gt = np.zeros(n)
        pref = np.zeros(n)
        q = np.zeros(n)
        for i, (r, pr) in bc.terminal_resistance.items():
            gt[i], pref[i] = 1.0 / (r * g_ref), pr
        for i, qi in bc.inflow_m3_s.items():
            q[i] = qi / g_ref  # in Pa after dividing by D below
        diag = diag + gt
        levels = list(bc.pressure_pa.values()) + [p for _, p in bc.terminal_resistance.values()]
        p_low = float(min(levels))
        p_span = float(max(levels) - p_low) or 1.0
        fixed = np.zeros(n, bool)
        fval = np.zeros(n)
        for i, p in bc.pressure_pa.items():
            fixed[i], fval[i] = True, (p - p_low) / p_span
        # inflow (Pa-equivalent) and reference pressures in scaled units
        bias = (gt * np.where(gt > 0, (pref - p_low) / p_span, 0.0) + q / p_span) / diag
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        gg = np.concatenate([g, g])
        xy = network.node_xy if network.node_xy is not None else np.zeros((n, 2))
        span = np.ptp(xy, axis=0)
        span[span == 0] = 1.0
        coords = 2 * (xy - xy.min(axis=0)) / span - 1
        t = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt)  # noqa: E731
        return cls(n, t(src, torch.long), t(dst, torch.long), t(gg / diag[dst]), t(fixed, torch.bool),
                   t(fval), t(bias), t(gt / diag), t(coords), t(gg), t(diag), p_low, p_span, network, bc, mu)

    def to(self, dtype) -> "GraphProblem":
        conv = {k: (getattr(self, k).to(dtype) if torch.is_floating_point(getattr(self, k)) else getattr(self, k))
                for k in ("src", "dst", "w", "fixed", "fixed_value", "bias", "robin_frac", "coords", "g", "diag")}
        return GraphProblem(self.n, **conv, p_low=self.p_low, p_span=self.p_span, network=self.network,
                            bc=self.bc, viscosity=self.viscosity)

    def neighbour_mean(self, p: torch.Tensor) -> torch.Tensor:
        """(sum_j g_ij p_j) / D_i for every node (works on a trailing batch dimension too)."""
        out = torch.zeros((self.n, *p.shape[1:]), dtype=p.dtype)
        wp = self.w.view(-1, *([1] * (p.dim() - 1))) * p[self.src]
        return out.index_add_(0, self.dst, wp)

    def apply_bc(self, p: torch.Tensor) -> torch.Tensor:
        """Hard Dirichlet constraint: fixed nodes take their boundary value."""
        return torch.where(self.fixed, self.fixed_value, p)

    def residual(self, p: torch.Tensor) -> torch.Tensor:
        """Normalised Kirchhoff residual at free nodes (scaled pressures in)."""
        r = p - self.neighbour_mean(p) - self.bias
        return r[~self.fixed]

    def loss(self, p: torch.Tensor) -> torch.Tensor:
        return (self.residual(p) ** 2).mean()

    def to_pa(self, p_scaled) -> np.ndarray:
        p = p_scaled.detach().cpu().numpy() if isinstance(p_scaled, torch.Tensor) else np.asarray(p_scaled)
        return self.p_low + self.p_span * p.astype(np.float64)

    def initial_guess(self) -> torch.Tensor:
        """Mean of the boundary values everywhere, boundary values imposed."""
        start = torch.full((self.n,), float(self.fixed_value[self.fixed].mean()) if self.fixed.any() else 0.5,
                           dtype=self.w.dtype)
        return self.apply_bc(start)
