"""Blood viscosity models and the nonlinear (shear-dependent) network solve.

Every model returns a per-segment *apparent* viscosity that is fed to the
Poiseuille network solver. That is a reduced-order approximation: within each
segment the velocity profile is still treated as parabolic.

Models
------
NewtonianViscosity
    One constant viscosity. Baseline.
PriesInVitroViscosity  (Fåhræus–Lindqvist effect)
    Pries, Neuhaus & Gaehtgens, Am J Physiol 263:H1770-H1778 (1992), Eqs. 3-5
    as reproduced in Secomb & Pries, C R Physique 14:470-478 (2013). Fitted to
    glass-tube data at high shear (mean velocity / diameter >= 50 1/s). Depends
    on ABSOLUTE diameter in micrometres, so image-scale calibration matters.
CarreauYasudaViscosity  (shear-thinning)
    Default parameters: Cho & Kensey (1991) Carreau fit, as tabulated in
    arXiv:2503.23049, Table II. Bulk-rheometry values for whole blood, not
    microvessel measurements. Evaluated at the Poiseuille wall shear rate.

A uniform multiplicative factor (e.g. the plasma viscosity) cancels out of
within-eye WSS rankings; only the *variation* of viscosity between segments
can reorder them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .network import VesselNetwork
from .solver import BoundaryConditions, NetworkSolution, solve_network


@dataclass(frozen=True)
class NewtonianViscosity:
    mu_pa_s: float = 3.5e-3  # ASSUMPTION: typical high-shear whole-blood value
    shear_dependent: bool = field(default=False, init=False)
    name: str = field(default="newtonian", init=False)

    def edge_viscosity(self, radius_m: ArrayLike,
                       wall_shear_rate_s: ArrayLike | None = None) -> NDArray[np.float64]:
        if not math.isfinite(self.mu_pa_s) or self.mu_pa_s <= 0:
            raise ValueError("mu_pa_s must be finite and > 0")
        return np.full(np.shape(radius_m), float(self.mu_pa_s))


def pries_relative_viscosity(diameter_um: ArrayLike, discharge_hematocrit: float) -> NDArray[np.float64]:
    """Relative apparent viscosity (to plasma), Pries et al. 1992 in vitro, Eqs. 3-5."""
    d = np.asarray(diameter_um, dtype=np.float64)
    hd = float(discharge_hematocrit)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("diameter must be finite and > 0 um")
    if not 0.0 <= hd < 1.0:
        raise ValueError("discharge hematocrit must be in [0, 1)")
    mu_045 = 220.0 * np.exp(-1.3 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645)
    sigmoid = 1.0 / (1.0 + 1e-11 * d**12)
    c = (0.8 + np.exp(-0.075 * d)) * (-1.0 + sigmoid) + sigmoid
    return 1.0 + (mu_045 - 1.0) * ((1.0 - hd) ** c - 1.0) / ((1.0 - 0.45) ** c - 1.0)


@dataclass(frozen=True)
class PriesInVitroViscosity:
    plasma_viscosity_pa_s: float = 1.2e-3  # ASSUMPTION: typical plasma value; cancels in rankings
    discharge_hematocrit: float = 0.45     # ASSUMPTION: systemic-like value; uncertain in microvessels
    shear_dependent: bool = field(default=False, init=False)
    name: str = field(default="pries_in_vitro", init=False)

    def edge_viscosity(self, radius_m: ArrayLike,
                       wall_shear_rate_s: ArrayLike | None = None) -> NDArray[np.float64]:
        if not math.isfinite(self.plasma_viscosity_pa_s) or self.plasma_viscosity_pa_s <= 0:
            raise ValueError("plasma_viscosity_pa_s must be finite and > 0")
        d_um = 2.0 * np.asarray(radius_m, dtype=np.float64) * 1e6
        return self.plasma_viscosity_pa_s * pries_relative_viscosity(d_um, self.discharge_hematocrit)


@dataclass(frozen=True)
class CarreauYasudaViscosity:
    """mu = mu_inf + (mu_0 - mu_inf) [1 + (lambda * gamma_dot)^a]^((n - 1) / a)."""

    mu_0_pa_s: float = 0.056
    mu_inf_pa_s: float = 0.0035
    lambda_s: float = 3.313
    n: float = 0.3568
    a: float = 2.0  # a = 2 is the Carreau special case used by Cho & Kensey
    shear_dependent: bool = field(default=True, init=False)
    name: str = field(default="carreau_yasuda", init=False)

    def __post_init__(self) -> None:
        if not (0 < self.mu_inf_pa_s <= self.mu_0_pa_s) or self.lambda_s < 0 or self.a <= 0:
            raise ValueError("need 0 < mu_inf <= mu_0, lambda >= 0, a > 0")

    def viscosity_at(self, shear_rate_s: ArrayLike) -> NDArray[np.float64]:
        g = np.abs(np.asarray(shear_rate_s, dtype=np.float64))
        factor = (1.0 + (self.lambda_s * g) ** self.a) ** ((self.n - 1.0) / self.a)
        return self.mu_inf_pa_s + (self.mu_0_pa_s - self.mu_inf_pa_s) * factor

    def edge_viscosity(self, radius_m: ArrayLike,
                       wall_shear_rate_s: ArrayLike | None = None) -> NDArray[np.float64]:
        if wall_shear_rate_s is None:
            raise ValueError("Carreau-Yasuda needs the wall shear rate of each segment")
        return self.viscosity_at(wall_shear_rate_s)


ViscosityModel = NewtonianViscosity | PriesInVitroViscosity | CarreauYasudaViscosity


@dataclass(frozen=True, eq=False)
class RheologySolution:
    solution: NetworkSolution
    model_name: str
    converged: bool
    iterations: int
    history: tuple[float, ...]  # max relative viscosity change per iteration
    self_consistency_error: float  # max |mu - model(mu-derived shear rate)| / mu at the end


def solve_with_viscosity_model(network: VesselNetwork, bc: BoundaryConditions,
                               model: ViscosityModel, max_iter: int = 200, tol: float = 1e-10,
                               damping: float = 0.5) -> RheologySolution:
    """Solve the network with a (possibly shear-dependent) viscosity model.

    Shear-dependent models use a damped fixed-point iteration in log-viscosity:
    solve -> wall shear rates -> new viscosities -> blend -> repeat, until the
    largest relative viscosity change is below `tol`. Not converging is
    reported (and warned), never hidden.
    """
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    r = network.radius_m
    if not model.shear_dependent:
        sol = solve_network(network, bc, model.edge_viscosity(r))
        return RheologySolution(sol, model.name, True, 0, (), 0.0)

    mu = np.full(network.n_edges, model.mu_inf_pa_s)
    history: list[float] = []
    converged = False
    for _ in range(max_iter):
        sol = solve_network(network, bc, mu)
        mu_target = model.edge_viscosity(r, sol.wall_shear_rate_s)
        change = float(np.max(np.abs(mu_target / mu - 1.0)))
        history.append(change)
        if change < tol:
            converged = True
            break
        mu = np.exp((1.0 - damping) * np.log(mu) + damping * np.log(mu_target))
    sol = solve_network(network, bc, mu)
    consistency = float(np.max(np.abs(model.edge_viscosity(r, sol.wall_shear_rate_s) / mu - 1.0)))
    if not converged:
        warnings.warn(f"{model.name}: no convergence in {max_iter} iterations "
                      f"(last change {history[-1]:.2e})", RuntimeWarning, stacklevel=2)
    return RheologySolution(sol, model.name, converged, len(history), tuple(history), consistency)
