import math

import numpy as np
import pytest

from drpinn.hemodynamics import (
    BoundaryConditions,
    CarreauYasudaViscosity,
    NewtonianViscosity,
    PriesInVitroViscosity,
    pries_relative_viscosity,
    random_tree,
    single_pipe,
    solve_with_viscosity_model,
)
from drpinn.uncertainty import spearman


def _bc(syn, p_in=5333.0, p_out=2000.0):
    return BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, p_in, syn.outlet_nodes, p_out)


# --- Pries et al. 1992 in-vitro law ------------------------------------------------

@pytest.mark.parametrize("d_um,expected", [(100.0, 2.443), (40.0, 1.924)])
def test_pries_matches_hand_calculation_at_hd_045(d_um, expected):
    # Hand-evaluated from Eq. 4: 3.2 - 2.44 exp(-0.06 D^0.645) (+ a negligible 220 e^-1.3D term)
    assert pries_relative_viscosity(d_um, 0.45) == pytest.approx(expected, abs=2e-3)


def test_pries_limits_and_monotonicity():
    d = np.array([8.0, 30.0, 120.0, 500.0])
    np.testing.assert_allclose(pries_relative_viscosity(d, 0.0), 1.0)  # plasma only
    lo, mid, hi = (pries_relative_viscosity(d, h) for h in (0.2, 0.45, 0.6))
    assert np.all(lo < mid) and np.all(mid < hi)
    assert pries_relative_viscosity(1e4, 0.45) == pytest.approx(3.2, abs=1e-3)  # large-tube limit
    # Fåhræus-Lindqvist: apparent viscosity falls as tubes narrow (in this range)
    assert np.all(np.diff(pries_relative_viscosity(np.array([20, 50, 100, 300.0]), 0.45)) > 0)


@pytest.mark.parametrize("d,hd", [(0.0, 0.45), (-5.0, 0.45), (math.nan, 0.45), (50.0, 1.0), (50.0, -0.1)])
def test_pries_rejects_invalid_inputs(d, hd):
    with pytest.raises(ValueError):
        pries_relative_viscosity(d, hd)


def test_plasma_viscosity_cancels_from_wss():
    syn = random_tree(7, seed=4)
    a = solve_with_viscosity_model(syn.network, _bc(syn), PriesInVitroViscosity(1.1e-3)).solution
    b = solve_with_viscosity_model(syn.network, _bc(syn), PriesInVitroViscosity(1.4e-3)).solution
    np.testing.assert_allclose(a.wss_pa, b.wss_pa, rtol=1e-9)


def test_pries_makes_image_scale_matter():
    """Diameter-dependent viscosity breaks the M1 scale invariance."""
    syn = random_tree(8, seed=6)
    for model, should_change in ((NewtonianViscosity(), False), (PriesInVitroViscosity(), True)):
        base = solve_with_viscosity_model(syn.network, _bc(syn), model).solution.wss_pa
        scaled = solve_with_viscosity_model(syn.network.scaled(0.6), _bc(syn), model).solution.wss_pa
        rho = spearman(base, scaled)
        assert (rho < 1 - 1e-6) if should_change else rho == pytest.approx(1.0, abs=1e-12)


# --- Carreau-Yasuda ----------------------------------------------------------------

def test_carreau_yasuda_limits():
    cy = CarreauYasudaViscosity()
    assert cy.viscosity_at(0.0) == pytest.approx(cy.mu_0_pa_s)
    assert cy.viscosity_at(1e9) == pytest.approx(cy.mu_inf_pa_s, rel=1e-3)
    assert np.all(np.diff(cy.viscosity_at(np.logspace(-2, 5, 50))) < 0)  # shear-thinning
    np.testing.assert_allclose(CarreauYasudaViscosity(n=1.0).viscosity_at([0.1, 1e3]), cy.mu_0_pa_s)


def test_carreau_yasuda_hand_value():
    # gamma = 1/s: 1 + 3.313^2 = 11.976; 11.976^(-0.3216) = 0.4500
    # mu = 0.0035 + 0.0525 * 0.4500 = 0.02712 Pa s
    assert CarreauYasudaViscosity().viscosity_at(1.0) == pytest.approx(0.02712, rel=1e-3)


def test_carreau_yasuda_rejects_bad_parameters():
    with pytest.raises(ValueError):
        CarreauYasudaViscosity(mu_0_pa_s=0.001, mu_inf_pa_s=0.0035)
    with pytest.raises(ValueError):
        CarreauYasudaViscosity().edge_viscosity([5e-5])


def test_nonlinear_solve_converges_and_is_self_consistent():
    syn = random_tree(8, seed=9, n_loops=5)
    res = solve_with_viscosity_model(syn.network, _bc(syn), CarreauYasudaViscosity())
    assert res.converged and res.iterations < 200
    assert res.self_consistency_error < 1e-8
    assert res.solution.max_rel_conservation_error < 1e-10
    assert res.history[-1] < 1e-10


def test_single_pipe_wss_is_set_by_force_balance_for_any_rheology():
    """tau = r dP / (2L) holds for any viscosity: rheology changes flow, not this balance."""
    length, r, dp = 1e-3, 40e-6, 800.0
    syn = single_pipe(length, r)
    bc = BoundaryConditions.inlet_outlet_pressure([0], dp, [1], 0.0)
    for model in (NewtonianViscosity(), PriesInVitroViscosity(), CarreauYasudaViscosity()):
        sol = solve_with_viscosity_model(syn.network, bc, model).solution
        assert sol.wss_pa[0] == pytest.approx(r * dp / (2 * length), rel=1e-9)


def test_non_convergence_is_reported_not_hidden():
    syn = random_tree(6, seed=1)
    with pytest.warns(RuntimeWarning, match="no convergence"):
        res = solve_with_viscosity_model(syn.network, _bc(syn), CarreauYasudaViscosity(), max_iter=2)
    assert not res.converged


def test_invalid_damping_rejected():
    syn = random_tree(4, seed=1)
    with pytest.raises(ValueError):
        solve_with_viscosity_model(syn.network, _bc(syn), CarreauYasudaViscosity(), damping=0.0)
