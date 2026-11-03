"""M8: physics-informed methods on vessel graphs vs the classical solver. Needs PyTorch."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drpinn.hemodynamics import (  # noqa: E402
    BoundaryConditions, murray_terminal_resistance, random_tree, series_chain, solve_network,
)
from drpinn.pinn.gnn import PhysicsGNN, batch_problems, gnn_solve, physics_loss, train_gnn  # noqa: E402
from drpinn.pinn.graph_physics import GraphProblem  # noqa: E402
from drpinn.pinn.methods import conjugate_gradient, coordinate_pinn, jacobi, nodal_pinn  # noqa: E402

MU = 3.5e-3
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m8_pinn_vs_solver.py"


def _tree(seed=3, gens=6, loops=3, bc_kind="dirichlet"):
    syn = random_tree(gens, seed=seed, n_loops=loops)
    net = syn.network
    tips = [int(o) for o in syn.outlet_nodes if net.degree()[o] == 1]
    if bc_kind == "dirichlet":
        bc = BoundaryConditions(pressure_pa={0: 5333.0, **{t: 2000.0 + 50 * (k % 3) for k, t in enumerate(tips)}})
    elif bc_kind == "robin":
        bc = BoundaryConditions(pressure_pa={0: 5333.0},
                                terminal_resistance=murray_terminal_resistance(net, tips, MU, 4, 20, 2000.0))
    else:
        bc = BoundaryConditions(inflow_m3_s={0: 3e-11},
                                terminal_resistance=murray_terminal_resistance(net, tips, MU, 4, 20, 2000.0))
    return net, bc


def _ref(pr):
    sol = solve_network(pr.network, pr.bc, MU)
    return (sol.pressure_pa - pr.p_low) / pr.p_span


@pytest.mark.parametrize("kind", ["dirichlet", "robin", "neumann+robin"])
def test_residual_vanishes_exactly_at_the_solver_solution(kind):
    net, bc = _tree(bc_kind=kind)
    pr = GraphProblem.build(net, bc, MU)
    s = torch.as_tensor(_ref(pr))
    assert pr.residual(s).abs().max() < 1e-10
    s2 = s.clone()
    free = int(torch.nonzero(~pr.fixed).flatten()[3])
    s2[free] += 0.01
    assert pr.residual(s2).abs().max() >= 0.01 - 1e-12
    np.testing.assert_allclose(pr.to_pa(s), solve_network(net, bc, MU).pressure_pa, rtol=1e-12)


def test_hard_boundary_conditions_hold_for_every_method():
    net, bc = _tree()
    pr = GraphProblem.build(net, bc, MU)
    fixed = pr.fixed.numpy()
    for r in (jacobi(pr, 5), conjugate_gradient(pr, 5), nodal_pinn(pr, adam_steps=5, lbfgs_steps=0),
              coordinate_pinn(pr, adam_steps=5, lbfgs_steps=0), gnn_solve(PhysicsGNN(steps=5), pr)):
        np.testing.assert_allclose(r.p_scaled[fixed], pr.fixed_value.numpy()[fixed], atol=1e-6)


@pytest.mark.parametrize("kind", ["dirichlet", "robin", "neumann+robin"])
def test_iterative_solvers_converge_to_the_direct_solve(kind):
    net, bc = _tree(bc_kind=kind)
    pr = GraphProblem.build(net, bc, MU)
    ref = _ref(pr)
    assert np.abs(conjugate_gradient(pr, 20 * pr.n).p_scaled - ref).max() < 1e-8
    assert np.abs(jacobi(pr, 20000).p_scaled - ref).max() < 1e-7
    assert np.abs(jacobi(pr, 5).p_scaled - ref).max() > 1e-3  # few steps: clearly not converged


def test_nodal_pinn_converges():
    net, bc = _tree(bc_kind="robin")
    pr = GraphProblem.build(net, bc, MU)
    r = nodal_pinn(pr, adam_steps=0, lbfgs_steps=1500)
    assert np.abs(r.p_scaled - _ref(pr)).max() < 1e-5


def test_coordinate_pinn_learns_a_linear_pressure_drop():
    syn = series_chain([1e-3] * 8, [50e-6] * 8)  # equal pipes in a line: pressure is linear in x
    bc = BoundaryConditions.inlet_outlet_pressure([0], 5333.0, [8], 2000.0)
    pr = GraphProblem.build(syn.network, bc, MU)
    r = coordinate_pinn(pr, adam_steps=300, lbfgs_steps=200, sigma=1.0)
    assert np.abs(r.p_scaled - _ref(pr)).max() < 1e-3


def test_untrained_gnn_is_exactly_jacobi():
    """Zero-initialised correction head: the untrained PI-GNN performs T Jacobi sweeps."""
    net, bc = _tree(bc_kind="robin")
    pr = GraphProblem.build(net, bc, MU)
    np.testing.assert_allclose(gnn_solve(PhysicsGNN(steps=12), pr).p_scaled, jacobi(pr, 12).p_scaled, atol=1e-6)


def test_batched_problems_keep_each_network_separate():
    probs = [GraphProblem.build(*_tree(seed=s, gens=4 + s % 2), MU) for s in range(3)]
    big = batch_problems(probs)
    p = torch.cat([torch.as_tensor(_ref(q)) for q in probs])
    assert big.n == sum(q.n for q in probs)
    assert big.residual(p).abs().max() < 1e-10


def test_gnn_training_lowers_the_physics_loss():
    def make(rng):
        return GraphProblem.build(*_tree(seed=int(rng.integers(10**6)), gens=int(rng.integers(4, 6)), loops=1), MU)
    model, info = train_gnn(make, n_iter=60, batch=4, hidden=16, steps=10, lr=3e-3, log_every=1000)
    rng = np.random.default_rng(99)
    batch = batch_problems([make(rng) for _ in range(6)])
    with torch.no_grad():
        trained = float(physics_loss(model, batch))
        untrained = float(physics_loss(PhysicsGNN(hidden=16, steps=10), batch))
    assert trained < 0.7 * untrained


def test_comparison_script_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location("m8", SCRIPT)
    m8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m8)
    res = m8.run(tmp_path, n_test=3, drive_dir=None, steps=10, gnn_iters=10, pinn_adam=100, pinn_lbfgs=30,
                 nodal_lbfgs=200, sigmas=(10.0,))
    s = res["summary"]["synthetic"]
    assert set(s) == {"direct solve (reference)", "Jacobi(10)", "CG(10)", "PI-GNN(10)", "nodal PINN", "coordinate PINN"}
    assert s["direct solve (reference)"]["pressure_error_worst"] < 1e-9
    assert s["direct solve (reference)"]["conservation_error_worst"] < 1e-9
    assert all(v["networks"] == 3 for v in s.values())
    assert json.loads((tmp_path / "m8_results.json").read_text())["settings"]["coordinate_sigma"] == 10.0
    for f in ("m8_per_network.csv", "m8_accuracy_vs_time.png", "pi_gnn.pt"):
        assert (tmp_path / f).exists()
