"""Tests for the UFXP estimator (Bray; Oguz and Bray 2026).

UFXP minimizes projected Bellman first-order conditions, so on a well-covered
panel its point estimate must land close to the NFXP maximum-likelihood
estimate and to the truth (consistency, Theorem 1 of the paper). The dual
trick is also checked directly: lambda' u_P must equal w' V_P.
"""

from __future__ import annotations

import numpy as np

from econirl.environments import RustBusEnvironment, random_mdp
from econirl.estimation import NFXPEstimator, UFXPEstimator
from econirl.simulation.synthetic import simulate_panel
from validation.benchmark.runner import _linear_utility


def _fit(env, panel, est):
    return est.estimate(panel, _linear_utility(env), env.problem_spec,
                        env.transition_matrices)


def test_ufxp_matches_nfxp_on_abstract_mdp() -> None:
    env = random_mdp(num_states=8, num_actions=2, num_features=2,
                     branching=3, discount_factor=0.9, seed=0)
    panel = simulate_panel(env, n_individuals=400, n_periods=60, seed=123)

    ufxp = _fit(env, panel, UFXPEstimator(weights="random", num_projections=64, seed=0))
    nfxp = _fit(env, panel, NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                                          compute_hessian=False, verbose=False))

    theta_u = np.asarray(ufxp.parameters)
    theta_n = np.asarray(nfxp.parameters)
    true = np.asarray(env.get_true_parameter_vector())

    assert ufxp.converged
    # Same estimand: UFXP within a small neighborhood of the MLE and the truth.
    assert np.max(np.abs(theta_u - theta_n)) < 0.15
    assert np.max(np.abs(theta_u - true)) < 0.2


def test_oufxp_efficiency_and_standard_errors() -> None:
    # Optimal weights (OUFXP): the point estimate should hug the MLE at least
    # as closely as the random-projection version, and the efficient-variance
    # standard errors should agree with NFXP's asymptotic SEs (Theorem 2 says
    # both are efficient, so they estimate the same limit).
    env = random_mdp(num_states=8, num_actions=2, num_features=2,
                     branching=3, discount_factor=0.9, seed=0)
    panel = simulate_panel(env, n_individuals=400, n_periods=60, seed=123)

    oufxp = _fit(env, panel, UFXPEstimator(weights="optimal"))
    random_z = _fit(env, panel, UFXPEstimator(weights="random",
                                              num_projections=64, seed=0))
    nfxp = _fit(env, panel, NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                                          compute_hessian=True, verbose=False))

    theta_o = np.asarray(oufxp.parameters)
    theta_r = np.asarray(random_z.parameters)
    theta_n = np.asarray(nfxp.parameters)

    assert oufxp.converged
    gap_o = float(np.max(np.abs(theta_o - theta_n)))
    gap_r = float(np.max(np.abs(theta_r - theta_n)))
    assert gap_o <= gap_r + 1e-6
    assert gap_o < 0.1

    se_o = np.asarray(oufxp.standard_errors)
    se_n = np.asarray(nfxp.standard_errors)
    assert np.all(np.isfinite(se_o)) and np.all(se_o > 0)
    # Both efficient: SEs within 30% of each other, parameter by parameter.
    assert np.all(np.abs(se_o - se_n) / se_n < 0.30)

    # The random-projection mode reports no standard errors.
    assert not np.any(np.isfinite(np.asarray(random_z.standard_errors)))


def test_ufxp_on_rust_bus() -> None:
    env = RustBusEnvironment(num_mileage_bins=10, operating_cost=0.01,
                             replacement_cost=2.0, discount_factor=0.9)
    panel = simulate_panel(env, n_individuals=400, n_periods=60, seed=5)
    res = _fit(env, panel, UFXPEstimator())
    true = np.asarray(env.get_true_parameter_vector())
    assert res.converged
    assert np.max(np.abs(np.asarray(res.parameters) - true)) < 0.25
    assert res.policy is not None and res.value_function is not None


def test_ufxp_seed_determinism() -> None:
    env = random_mdp(num_states=8, num_actions=2, num_features=2,
                     branching=3, discount_factor=0.9, seed=0)
    panel = simulate_panel(env, n_individuals=200, n_periods=40, seed=9)
    a = _fit(env, panel, UFXPEstimator(weights="random", num_projections=32, seed=3))
    b = _fit(env, panel, UFXPEstimator(weights="random", num_projections=32, seed=3))
    assert np.allclose(np.asarray(a.parameters), np.asarray(b.parameters))
    # Optimal weights are deterministic by construction (no randomness at all).
    c = _fit(env, panel, UFXPEstimator(weights="optimal"))
    d = _fit(env, panel, UFXPEstimator(weights="optimal"))
    assert np.allclose(np.asarray(c.parameters), np.asarray(d.parameters))


def test_ufxp_sklearn_wrapper() -> None:
    # The high-level UFXP class mirrors NFXP/CCP: DataFrame in, params_ out.
    import pandas as pd

    from econirl import UFXP

    env = RustBusEnvironment(num_mileage_bins=10, operating_cost=0.01,
                             replacement_cost=2.0, discount_factor=0.9)
    panel = simulate_panel(env, n_individuals=300, n_periods=60, seed=21)
    rows = []
    for i, traj in enumerate(panel.trajectories):
        for s, a in zip(np.asarray(traj.states), np.asarray(traj.actions)):
            rows.append({"bus_id": i, "mileage": int(s), "replaced": int(a)})
    df = pd.DataFrame(rows)

    model = UFXP(n_states=10, discount=0.9)
    model.fit(df, state="mileage", action="replaced", id="bus_id")

    assert model.converged_
    assert model.params_ is not None and len(model.params_) == 2
    assert all(np.isfinite(v) for v in model.params_.values())
    assert model.se_ is not None and all(np.isfinite(v) for v in model.se_.values())
    assert model.policy_.shape == (10, 2)


def test_ufxp_dual_identity() -> None:
    # lambda' u = w' V with V = (I - beta F_P)^{-1} u, for arbitrary w, u.
    rng = np.random.default_rng(0)
    S, beta = 12, 0.9
    F_P = rng.dirichlet(np.ones(S), size=S)  # row-stochastic (S, S)
    w = rng.standard_normal(S)
    u = rng.standard_normal(S)
    V = np.linalg.solve(np.eye(S) - beta * F_P, u)
    lam = np.linalg.solve(np.eye(S) - beta * F_P.T, w)
    assert np.isclose(lam @ u, w @ V)
