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

    ufxp = _fit(env, panel, UFXPEstimator(num_projections=64, seed=0))
    nfxp = _fit(env, panel, NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                                          compute_hessian=False, verbose=False))

    theta_u = np.asarray(ufxp.parameters)
    theta_n = np.asarray(nfxp.parameters)
    true = np.asarray(env.get_true_parameter_vector())

    assert ufxp.converged
    # Same estimand: UFXP within a small neighborhood of the MLE and the truth.
    assert np.max(np.abs(theta_u - theta_n)) < 0.15
    assert np.max(np.abs(theta_u - true)) < 0.2


def test_ufxp_on_rust_bus() -> None:
    env = RustBusEnvironment(num_mileage_bins=10, operating_cost=0.01,
                             replacement_cost=2.0, discount_factor=0.9)
    panel = simulate_panel(env, n_individuals=400, n_periods=60, seed=5)
    res = _fit(env, panel, UFXPEstimator(num_projections=64, seed=0))
    true = np.asarray(env.get_true_parameter_vector())
    assert res.converged
    assert np.max(np.abs(np.asarray(res.parameters) - true)) < 0.25
    assert res.policy is not None and res.value_function is not None


def test_ufxp_seed_determinism() -> None:
    env = random_mdp(num_states=8, num_actions=2, num_features=2,
                     branching=3, discount_factor=0.9, seed=0)
    panel = simulate_panel(env, n_individuals=200, n_periods=40, seed=9)
    a = _fit(env, panel, UFXPEstimator(num_projections=32, seed=3))
    b = _fit(env, panel, UFXPEstimator(num_projections=32, seed=3))
    assert np.allclose(np.asarray(a.parameters), np.asarray(b.parameters))


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
