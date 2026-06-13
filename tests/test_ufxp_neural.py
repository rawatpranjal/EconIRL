"""Tests for the neural-utility UFXP estimator (Oguz and Bray 2026, Phase 1).

The headline: where an omitted interaction defeats a linear utility, the neural
UFXP learns it from the same features and recovers behavior, reusing the linear
estimator's precomputed dual (no Bellman solve in the training loop).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.array_mdp import ArrayMDP
from econirl.environments import random_mdp
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration


def _policy_tv(pi_a, pi_b):
    return float(0.5 * np.abs(np.asarray(pi_a) - np.asarray(pi_b)).sum(axis=1).mean())


def _oracle(env):
    op = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    sol = value_iteration(op, env.compute_utility_matrix())
    return np.asarray(sol.policy), np.asarray(sol.V)


class _InteractionMDP(ArrayMDP):
    """True reward adds gamma * phi0 * phi1; estimators see only [phi0, phi1]."""

    def __init__(self, T, features, theta, gamma, **kwargs):
        super().__init__(T, features, theta, **kwargs)
        phi = np.asarray(features)
        self._R = jnp.asarray(
            np.einsum("sak,k->sa", phi, np.asarray(theta))
            + gamma * phi[:, :, 0] * phi[:, :, 1], dtype=jnp.float32)

    def compute_utility_matrix(self, parameters=None):
        return self._R if parameters is None else super().compute_utility_matrix(parameters)

    @property
    def true_reward_matrix(self):
        return self._R


def _interaction_env(S=24, A=3, gamma=2.5, seed=404):
    rng = np.random.default_rng(seed)
    T = np.zeros((A, S, S))
    for a in range(A):
        for s in range(S):
            sup = rng.choice(S, 4, replace=False)
            T[a, s, sup] = rng.dirichlet(np.ones(4))
    T /= T.sum(axis=2, keepdims=True)
    x = np.linspace(0, 1, S)
    phi = np.zeros((S, A, 2))
    phi[:, 1, 0] = x
    phi[:, 1, 1] = np.sin(np.pi * x)
    phi[:, 2, 0] = 1 - x
    phi[:, 2, 1] = np.cos(np.pi * x)
    return _InteractionMDP(T, phi, np.array([1.0, -0.8]), gamma=gamma,
                           discount_factor=0.95, seed=seed)


def test_neural_ufxp_recovers_interaction_where_linear_fails():
    """Neural UFXP learns the omitted interaction; linear UFXP cannot."""
    from econirl import NeuralUFXP
    from econirl.estimation import UFXPEstimator
    from validation.benchmark.runner import _linear_utility

    env = _interaction_env()
    oracle_policy, _ = _oracle(env)
    panel = simulate_panel(env, n_individuals=300, n_periods=50, seed=1404)

    lin = UFXPEstimator(weights="optimal").estimate(
        panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
    lin_tv = _policy_tv(lin.policy, oracle_policy)

    neu = NeuralUFXP(discount=0.95, num_projections=64, max_epochs=2000,
                     lr=1e-2, seed=0).fit(
        panel, features=np.asarray(env.feature_matrix),
        transitions=np.asarray(env.transition_matrices))
    neu_tv = _policy_tv(neu.policy_, oracle_policy)

    assert lin_tv > 0.08, f"linear UFXP should hit the interaction wall, got {lin_tv}"
    assert neu_tv < 0.05, f"neural UFXP should recover, got {neu_tv}"
    assert neu_tv < lin_tv - 0.04
    assert neu.converged_


def test_neural_ufxp_linear_dgp_sanity():
    """On a linear-reward DGP the neural utility still recovers behavior.

    The objective constrains the choice-relevant utility, not the utility level,
    so the learned function is identified only up to behavior-preserving
    transformations. Behavior recovery is the contract; the feature projection
    is a gauge-confounded best-effort summary and is not asserted here.
    """
    from econirl import NeuralUFXP

    env = random_mdp(num_states=12, num_actions=2, num_features=2,
                     branching=3, discount_factor=0.9, seed=7)
    oracle_policy, _ = _oracle(env)
    panel = simulate_panel(env, n_individuals=400, n_periods=60, seed=11)

    m = NeuralUFXP(discount=0.9, num_projections=32, max_epochs=2000,
                   lr=1e-2, seed=0).fit(
        panel, features=np.asarray(env.feature_matrix),
        transitions=np.asarray(env.transition_matrices))

    assert _policy_tv(m.policy_, oracle_policy) < 0.05
    assert np.isfinite(m.projection_r2_)


def test_neural_ufxp_sklearn_attributes():
    """The public class exposes the neural-estimator attribute surface."""
    from econirl import NeuralUFXP

    env = random_mdp(num_states=10, num_actions=2, num_features=2, seed=3)
    panel = simulate_panel(env, n_individuals=200, n_periods=40, seed=5)
    m = NeuralUFXP(discount=0.9, num_projections=16, max_epochs=500, seed=0).fit(
        panel, features=np.asarray(env.feature_matrix),
        transitions=np.asarray(env.transition_matrices))

    assert m.policy_.shape == (10, 2)
    assert m.value_.shape == (10,)
    assert m.reward_.shape == (10, 2)
    assert isinstance(m.params_, dict) and len(m.params_) == 2
    assert isinstance(m.se_, dict)
    assert m.converged_ is True


def test_neural_ufxp_requires_features_and_transitions():
    from econirl import NeuralUFXP

    env = random_mdp(num_states=8, num_actions=2, num_features=2, seed=1)
    panel = simulate_panel(env, n_individuals=50, n_periods=20, seed=2)
    with pytest.raises(ValueError):
        NeuralUFXP().fit(panel, transitions=np.asarray(env.transition_matrices))
    with pytest.raises(ValueError):
        NeuralUFXP().fit(panel, features=np.asarray(env.feature_matrix))
