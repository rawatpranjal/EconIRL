"""Smoke tests for Deep MaxEnt IRL (neural reward network)."""

import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.contrib.deep_maxent_irl import DeepMaxEntIRLEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


@pytest.fixture
def small_deep_maxent_setup():
    """Small environment for quick Deep MaxEnt IRL tests."""
    env = RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        seed=42,
    )
    panel = simulate_panel(env, n_individuals=50, n_periods=50, seed=123)
    utility = LinearUtility.from_environment(env)
    return env, panel, utility


def test_deep_maxent_init():
    """DeepMaxEntIRLEstimator can be instantiated without error."""
    estimator = DeepMaxEntIRLEstimator(
        hidden_dims=[16], max_epochs=5, verbose=False, seed=42,
    )
    assert estimator is not None


def test_deep_maxent_estimate(small_deep_maxent_setup):
    """Deep MaxEnt IRL runs on a small problem and returns a result with a policy."""
    env, panel, utility = small_deep_maxent_setup
    estimator = DeepMaxEntIRLEstimator(
        hidden_dims=[16],
        max_epochs=30,
        lr=1e-3,
        verbose=False,
        seed=42,
    )
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    assert result.policy is not None
    assert result.policy.shape == (20, 2)


def test_deep_maxent_reward_finite(small_deep_maxent_setup):
    """Learned reward matrix should have no NaN or Inf values."""
    env, panel, utility = small_deep_maxent_setup
    estimator = DeepMaxEntIRLEstimator(
        hidden_dims=[16],
        max_epochs=30,
        lr=1e-3,
        verbose=False,
        seed=42,
    )
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    reward = result.metadata.get("reward_matrix", result.parameters)
    assert jnp.all(jnp.isfinite(reward)), "Reward matrix contains NaN or Inf"


def test_deep_maxent_policy_valid(small_deep_maxent_setup):
    """Deep MaxEnt IRL policy should be a valid probability distribution."""
    env, panel, utility = small_deep_maxent_setup
    estimator = DeepMaxEntIRLEstimator(
        hidden_dims=[16],
        max_epochs=30,
        lr=1e-3,
        verbose=False,
        seed=42,
    )
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    policy = result.policy
    # Non-negative
    assert float(policy.min()) >= 0.0, "Policy has negative probabilities"
    # Rows sum to 1
    row_sums = policy.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)
