"""Smoke tests for Bayesian IRL (MCMC posterior over rewards)."""

import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.contrib.bayesian_irl import BayesianIRLEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


@pytest.fixture
def small_bayesian_setup():
    """Small environment for quick Bayesian IRL tests."""
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


def test_bayesian_irl_init():
    """BayesianIRLEstimator can be instantiated without error."""
    estimator = BayesianIRLEstimator(
        n_samples=10, burnin=5, verbose=False, seed=42,
    )
    assert estimator is not None


def test_bayesian_irl_estimate(small_bayesian_setup):
    """Bayesian IRL runs on a small problem and returns a result with a policy."""
    env, panel, utility = small_bayesian_setup
    estimator = BayesianIRLEstimator(
        n_samples=50,
        burnin=10,
        proposal_sigma=0.1,
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


def test_bayesian_irl_posterior_finite(small_bayesian_setup):
    """All posterior samples should be finite."""
    env, panel, utility = small_bayesian_setup
    estimator = BayesianIRLEstimator(
        n_samples=50,
        burnin=10,
        proposal_sigma=0.1,
        verbose=False,
        seed=42,
    )
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    samples = result.metadata.get("samples")
    assert samples is not None, "Metadata missing 'samples' key"
    assert np.all(np.isfinite(samples)), "Posterior samples contain NaN or Inf"


def test_bayesian_irl_policy_valid(small_bayesian_setup):
    """Bayesian IRL policy should be a valid probability distribution."""
    env, panel, utility = small_bayesian_setup
    estimator = BayesianIRLEstimator(
        n_samples=50,
        burnin=10,
        proposal_sigma=0.1,
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
