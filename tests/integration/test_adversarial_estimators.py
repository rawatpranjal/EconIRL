"""Integration tests for adversarial IRL estimators."""

import pytest
import jax.numpy as jnp

from econirl.core.types import DDCProblem, Panel
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation import (
    GAILEstimator,
    GAILConfig,
    AIRLEstimator,
    AIRLConfig,
    MCEIRLEstimator,
    MCEIRLConfig,
)
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation import simulate_panel


@pytest.fixture
def rust_env():
    """Create Rust bus environment."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        discount_factor=0.9,
    )


@pytest.fixture
def medium_panel(rust_env):
    """Generate medium-sized panel from Rust bus environment."""
    return simulate_panel(
        rust_env,
        n_individuals=50,
        n_periods=20,
        seed=42,
    )


class TestAdversarialOnRustBus:
    """Test adversarial methods on Rust bus problem."""

    def test_gail_on_rust_bus(self, rust_env, medium_panel):
        """GAIL should learn reasonable policy on Rust bus."""
        reward_fn = ActionDependentReward.from_rust_environment(rust_env)

        config = GAILConfig(max_rounds=30, verbose=False)
        estimator = GAILEstimator(config=config)

        result = estimator.estimate(
            panel=medium_panel,
            utility=reward_fn,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        # Policy should be valid distribution
        n_states = rust_env.problem_spec.num_states
        assert jnp.allclose(result.policy.sum(axis=1), jnp.ones(n_states), atol=1e-5)

        # Policy should favor maintenance (action 0) at low mileage
        # and replacement (action 1) at high mileage
        assert result.policy[0, 0] > 0.5  # Low mileage: maintain
        assert result.policy[-1, 1] > result.policy[0, 1]  # Higher replace prob at high mileage

    def test_airl_on_rust_bus(self, rust_env, medium_panel):
        """AIRL should learn reasonable reward on Rust bus."""
        reward_fn = ActionDependentReward.from_rust_environment(rust_env)

        config = AIRLConfig(max_rounds=30, verbose=False)
        estimator = AIRLEstimator(config=config)

        result = estimator.estimate(
            panel=medium_panel,
            utility=reward_fn,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        n_states = rust_env.problem_spec.num_states
        assert jnp.allclose(result.policy.sum(axis=1), jnp.ones(n_states), atol=1e-5)

    @pytest.mark.slow
    def test_gail_vs_mce_irl_similar_policy(self, rust_env, medium_panel):
        """GAIL and MCE IRL should produce similar policies."""
        reward_fn = ActionDependentReward.from_rust_environment(rust_env)

        # MCE IRL (reduced iterations for faster test)
        mce_config = MCEIRLConfig(outer_max_iter=20, verbose=False)
        mce_estimator = MCEIRLEstimator(config=mce_config)
        mce_result = mce_estimator.estimate(
            panel=medium_panel,
            utility=reward_fn,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        # GAIL (reduced iterations for faster test)
        gail_config = GAILConfig(max_rounds=20, verbose=False)
        gail_estimator = GAILEstimator(config=gail_config)
        gail_result = gail_estimator.estimate(
            panel=medium_panel,
            utility=reward_fn,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        # Policies should be similar (not identical due to different algorithms)
        # Check that they agree on action preference in most states
        mce_preferred = mce_result.policy.argmax(axis=1)
        gail_preferred = gail_result.policy.argmax(axis=1)
        agreement = (mce_preferred == gail_preferred).astype(jnp.float32).mean()

        # Should agree on at least 60% of states (relaxed due to reduced iterations)
        assert agreement > 0.6, f"Agreement was only {float(agreement):.2f}"
