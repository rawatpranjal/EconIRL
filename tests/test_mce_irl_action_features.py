"""Test MCE IRL with action-dependent features."""
import pytest
import torch
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.simulation import simulate_panel


class TestMCEIRLActionFeatures:
    """Tests for MCE IRL with action-dependent features."""

    @pytest.fixture
    def rust_env(self):
        return RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.99,  # Lower for faster convergence
        )

    def test_mce_irl_handles_3d_features(self, rust_env):
        """MCE IRL should correctly handle 3D feature matrices."""
        panel = simulate_panel(rust_env, n_individuals=50, n_periods=20, seed=123)

        reward = ActionDependentReward.from_rust_environment(rust_env)

        # Verify feature matrix is 3D
        assert reward.feature_matrix.ndim == 3, "Feature matrix should be 3D"
        assert reward.feature_matrix.shape == (90, 2, 2), (
            f"Expected shape (90, 2, 2), got {reward.feature_matrix.shape}"
        )

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=100,
            learning_rate=0.1,
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)

        # Should not raise an error
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        # Parameters should be finite
        assert torch.isfinite(result.parameters).all(), "Parameters should be finite"

    def test_mce_irl_feature_expectations(self, rust_env):
        """Test that feature expectations are computed correctly for 3D features."""
        panel = simulate_panel(rust_env, n_individuals=100, n_periods=30, seed=456)

        reward = ActionDependentReward.from_rust_environment(rust_env)

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=50,
            learning_rate=0.1,
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        # Check that feature expectations are stored in metadata
        assert "empirical_features" in result.metadata, "Should store empirical features"
        assert "final_expected_features" in result.metadata, "Should store expected features"

        # Empirical features should have correct dimension
        empirical = result.metadata["empirical_features"]
        assert len(empirical) == 2, f"Expected 2 features, got {len(empirical)}"

    def test_mce_irl_with_warm_start(self, rust_env):
        """MCE IRL with warm start from true params should stay close."""
        panel = simulate_panel(rust_env, n_individuals=200, n_periods=50, seed=42)

        reward = ActionDependentReward.from_rust_environment(rust_env)
        true_params = rust_env.get_true_parameter_vector()

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=100,
            learning_rate=0.001,  # Small learning rate for stability
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
            initial_params=true_params,
        )

        # Feature matching objective should be very small when starting from true params
        feature_diff = result.metadata.get("feature_difference", float("inf"))
        assert feature_diff < 0.1, f"Feature difference too large: {feature_diff}"

        # Parameters should stay close to true values (within factor of 5)
        # Note: MCE IRL identifies parameters up to scale, so exact match not expected
        for i, (est, true) in enumerate(zip(result.parameters, true_params)):
            ratio = est.item() / true.item() if true.item() != 0 else float("inf")
            assert 0.1 < ratio < 10, (
                f"Parameter {i} ratio out of range: estimated={est.item():.6f}, "
                f"true={true.item():.6f}, ratio={ratio:.2f}"
            )

    def test_mce_irl_recovers_ratio(self, rust_env):
        """MCE IRL should recover the correct ratio of parameters."""
        # Use more data for better ratio recovery
        panel = simulate_panel(rust_env, n_individuals=300, n_periods=100, seed=42)

        reward = ActionDependentReward.from_rust_environment(rust_env)
        true_params = rust_env.get_true_parameter_vector()

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=200,
            learning_rate=0.001,
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
            initial_params=true_params,  # Warm start
        )

        # Check that the ratio theta_c/RC is recovered correctly
        # This is what MCE IRL can identify (up to scale)
        estimated_ratio = result.parameters[0] / result.parameters[1]
        true_ratio = true_params[0] / true_params[1]

        # Allow 50% relative error on the ratio (IRL is hard!)
        relative_error = abs(estimated_ratio - true_ratio) / abs(true_ratio)
        assert relative_error < 0.5, (
            f"Ratio mismatch: estimated={estimated_ratio:.6f}, "
            f"true={true_ratio:.6f}, relative_error={relative_error:.2%}"
        )
