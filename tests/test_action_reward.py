"""Tests for action-dependent reward functions."""
import pytest
import jax.numpy as jnp
import numpy as np
from econirl.preferences.action_reward import ActionDependentReward


class TestActionDependentReward:
    """Tests for ActionDependentReward class."""

    def test_init(self):
        """ActionDependentReward initializes correctly."""
        n_states = 90
        n_actions = 2
        n_features = 2

        # Feature matrix: (states, actions, features)
        features = jnp.zeros((n_states, n_actions, n_features))
        for s in range(n_states):
            features = features.at[s, 0, 0].set(-s)  # Keep: operating cost
            features = features.at[s, 1, 1].set(-1)  # Replace: fixed cost

        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["theta_c", "RC"],
        )

        assert reward.num_states == n_states
        assert reward.num_actions == n_actions
        assert reward.num_parameters == n_features

    def test_compute_rust_utility(self):
        """Compute action-dependent reward for Rust model."""
        n_states = 90
        features = jnp.zeros((n_states, 2, 2))
        for s in range(n_states):
            features = features.at[s, 0, 0].set(-s)
            features = features.at[s, 1, 1].set(-1)

        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["theta_c", "RC"],
        )

        params = jnp.array([0.001, 3.0])
        R = reward.compute(params)

        assert R.shape == (n_states, 2)

        # Check values match Rust model
        assert jnp.isclose(R[0, 0], jnp.array(0.0), atol=1e-6)  # U(0, keep)
        assert jnp.isclose(R[0, 1], jnp.array(-3.0), atol=1e-6)  # U(0, replace)
        assert jnp.isclose(R[5, 0], jnp.array(-0.005), atol=1e-6)  # U(5, keep)
        assert jnp.isclose(R[5, 1], jnp.array(-3.0), atol=1e-6)  # U(5, replace)

    def test_compute_gradient(self):
        """Gradient is the feature matrix."""
        n_states = 10
        n_actions = 2
        n_features = 3

        np.random.seed(0)
        features = jnp.array(np.random.randn(n_states, n_actions, n_features))
        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["a", "b", "c"],
        )

        np.random.seed(1)
        params = jnp.array(np.random.randn(n_features))
        grad = reward.compute_gradient(params)

        assert grad.shape == (n_states, n_actions, n_features)
        # For linear reward, gradient is the feature matrix
        np.testing.assert_allclose(np.asarray(grad), np.asarray(features))

    def test_invalid_feature_matrix_shape(self):
        """Reject non-3D feature matrices."""
        with pytest.raises(ValueError, match="must be 3D"):
            ActionDependentReward(
                feature_matrix=jnp.zeros((10, 5)),  # 2D, should be 3D
                parameter_names=["a", "b", "c", "d", "e"],
            )

    def test_parameter_names_length_mismatch(self):
        """Reject mismatched parameter names length."""
        with pytest.raises(ValueError, match="parameter_names length"):
            ActionDependentReward(
                feature_matrix=jnp.zeros((10, 2, 3)),
                parameter_names=["a", "b"],  # Should have 3 names
            )

    def test_from_rust_environment(self):
        """Test factory method from_rust_environment."""
        # Create a mock environment
        class MockEnv:
            feature_matrix = jnp.zeros((90, 2, 2))
            parameter_names = ["theta_c", "RC"]

            def __init__(self):
                for s in range(90):
                    self.feature_matrix = self.feature_matrix.at[s, 0, 0].set(-s)
                    self.feature_matrix = self.feature_matrix.at[s, 1, 1].set(-1)

        env = MockEnv()
        reward = ActionDependentReward.from_rust_environment(env)

        assert reward.num_states == 90
        assert reward.num_actions == 2
        assert reward.parameter_names == ["theta_c", "RC"]

    def test_parameter_validation(self):
        """Validate parameter shape."""
        features = jnp.zeros((10, 2, 3))
        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["a", "b", "c"],
        )

        # Wrong shape should raise
        with pytest.raises(ValueError, match="Expected parameters"):
            reward.compute(jnp.zeros(2))  # Should be 3

    def test_get_initial_parameters(self):
        """Initial parameters are zeros."""
        features = jnp.zeros((10, 2, 3))
        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["a", "b", "c"],
        )

        initial = reward.get_initial_parameters()
        assert initial.shape == (3,)
        np.testing.assert_allclose(np.asarray(initial), np.asarray(jnp.zeros(3)))

    def test_repr(self):
        """String representation is informative."""
        features = jnp.zeros((10, 2, 3))
        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["a", "b", "c"],
        )

        repr_str = repr(reward)
        assert "ActionDependentReward" in repr_str
        assert "num_states=10" in repr_str
        assert "num_actions=2" in repr_str
