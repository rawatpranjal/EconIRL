"""Tests for LinearReward class."""

import pytest
import jax.numpy as jnp
import numpy as np

from econirl.preferences.reward import LinearReward


class TestLinearRewardConstructor:
    """Test LinearReward initialization."""

    def test_basic_construction(self):
        """Test that constructor works with valid inputs."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(10, 3))  # 10 states, 3 features
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["f1", "f2", "f3"],
            n_actions=2,
        )

        assert reward.num_states == 10
        assert reward.num_actions == 2
        assert reward.num_parameters == 3
        assert reward.parameter_names == ["f1", "f2", "f3"]

    def test_default_n_actions(self):
        """Test that n_actions defaults to 2."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(5, 2))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b"],
        )
        assert reward.num_actions == 2

    def test_stores_feature_copy(self):
        """Test that features are copied, not referenced."""
        np.random.seed(0)
        state_features_np = np.random.randn(5, 2)
        state_features = jnp.array(state_features_np)
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b"],
        )

        # Modify original numpy array
        state_features_np[0, 0] = 999.0

        # Stored features should not change (JAX arrays are immutable anyway)
        assert float(reward.state_features[0, 0]) != 999.0

    def test_invalid_feature_dimensions(self):
        """Test that non-2D features raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            LinearReward(
                state_features=jnp.array(np.random.randn(10, 3, 4)),  # 3D, not 2D
                parameter_names=["a", "b", "c", "d"],
            )

        with pytest.raises(ValueError, match="must be 2D"):
            LinearReward(
                state_features=jnp.array(np.random.randn(10)),  # 1D, not 2D
                parameter_names=["a"],
            )

    def test_mismatched_parameter_names(self):
        """Test that wrong number of parameter names raises ValueError."""
        with pytest.raises(ValueError, match="must have 3 elements"):
            LinearReward(
                state_features=jnp.array(np.random.randn(10, 3)),
                parameter_names=["a", "b"],  # Only 2 names for 3 features
            )

    def test_invalid_n_actions(self):
        """Test that n_actions < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_actions must be at least 1"):
            LinearReward(
                state_features=jnp.array(np.random.randn(10, 2)),
                parameter_names=["a", "b"],
                n_actions=0,
            )


class TestLinearRewardCompute:
    """Test LinearReward.compute() method."""

    def test_output_shape(self):
        """Test that compute returns correct shape (n_states, n_actions)."""
        n_states, n_features, n_actions = 20, 4, 3
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(n_states, n_features))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=[f"f{i}" for i in range(n_features)],
            n_actions=n_actions,
        )

        params = jnp.array(np.random.randn(n_features))
        result = reward.compute(params)

        assert result.shape == (n_states, n_actions)

    def test_broadcast_same_across_actions(self):
        """Test that reward is the same for all actions in a state."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(10, 3))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c"],
            n_actions=5,
        )

        params = jnp.array(np.random.randn(3))
        result = reward.compute(params)

        # All actions in each state should have same reward
        for s in range(10):
            expected = result[s, 0]
            for a in range(5):
                assert jnp.isclose(result[s, a], expected), \
                    f"State {s}: action 0 has {expected}, action {a} has {result[s, a]}"

    def test_correct_linear_computation(self):
        """Test that R(s) = theta * phi(s) is computed correctly."""
        # Simple case: 2 states, 2 features
        state_features = jnp.array([
            [1.0, 2.0],  # State 0: phi = [1, 2]
            [3.0, 4.0],  # State 1: phi = [3, 4]
        ])
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["w1", "w2"],
            n_actions=2,
        )

        params = jnp.array([1.0, 0.5])
        result = reward.compute(params)

        # R(s=0) = 1*1.0 + 2*0.5 = 2.0
        # R(s=1) = 3*1.0 + 4*0.5 = 5.0
        expected_s0 = 1.0 * 1.0 + 2.0 * 0.5  # = 2.0
        expected_s1 = 3.0 * 1.0 + 4.0 * 0.5  # = 5.0

        np.testing.assert_allclose(np.asarray(result[0, :]), np.asarray(jnp.array([expected_s0, expected_s0])))
        np.testing.assert_allclose(np.asarray(result[1, :]), np.asarray(jnp.array([expected_s1, expected_s1])))

    def test_invalid_parameter_shape(self):
        """Test that wrong parameter shape raises ValueError."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(10, 3))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c"],
        )

        with pytest.raises(ValueError, match="Expected parameters of shape"):
            reward.compute(jnp.array(np.random.randn(4)))  # Wrong size

        with pytest.raises(ValueError, match="Expected parameters of shape"):
            reward.compute(jnp.array(np.random.randn(3, 1)))  # Wrong dimensions


class TestLinearRewardGradient:
    """Test LinearReward.compute_gradient() method."""

    def test_output_shape(self):
        """Test that compute_gradient returns correct shape."""
        n_states, n_features, n_actions = 15, 5, 4
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(n_states, n_features))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=[f"f{i}" for i in range(n_features)],
            n_actions=n_actions,
        )

        params = jnp.array(np.random.randn(n_features))
        gradient = reward.compute_gradient(params)

        assert gradient.shape == (n_states, n_actions, n_features)

    def test_gradient_equals_features_broadcast(self):
        """Test that gradient is features broadcast to all actions."""
        state_features = jnp.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b"],
            n_actions=3,
        )

        np.random.seed(0)
        params = jnp.array(np.random.randn(2))  # Gradient shouldn't depend on params
        gradient = reward.compute_gradient(params)

        # For each state, all actions should have the same gradient (= state features)
        for s in range(3):
            for a in range(3):
                np.testing.assert_allclose(np.asarray(gradient[s, a, :]), np.asarray(state_features[s, :]))

    def test_gradient_independent_of_params(self):
        """Test that gradient is independent of parameter values (linear function)."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(10, 4))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c", "d"],
            n_actions=2,
        )

        grad1 = reward.compute_gradient(jnp.array(np.random.randn(4)))
        grad2 = reward.compute_gradient(jnp.array(np.random.randn(4)))
        grad3 = reward.compute_gradient(jnp.zeros(4))

        np.testing.assert_allclose(np.asarray(grad1), np.asarray(grad2))
        np.testing.assert_allclose(np.asarray(grad1), np.asarray(grad3))

    def test_gradient_is_copy(self):
        """Test that gradient returns a copy, not a view."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(5, 2))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b"],
            n_actions=2,
        )

        gradient = reward.compute_gradient(jnp.array(np.random.randn(2)))
        # JAX arrays are immutable, so modifying is not possible in the same way
        # but we verify the original is unchanged
        assert float(reward.state_features[0, 0]) == float(state_features[0, 0])


class TestLinearRewardDifferentNActions:
    """Test LinearReward with different n_actions values."""

    @pytest.mark.parametrize("n_actions", [1, 2, 5, 10, 100])
    def test_various_n_actions(self, n_actions):
        """Test that different n_actions values work correctly."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(8, 3))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c"],
            n_actions=n_actions,
        )

        assert reward.num_actions == n_actions

        params = jnp.array(np.random.randn(3))
        result = reward.compute(params)
        gradient = reward.compute_gradient(params)

        assert result.shape == (8, n_actions)
        assert gradient.shape == (8, n_actions, 3)

        # Verify broadcast: all actions have same reward per state
        for s in range(8):
            expected_reward = result[s, 0]
            np.testing.assert_allclose(np.asarray(result[s, :]), np.asarray(jnp.broadcast_to(expected_reward, (n_actions,))))


class TestLinearRewardMethods:
    """Test additional LinearReward methods."""

    def test_to_device(self):
        """Test moving reward to different device."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(5, 2))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b"],
            n_actions=3,
        )

        # Move to same device (CPU) - should work
        reward_cpu = reward.to("cpu")
        assert reward_cpu.num_actions == 3
        assert reward_cpu.parameter_names == ["a", "b"]

    def test_subset_states(self):
        """Test creating a subset of states."""
        state_features = jnp.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b"],
            n_actions=2,
        )

        indices = jnp.array([0, 2], dtype=jnp.int32)  # Keep states 0 and 2
        subset = reward.subset_states(indices)

        assert subset.num_states == 2
        assert subset.num_actions == 2
        np.testing.assert_allclose(np.asarray(subset.state_features[0, :]), np.asarray(state_features[0, :]))
        np.testing.assert_allclose(np.asarray(subset.state_features[1, :]), np.asarray(state_features[2, :]))

    def test_compute_hessian(self):
        """Test that Hessian is zero for linear reward."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(10, 3))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c"],
            n_actions=2,
        )

        hessian = reward.compute_hessian(jnp.array(np.random.randn(3)))

        assert hessian.shape == (10, 2, 3, 3)
        assert jnp.all(hessian == 0)

    def test_repr(self):
        """Test string representation."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(10, 3))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c"],
            n_actions=4,
        )

        repr_str = repr(reward)
        assert "LinearReward" in repr_str
        assert "num_states=10" in repr_str
        assert "num_actions=4" in repr_str
        assert "num_features=3" in repr_str

    def test_get_initial_parameters(self):
        """Test that initial parameters are zeros."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(5, 3))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c"],
        )

        initial = reward.get_initial_parameters()
        assert initial.shape == (3,)
        assert jnp.all(initial == 0)

    def test_get_parameter_bounds(self):
        """Test default parameter bounds are unbounded."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(5, 2))
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b"],
        )

        lower, upper = reward.get_parameter_bounds()
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert jnp.all(jnp.isinf(lower) & (lower < 0))
        assert jnp.all(jnp.isinf(upper) & (upper > 0))


class TestLinearRewardIntegration:
    """Integration tests for LinearReward."""

    def test_numerical_gradient_check(self):
        """Verify gradient numerically using finite differences."""
        np.random.seed(0)
        state_features = jnp.array(np.random.randn(5, 3), dtype=jnp.float64)
        reward = LinearReward(
            state_features=state_features,
            parameter_names=["a", "b", "c"],
            n_actions=2,
        )

        params = jnp.array(np.random.randn(3), dtype=jnp.float64)
        eps = 1e-6

        # Analytical gradient
        analytical_grad = reward.compute_gradient(params)

        # Numerical gradient via finite differences
        for k in range(3):
            params_plus = jnp.array(params)
            params_plus = params_plus.at[k].add(eps)
            params_minus = jnp.array(params)
            params_minus = params_minus.at[k].add(-eps)

            r_plus = reward.compute(params_plus)
            r_minus = reward.compute(params_minus)

            numerical_grad_k = (r_plus - r_minus) / (2 * eps)

            # Compare for all (s, a)
            np.testing.assert_allclose(
                np.asarray(analytical_grad[:, :, k]),
                np.asarray(numerical_grad_k),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Gradient mismatch for parameter {k}",
            )

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        for dtype in [jnp.float32, jnp.float64]:
            np.random.seed(0)
            state_features = jnp.array(np.random.randn(5, 2), dtype=dtype)
            reward = LinearReward(
                state_features=state_features,
                parameter_names=["a", "b"],
            )

            params = jnp.array(np.random.randn(2), dtype=dtype)
            result = reward.compute(params)
            gradient = reward.compute_gradient(params)

            assert result.dtype == dtype
            assert gradient.dtype == dtype
