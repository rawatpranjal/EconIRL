"""Tests for ActionDependentUtility class.

Tests cover:
1. Initialization and basic properties
2. Rust (1987) utility computation
3. Gradient computation
4. Parameter bounds and initial values
5. Factory methods (for_rust_model, from_environment)
6. Edge cases and validation
"""

import pytest
import jax.numpy as jnp
import numpy as np

from econirl.preferences.action_utility import ActionDependentUtility
from econirl.preferences.base import UtilityFunction


class TestActionDependentUtilityInit:
    """Tests for ActionDependentUtility initialization."""

    def test_basic_init(self):
        """Test basic initialization with required parameters."""
        utility = ActionDependentUtility(
            num_states=10,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
        )

        assert utility.num_states == 10
        assert utility.num_actions == 2
        assert utility.num_parameters == 2
        assert utility.parameter_names == ["theta_c", "RC"]

    def test_state_scale_default(self):
        """Test that state_scale defaults to 1.0."""
        utility = ActionDependentUtility(
            num_states=10,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
        )
        assert utility.state_scale == 1.0

    def test_state_scale_custom(self):
        """Test custom state_scale."""
        utility = ActionDependentUtility(
            num_states=10,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
            state_scale=10.0,
        )
        assert utility.state_scale == 10.0

    def test_implements_protocol(self):
        """Test that ActionDependentUtility implements UtilityFunction protocol."""
        utility = ActionDependentUtility(
            num_states=10,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
        )
        assert isinstance(utility, UtilityFunction)

    def test_repr(self):
        """Test string representation."""
        utility = ActionDependentUtility(
            num_states=10,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
            state_scale=10.0,
        )
        repr_str = repr(utility)

        assert "ActionDependentUtility" in repr_str
        assert "num_states=10" in repr_str
        assert "num_actions=2" in repr_str
        assert "state_scale=10.0" in repr_str


class TestRustUtilityComputation:
    """Tests for Rust (1987) utility computation."""

    @pytest.fixture
    def rust_utility(self):
        """Create utility for Rust model with 10 states."""
        return ActionDependentUtility.for_rust_model(num_states=10)

    def test_utility_shape(self, rust_utility):
        """Test that utility has correct shape."""
        params = jnp.array([0.001, 3.0])
        U = rust_utility.compute(params)

        assert U.shape == (10, 2)

    def test_keep_action_utility(self, rust_utility):
        """Test that keep action utility is -theta_c * s / scale."""
        theta_c = 0.01
        params = jnp.array([theta_c, 3.0])
        U = rust_utility.compute(params)

        # U[:, 0] should be -theta_c * s / 10 (scale = num_states = 10)
        expected_keep = -theta_c * jnp.arange(10, dtype=jnp.float32) / 10.0
        np.testing.assert_allclose(np.asarray(U[:, 0]), np.asarray(expected_keep))

    def test_replace_action_utility(self, rust_utility):
        """Test that replace action utility is -RC."""
        rc = 3.0
        params = jnp.array([0.01, rc])
        U = rust_utility.compute(params)

        # U[:, 1] should be -RC for all states
        expected_replace = jnp.full((10,), -rc)
        np.testing.assert_allclose(np.asarray(U[:, 1]), np.asarray(expected_replace))

    def test_utility_at_state_zero(self, rust_utility):
        """Test utility at state 0 (new engine)."""
        params = jnp.array([0.001, 3.0])
        U = rust_utility.compute(params)

        # At s=0: keep utility = 0, replace utility = -3.0
        assert float(U[0, 0]) == 0.0
        assert float(U[0, 1]) == pytest.approx(-3.0)

    def test_utility_at_high_mileage(self, rust_utility):
        """Test utility at high mileage state."""
        params = jnp.array([0.1, 3.0])  # Higher operating cost
        U = rust_utility.compute(params)

        # At s=9: keep utility should be negative
        # U[9, keep] = -0.1 * 9 / 10 = -0.09
        assert float(U[9, 0]) == pytest.approx(-0.09)

    def test_utility_scales_with_theta_c(self, rust_utility):
        """Test that keep utility scales linearly with theta_c."""
        params1 = jnp.array([0.01, 3.0])
        params2 = jnp.array([0.02, 3.0])  # Double theta_c

        U1 = rust_utility.compute(params1)
        U2 = rust_utility.compute(params2)

        # Keep utility should double
        np.testing.assert_allclose(np.asarray(U2[:, 0]), np.asarray(2 * U1[:, 0]))

    def test_utility_scales_with_rc(self, rust_utility):
        """Test that replace utility scales linearly with RC."""
        params1 = jnp.array([0.01, 3.0])
        params2 = jnp.array([0.01, 6.0])  # Double RC

        U1 = rust_utility.compute(params1)
        U2 = rust_utility.compute(params2)

        # Replace utility should double (in magnitude)
        np.testing.assert_allclose(np.asarray(U2[:, 1]), np.asarray(2 * U1[:, 1]))


class TestGradientComputation:
    """Tests for gradient computation."""

    @pytest.fixture
    def rust_utility(self):
        """Create utility for Rust model with 10 states."""
        return ActionDependentUtility.for_rust_model(num_states=10)

    def test_gradient_shape(self, rust_utility):
        """Test that gradient has correct shape."""
        params = jnp.array([0.001, 3.0])
        grad = rust_utility.compute_gradient(params)

        # Shape: (num_states, num_actions, num_parameters)
        assert grad.shape == (10, 2, 2)

    def test_gradient_theta_c_keep(self, rust_utility):
        """Test dU/d(theta_c) for keep action."""
        params = jnp.array([0.001, 3.0])
        grad = rust_utility.compute_gradient(params)

        # dU[:, keep]/d(theta_c) = -s / scale
        expected = -jnp.arange(10, dtype=jnp.float32) / 10.0
        np.testing.assert_allclose(np.asarray(grad[:, 0, 0]), np.asarray(expected))

    def test_gradient_theta_c_replace(self, rust_utility):
        """Test dU/d(theta_c) for replace action (should be 0)."""
        params = jnp.array([0.001, 3.0])
        grad = rust_utility.compute_gradient(params)

        # dU[:, replace]/d(theta_c) = 0
        np.testing.assert_allclose(np.asarray(grad[:, 1, 0]), np.asarray(jnp.zeros(10)))

    def test_gradient_rc_keep(self, rust_utility):
        """Test dU/d(RC) for keep action (should be 0)."""
        params = jnp.array([0.001, 3.0])
        grad = rust_utility.compute_gradient(params)

        # dU[:, keep]/d(RC) = 0
        np.testing.assert_allclose(np.asarray(grad[:, 0, 1]), np.asarray(jnp.zeros(10)))

    def test_gradient_rc_replace(self, rust_utility):
        """Test dU/d(RC) for replace action."""
        params = jnp.array([0.001, 3.0])
        grad = rust_utility.compute_gradient(params)

        # dU[:, replace]/d(RC) = -1
        np.testing.assert_allclose(np.asarray(grad[:, 1, 1]), np.asarray(jnp.full((10,), -1.0)))

    def test_gradient_numerical(self, rust_utility):
        """Test gradient against numerical approximation.

        Uses central difference formula: df/dx approx (f(x+h) - f(x-h)) / (2h)
        """
        params = jnp.array([0.01, 3.0])
        analytical_grad = rust_utility.compute_gradient(params)

        # Compute numerical gradient
        eps = 1e-4
        numerical_grad = jnp.zeros_like(analytical_grad)

        for k in range(rust_utility.num_parameters):
            params_plus = jnp.array(params)
            params_minus = jnp.array(params)
            params_plus = params_plus.at[k].set(params[k] + eps)
            params_minus = params_minus.at[k].set(params[k] - eps)

            U_plus = rust_utility.compute(params_plus)
            U_minus = rust_utility.compute(params_minus)

            numerical_grad = numerical_grad.at[:, :, k].set((U_plus - U_minus) / (2 * eps))

        # Check element by element for diagnostics
        # dU/d(theta_c) for keep action should be -s/scale
        np.testing.assert_allclose(np.asarray(analytical_grad[:, 0, 0]), np.asarray(numerical_grad[:, 0, 0]), atol=1e-2)
        # dU/d(theta_c) for replace action should be 0
        np.testing.assert_allclose(np.asarray(analytical_grad[:, 1, 0]), np.asarray(numerical_grad[:, 1, 0]), atol=1e-2)
        # dU/d(RC) for keep action should be 0
        np.testing.assert_allclose(np.asarray(analytical_grad[:, 0, 1]), np.asarray(numerical_grad[:, 0, 1]), atol=1e-2)
        # dU/d(RC) for replace action should be -1
        np.testing.assert_allclose(np.asarray(analytical_grad[:, 1, 1]), np.asarray(numerical_grad[:, 1, 1]), atol=1e-2)


class TestHessianComputation:
    """Tests for Hessian computation."""

    def test_hessian_shape(self):
        """Test that Hessian has correct shape."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        params = jnp.array([0.001, 3.0])
        hessian = utility.compute_hessian(params)

        # Shape: (num_states, num_actions, num_parameters, num_parameters)
        assert hessian.shape == (10, 2, 2, 2)

    def test_hessian_is_zero(self):
        """Test that Hessian is zero for linear utility."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        params = jnp.array([0.001, 3.0])
        hessian = utility.compute_hessian(params)

        # Linear utility has zero Hessian
        np.testing.assert_allclose(np.asarray(hessian), np.asarray(jnp.zeros_like(hessian)))


class TestParameterBoundsAndInitialValues:
    """Tests for parameter bounds and initial values."""

    def test_initial_parameters_shape(self):
        """Test that initial parameters have correct shape."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        init_params = utility.get_initial_parameters()

        assert init_params.shape == (2,)

    def test_initial_parameters_reasonable(self):
        """Test that initial parameters are reasonable for Rust model."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        init_params = utility.get_initial_parameters()

        # theta_c should be small positive
        assert float(init_params[0]) > 0
        assert float(init_params[0]) < 1.0

        # RC should be moderate positive
        assert float(init_params[1]) > 0
        assert float(init_params[1]) < 10.0

    def test_parameter_bounds_shape(self):
        """Test that parameter bounds have correct shape."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        lower, upper = utility.get_parameter_bounds()

        assert lower.shape == (2,)
        assert upper.shape == (2,)

    def test_parameter_bounds_non_negative(self):
        """Test that Rust model parameters are bounded to be non-negative."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        lower, upper = utility.get_parameter_bounds()

        # Lower bounds should be 0 (non-negative)
        assert float(lower[0]) == 0.0
        assert float(lower[1]) == 0.0

        # Upper bounds should be infinity
        assert float(upper[0]) == float("inf")
        assert float(upper[1]) == float("inf")

    def test_initial_within_bounds(self):
        """Test that initial parameters are within bounds."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        init_params = utility.get_initial_parameters()
        lower, upper = utility.get_parameter_bounds()

        assert (init_params >= lower).all()
        assert (init_params <= upper).all()


class TestFactoryMethods:
    """Tests for factory methods."""

    def test_for_rust_model_basic(self):
        """Test for_rust_model factory method."""
        utility = ActionDependentUtility.for_rust_model(num_states=90)

        assert utility.num_states == 90
        assert utility.num_actions == 2
        assert utility.parameter_names == ["theta_c", "RC"]
        assert utility.state_scale == 90.0

    def test_for_rust_model_custom_scale(self):
        """Test for_rust_model with custom state_scale."""
        utility = ActionDependentUtility.for_rust_model(
            num_states=90,
            state_scale=45.0,
        )

        assert utility.state_scale == 45.0

    def test_from_environment(self, rust_env_small):
        """Test from_environment factory method."""
        utility = ActionDependentUtility.from_environment(rust_env_small)

        assert utility.num_states == rust_env_small.num_states
        assert utility.num_actions == rust_env_small.num_actions
        assert utility.parameter_names == rust_env_small.parameter_names


class TestDeviceMovement:
    """Tests for moving tensors between devices."""

    def test_to_cpu(self):
        """Test moving to CPU."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        utility_cpu = utility.to("cpu")

        params = jnp.array([0.001, 3.0])
        U = utility_cpu.compute(params)

        # JAX arrays are always on the default device
        assert U.shape == (10, 2)


class TestValidation:
    """Tests for parameter validation."""

    def test_validate_correct_shape(self):
        """Test that validation passes for correct shape."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)
        params = jnp.array([0.001, 3.0])

        # Should not raise
        utility.validate_parameters(params)

    def test_validate_wrong_shape(self):
        """Test that validation fails for wrong shape."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)

        # Wrong number of parameters
        params = jnp.array([0.001])

        with pytest.raises(ValueError, match="Expected parameters"):
            utility.validate_parameters(params)

    def test_validate_extra_dimension(self):
        """Test that validation fails for extra dimensions."""
        utility = ActionDependentUtility.for_rust_model(num_states=10)

        # Extra dimension
        params = jnp.array([[0.001, 3.0]])

        with pytest.raises(ValueError, match="Expected parameters"):
            utility.validate_parameters(params)


class TestCustomUtilityFunction:
    """Tests for custom utility and gradient functions."""

    def test_custom_utility_fn(self):
        """Test that custom utility function is used."""
        def custom_utility(params, states):
            # Custom: U = -params[0] * states^2
            U = jnp.zeros((len(states), 2))
            U = U.at[:, 0].set(-params[0] * states ** 2)
            U = U.at[:, 1].set(-params[1])
            return U

        utility = ActionDependentUtility(
            num_states=10,
            num_actions=2,
            parameter_names=["a", "b"],
            utility_fn=custom_utility,
        )

        params = jnp.array([0.1, 2.0])
        U = utility.compute(params)

        # Check custom formula
        expected_keep = -0.1 * jnp.arange(10, dtype=jnp.float32) ** 2
        np.testing.assert_allclose(np.asarray(U[:, 0]), np.asarray(expected_keep))

    def test_custom_gradient_fn(self):
        """Test that custom gradient function is used."""
        def custom_gradient(params, states):
            grad = jnp.zeros((len(states), 2, 2))
            grad = grad.at[:, 0, 0].set(-states ** 2)  # dU/da for keep
            grad = grad.at[:, 1, 1].set(-1.0)  # dU/db for replace
            return grad

        utility = ActionDependentUtility(
            num_states=10,
            num_actions=2,
            parameter_names=["a", "b"],
            gradient_fn=custom_gradient,
        )

        params = jnp.array([0.1, 2.0])
        grad = utility.compute_gradient(params)

        # Check custom gradient
        expected_keep_grad = -jnp.arange(10, dtype=jnp.float32) ** 2
        np.testing.assert_allclose(np.asarray(grad[:, 0, 0]), np.asarray(expected_keep_grad))


class TestIntegrationWithRustEnvironment:
    """Integration tests with RustBusEnvironment."""

    def test_utility_matches_environment_unscaled(self, rust_env_small):
        """Test that utility matches RustBusEnvironment's utility matrix when unscaled.

        RustBusEnvironment uses raw state indices without scaling:
            U(s, keep) = -theta_c * s
            U(s, replace) = -RC

        ActionDependentUtility with state_scale=1.0 should match this.
        """
        # Create with state_scale=1.0 to match environment's unscaled convention
        utility = ActionDependentUtility(
            num_states=rust_env_small.num_states,
            num_actions=rust_env_small.num_actions,
            parameter_names=rust_env_small.parameter_names,
            state_scale=1.0,  # No scaling to match environment
        )
        true_params = rust_env_small.get_true_parameter_vector()

        U_action_utility = utility.compute(true_params)

        # Get utility from environment's feature matrix
        from econirl.preferences.linear import LinearUtility
        linear_utility = LinearUtility.from_environment(rust_env_small)
        U_linear = linear_utility.compute(true_params)

        # Both should give same utility (up to numerical precision)
        np.testing.assert_allclose(np.asarray(U_action_utility), np.asarray(U_linear), atol=1e-5)

    def test_gradient_matches_environment_unscaled(self, rust_env_small):
        """Test that gradient matches LinearUtility gradient when unscaled."""
        # Create with state_scale=1.0 to match environment's unscaled convention
        utility = ActionDependentUtility(
            num_states=rust_env_small.num_states,
            num_actions=rust_env_small.num_actions,
            parameter_names=rust_env_small.parameter_names,
            state_scale=1.0,  # No scaling to match environment
        )
        true_params = rust_env_small.get_true_parameter_vector()

        grad_action = utility.compute_gradient(true_params)

        # Get gradient from LinearUtility
        from econirl.preferences.linear import LinearUtility
        linear_utility = LinearUtility.from_environment(rust_env_small)
        grad_linear = linear_utility.compute_gradient(true_params)

        # Gradients should match
        np.testing.assert_allclose(np.asarray(grad_action), np.asarray(grad_linear), atol=1e-5)

    def test_scaled_utility_is_proportional(self, rust_env_small):
        """Test that scaled utility is proportional to unscaled.

        When state_scale = N, the operating cost parameter is effectively
        scaled by N. This is useful for numerical stability.
        """
        true_params = rust_env_small.get_true_parameter_vector()
        scale = float(rust_env_small.num_states)

        # Unscaled utility
        utility_unscaled = ActionDependentUtility(
            num_states=rust_env_small.num_states,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
            state_scale=1.0,
        )

        # Scaled utility
        utility_scaled = ActionDependentUtility(
            num_states=rust_env_small.num_states,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
            state_scale=scale,
        )

        U_unscaled = utility_unscaled.compute(true_params)
        U_scaled = utility_scaled.compute(true_params)

        # Keep action: U_scaled[:, 0] = U_unscaled[:, 0] / scale
        np.testing.assert_allclose(np.asarray(U_scaled[:, 0]), np.asarray(U_unscaled[:, 0] / scale), atol=1e-5)

        # Replace action: unchanged
        np.testing.assert_allclose(np.asarray(U_scaled[:, 1]), np.asarray(U_unscaled[:, 1]), atol=1e-5)


# Import fixture from conftest
@pytest.fixture
def rust_env_small():
    """Small Rust bus environment for faster tests."""
    from econirl.environments.rust_bus import RustBusEnvironment
    return RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        scale_parameter=1.0,
        seed=42,
    )
