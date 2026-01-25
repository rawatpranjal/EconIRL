"""Test NFXP inner loop convergence with high discount factors.

This module tests that the NFXP estimator converges properly with high
discount factors (e.g., beta=0.9999). The key insight is that value
iteration convergence is O(1/(1-beta)), so beta=0.9999 requires ~10000x
more iterations than beta=0.9.

Without sufficient inner loop iterations, the value function doesn't
converge, leading to incorrect choice probabilities and completely
wrong parameter estimates.

Reference:
    The convergence rate for value iteration is geometric with rate beta.
    For beta=0.9999, reaching tolerance 1e-10 requires:
        log(1e-10) / log(0.9999) ~ 230,000 iterations
"""
import pytest
import torch
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation import simulate_panel


class TestNFXPConvergence:
    """Tests for NFXP convergence with Rust bus data."""

    @pytest.fixture
    def rust_env(self):
        """Standard Rust bus environment."""
        return RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.9999,
        )

    @pytest.mark.slow
    def test_nfxp_recovers_true_parameters(self, rust_env):
        """NFXP should recover ground truth parameters within 3 SEs."""
        panel = simulate_panel(rust_env, n_individuals=300, n_periods=100, seed=42)
        utility = LinearUtility.from_environment(rust_env)

        # Use sufficient inner iterations for high discount
        estimator = NFXPEstimator(
            se_method="asymptotic",
            inner_max_iter=300000,  # Critical for beta=0.9999
            inner_tol=1e-10,
            verbose=False,
        )

        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        true_params = rust_env.get_true_parameter_vector()

        # Check recovery within 3 standard errors
        for i, name in enumerate(result.parameter_names):
            error = abs(result.parameters[i].item() - true_params[i].item())
            se = result.standard_errors[i].item()
            assert error < 3 * se, (
                f"Parameter {name}: estimate={result.parameters[i].item():.6f}, "
                f"true={true_params[i].item():.6f}, error={error:.6f}, 3*SE={3*se:.6f}"
            )

    @pytest.mark.slow
    def test_inner_loop_converges_with_high_discount(self, rust_env):
        """Inner loop should converge without warnings when given enough iterations."""
        panel = simulate_panel(rust_env, n_individuals=50, n_periods=20, seed=42)
        utility = LinearUtility.from_environment(rust_env)

        # Default inner_max_iter is too low for beta=0.9999
        estimator = NFXPEstimator(
            se_method="asymptotic",
            inner_max_iter=300000,
            inner_tol=1e-10,
            verbose=False,
        )

        # This should not print "Inner loop did not converge" warnings
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        assert result.converged

    def test_warning_for_high_discount_low_iter(self, rust_env):
        """Should warn when discount is high but inner_max_iter is low."""
        panel = simulate_panel(rust_env, n_individuals=10, n_periods=10, seed=42)
        utility = LinearUtility.from_environment(rust_env)

        # Low inner_max_iter with high beta should trigger warning
        estimator = NFXPEstimator(
            se_method="asymptotic",
            inner_max_iter=1000,  # Too low for beta=0.9999
            inner_tol=1e-10,
            verbose=False,
        )

        with pytest.warns(
            UserWarning,
            match=r"High discount factor.*may require inner_max_iter",
        ):
            estimator.estimate(
                panel=panel,
                utility=utility,
                problem=rust_env.problem_spec,
                transitions=rust_env.transition_matrices,
            )

    def test_no_warning_for_moderate_discount(self):
        """Should not warn when discount factor is moderate."""
        env = RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.95,  # Moderate discount
        )
        panel = simulate_panel(env, n_individuals=10, n_periods=10, seed=42)
        utility = LinearUtility.from_environment(env)

        estimator = NFXPEstimator(
            se_method="asymptotic",
            inner_max_iter=1000,  # Should be enough for beta=0.95
            inner_tol=1e-10,
            verbose=False,
        )

        # Should not raise UserWarning about high discount
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estimator.estimate(
                panel=panel,
                utility=utility,
                problem=env.problem_spec,
                transitions=env.transition_matrices,
            )

            # Filter for our specific warning (ignore scipy deprecation warnings)
            high_discount_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "High discount factor" in str(warning.message)
            ]
            assert len(high_discount_warnings) == 0, (
                "Should not warn about high discount factor for moderate beta=0.95"
            )
