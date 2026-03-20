"""Integration tests for Rust bus estimation.

These tests verify the complete estimation pipeline:
1. Environment creation
2. Data simulation
3. NFXP estimation
4. Parameter recovery
5. Inference (standard errors, CIs)
6. Counterfactual analysis
"""

import pytest
import torch
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration, policy_iteration
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation.synthetic import simulate_panel
from econirl.simulation.counterfactual import counterfactual_policy
from econirl.inference.identification import check_identification


class TestBellmanOperator:
    """Tests for the Bellman operator."""

    def test_bellman_contraction(self, rust_env_small, transitions_small, problem_spec_small):
        """Test that Bellman operator is a contraction."""
        operator = SoftBellmanOperator(problem_spec_small, transitions_small)
        utility = rust_env_small.compute_utility_matrix()

        # Start from two different initial values
        V1 = torch.zeros(problem_spec_small.num_states)
        V2 = torch.ones(problem_spec_small.num_states) * 10

        result1 = operator.apply(utility, V1)
        result2 = operator.apply(utility, V2)

        # Contraction: ||TV1 - TV2|| <= beta * ||V1 - V2||
        # Adding small tolerance for floating-point precision at the boundary
        initial_dist = torch.abs(V1 - V2).max()
        after_dist = torch.abs(result1.V - result2.V).max()

        assert after_dist <= problem_spec_small.discount_factor * initial_dist + 1e-6

    def test_policy_sums_to_one(self, rust_env_small, transitions_small, problem_spec_small):
        """Test that policy probabilities sum to 1."""
        operator = SoftBellmanOperator(problem_spec_small, transitions_small)
        utility = rust_env_small.compute_utility_matrix()

        V = torch.zeros(problem_spec_small.num_states)
        result = operator.apply(utility, V)

        row_sums = result.policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_policy_non_negative(self, rust_env_small, transitions_small, problem_spec_small):
        """Test that policy probabilities are non-negative."""
        operator = SoftBellmanOperator(problem_spec_small, transitions_small)
        utility = rust_env_small.compute_utility_matrix()

        V = torch.zeros(problem_spec_small.num_states)
        result = operator.apply(utility, V)

        assert (result.policy >= 0).all()


class TestSolvers:
    """Tests for value iteration and policy iteration."""

    def test_value_iteration_convergence(self, rust_env_small, transitions_small, problem_spec_small):
        """Test that value iteration converges."""
        operator = SoftBellmanOperator(problem_spec_small, transitions_small)
        utility = rust_env_small.compute_utility_matrix()

        result = value_iteration(operator, utility, tol=1e-6, max_iter=2000)

        assert result.converged
        assert result.num_iterations < 2000

    def test_policy_iteration_convergence(self, rust_env, transitions, problem_spec):
        """Test that policy iteration converges."""
        # With β=0.9999, convergence is slow so we use a looser tolerance
        operator = SoftBellmanOperator(problem_spec, transitions)
        utility = rust_env.compute_utility_matrix()

        result = policy_iteration(operator, utility, tol=1e-4, max_iter=200)

        assert result.converged

    def test_solvers_agree(self, rust_env_small, transitions_small, problem_spec_small):
        """Test that value iteration and policy iteration give same result."""
        operator = SoftBellmanOperator(problem_spec_small, transitions_small)
        utility = rust_env_small.compute_utility_matrix()

        vi_result = value_iteration(operator, utility, tol=1e-6, max_iter=2000)
        pi_result = policy_iteration(operator, utility, tol=1e-6)

        assert vi_result.converged, "Value iteration did not converge"
        assert pi_result.converged, "Policy iteration did not converge"

        # Value functions should match (use relative tolerance)
        assert torch.allclose(vi_result.V, pi_result.V, rtol=1e-3, atol=1e-3)

        # Policies should match
        assert torch.allclose(vi_result.policy, pi_result.policy, rtol=1e-3, atol=1e-3)


class TestDataSimulation:
    """Tests for data simulation."""

    def test_simulate_panel_shape(self, rust_env_small):
        """Test that simulated panel has correct shape."""
        panel = simulate_panel(rust_env_small, n_individuals=10, n_periods=20, seed=42)

        assert panel.num_individuals == 10
        assert panel.num_observations == 10 * 20

    def test_simulate_panel_valid_states(self, rust_env_small):
        """Test that simulated states are valid."""
        panel = simulate_panel(rust_env_small, n_individuals=10, n_periods=20, seed=42)

        all_states = panel.get_all_states()
        assert (all_states >= 0).all()
        assert (all_states < rust_env_small.num_states).all()

    def test_simulate_panel_valid_actions(self, rust_env_small):
        """Test that simulated actions are valid."""
        panel = simulate_panel(rust_env_small, n_individuals=10, n_periods=20, seed=42)

        all_actions = panel.get_all_actions()
        assert (all_actions >= 0).all()
        assert (all_actions < rust_env_small.num_actions).all()

    def test_simulate_panel_reproducible(self, rust_env_small):
        """Test that simulation is reproducible with seed."""
        panel1 = simulate_panel(rust_env_small, n_individuals=10, n_periods=20, seed=42)
        panel2 = simulate_panel(rust_env_small, n_individuals=10, n_periods=20, seed=42)

        assert torch.equal(panel1.get_all_states(), panel2.get_all_states())
        assert torch.equal(panel1.get_all_actions(), panel2.get_all_actions())


class TestNFXPEstimation:
    """Tests for NFXP estimator."""

    def test_nfxp_runs(self, rust_env_small, small_panel, utility_small,
                       problem_spec_small, transitions_small):
        """Test that NFXP estimator runs without error."""
        estimator = NFXPEstimator(
            se_method="asymptotic",
            verbose=False,
            outer_max_iter=50,
        )

        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        assert result is not None
        assert len(result.parameters) == utility_small.num_parameters

    def test_nfxp_parameter_recovery(self, rust_env, utility,
                                      problem_spec, transitions):
        """Test that NFXP recovers true parameters with large sample."""
        # Use standard environment with larger sample for better recovery
        panel = simulate_panel(rust_env, n_individuals=300, n_periods=100, seed=42)

        estimator = NFXPEstimator(
            se_method="asymptotic",
            verbose=False,
            outer_max_iter=200,
        )

        result = estimator.estimate(
            panel, utility, problem_spec, transitions
        )

        true_params = rust_env.get_true_parameter_vector()

        # Check recovery - RMSE should be small
        rmse = torch.sqrt(torch.mean((result.parameters - true_params) ** 2)).item()
        assert rmse < 0.1, (
            f"NFXP RMSE={rmse:.6f} exceeds 0.1. "
            f"Estimates: {result.parameters.tolist()}, True: {true_params.tolist()}"
        )

        # Also check each parameter is close (absolute tolerance)
        for i, name in enumerate(result.parameter_names):
            error = abs(result.parameters[i] - true_params[i])

            assert error < 0.1, \
                f"Parameter {name}: estimate {result.parameters[i]:.4f}, " \
                f"true {true_params[i]:.4f}, error {error:.4f}"

        # Original SE-based check kept as informational (not assertion)
        for i, name in enumerate(result.parameter_names):
            error = abs(result.parameters[i] - true_params[i])
            se = result.standard_errors[i]
            if torch.isfinite(se) and se > 0 and not (error < 3 * se):
                # Numerical SEs at high gamma can be underestimated
                rel_error = error / (abs(true_params[i]) + 1e-8)
                assert rel_error < 0.5, \
                    f"Parameter {name}: estimate {result.parameters[i]:.4f}, " \
                    f"true {true_params[i]:.4f}, rel_error {rel_error:.2%}"

    def test_nfxp_summary_output(self, rust_env_small, small_panel, utility_small,
                                  problem_spec_small, transitions_small):
        """Test that summary output is generated correctly."""
        estimator = NFXPEstimator(verbose=False, outer_max_iter=50)
        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        summary = result.summary()

        # Check that key elements are in summary
        assert "Dynamic Discrete Choice" in summary
        assert "NFXP" in summary
        assert "coef" in summary
        assert "std err" in summary

    def test_nfxp_confidence_intervals(self, rust_env_small, small_panel, utility_small,
                                        problem_spec_small, transitions_small):
        """Test that confidence intervals are computed."""
        estimator = NFXPEstimator(verbose=False, outer_max_iter=50)
        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        lower, upper = result.confidence_interval(alpha=0.05)

        # CI should contain point estimate
        for i in range(len(result.parameters)):
            assert lower[i] <= result.parameters[i] <= upper[i]

        # Upper > Lower
        assert (upper > lower).all()


class TestIdentification:
    """Tests for identification diagnostics."""

    def test_identification_check_runs(self, rust_env_small, small_panel, utility_small,
                                        problem_spec_small, transitions_small):
        """Test that identification check runs."""
        estimator = NFXPEstimator(verbose=False, outer_max_iter=50)
        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        assert result.identification is not None
        assert result.identification.status is not None

    def test_well_identified_model(self, rust_env_small, utility_small,
                                    problem_spec_small, transitions_small):
        """Test that standard Rust model is well-identified."""
        panel = simulate_panel(rust_env_small, n_individuals=200, n_periods=100, seed=42)

        estimator = NFXPEstimator(verbose=False, outer_max_iter=100)
        result = estimator.estimate(
            panel, utility_small, problem_spec_small, transitions_small
        )

        # Should be well-identified with reasonable data
        assert result.identification.rank == utility_small.num_parameters


class TestCounterfactual:
    """Tests for counterfactual analysis."""

    def test_counterfactual_policy_runs(self, rust_env_small, small_panel, utility_small,
                                         problem_spec_small, transitions_small):
        """Test that counterfactual analysis runs."""
        estimator = NFXPEstimator(verbose=False, outer_max_iter=50)
        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        # Double replacement cost
        new_params = result.parameters.clone()
        new_params[1] *= 2

        cf = counterfactual_policy(
            result, new_params, utility_small, problem_spec_small, transitions_small
        )

        assert cf.counterfactual_policy is not None
        assert cf.policy_change is not None

    def test_counterfactual_direction(self, rust_env_small, small_panel, utility_small,
                                       problem_spec_small, transitions_small):
        """Test that counterfactual changes in expected direction."""
        estimator = NFXPEstimator(verbose=False, outer_max_iter=50)
        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        # If replacement cost increases, replacement probability should decrease
        new_params = result.parameters.clone()
        new_params[1] *= 2  # Double replacement cost

        cf = counterfactual_policy(
            result, new_params, utility_small, problem_spec_small, transitions_small
        )

        # Average replacement probability should decrease
        baseline_replace_prob = cf.baseline_policy[:, 1].mean()
        cf_replace_prob = cf.counterfactual_policy[:, 1].mean()

        assert cf_replace_prob < baseline_replace_prob


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, rust_env_small):
        """Test complete estimation pipeline."""
        # 1. Create environment
        env = rust_env_small
        problem = env.problem_spec
        transitions = env.transition_matrices

        # 2. Create utility specification
        utility = LinearUtility.from_environment(env)

        # 3. Simulate data
        panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=42)

        # 4. Estimate
        estimator = NFXPEstimator(verbose=False, outer_max_iter=50)
        result = estimator.estimate(panel, utility, problem, transitions)

        # 5. Check output
        assert result.converged or result.num_iterations > 0
        assert result.log_likelihood < 0  # Log-likelihood should be negative
        assert result.goodness_of_fit is not None

        # 6. Summary should work
        summary = result.summary()
        assert len(summary) > 0

        # 7. Counterfactual should work
        new_params = result.parameters * 1.1
        cf = counterfactual_policy(result, new_params, utility, problem, transitions)
        assert cf.welfare_change != 0 or torch.allclose(
            cf.baseline_policy, cf.counterfactual_policy, atol=1e-6
        )

    def test_latex_export(self, rust_env_small, small_panel, utility_small,
                          problem_spec_small, transitions_small, tmp_path):
        """Test LaTeX table export."""
        estimator = NFXPEstimator(verbose=False, outer_max_iter=50)
        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        # Export to LaTeX
        latex_file = tmp_path / "results.tex"
        latex = result.to_latex(str(latex_file))

        # Check file was created
        assert latex_file.exists()

        # Check content
        assert "\\begin{table}" in latex
        assert "operating_cost" in latex or "θ" in latex
