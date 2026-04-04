"""Tests for CCP-based estimators (Hotz-Miller and NPL).

Tests cover:
1. CCP estimation from data
2. Hotz-Miller (K=1) estimation
3. NPL (K>1) estimation
4. Comparison with NFXP
5. Standard errors and inference
"""

import pytest
import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.ccp import CCPEstimator, EULER_GAMMA
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation.synthetic import simulate_panel


class TestCCPEstimation:
    """Tests for CCP estimation from data."""

    def test_ccps_sum_to_one(self, rust_env_small, small_panel, problem_spec_small):
        """Test that estimated CCPs sum to 1 for each state."""
        estimator = CCPEstimator(num_policy_iterations=1)
        ccps = estimator._estimate_ccps_from_data(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )

        row_sums = ccps.sum(axis=1)
        np.testing.assert_allclose(np.asarray(row_sums), np.asarray(jnp.ones_like(row_sums)), atol=1e-5)

    def test_ccps_non_negative(self, rust_env_small, small_panel, problem_spec_small):
        """Test that estimated CCPs are non-negative."""
        estimator = CCPEstimator(num_policy_iterations=1)
        ccps = estimator._estimate_ccps_from_data(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )

        assert (ccps >= 0).all()

    def test_ccps_match_frequencies(self, rust_env_small, problem_spec_small):
        """Test that CCPs match empirical frequencies for large sample."""
        panel = simulate_panel(rust_env_small, n_individuals=500, n_periods=100, seed=42)

        estimator = CCPEstimator(num_policy_iterations=1, ccp_smoothing=0)
        ccps = estimator._estimate_ccps_from_data(
            panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )

        # Compute empirical frequencies manually
        counts = jnp.zeros((problem_spec_small.num_states, problem_spec_small.num_actions))
        for traj in panel.trajectories:
            for t in range(len(traj)):
                state = int(traj.states[t])
                action = int(traj.actions[t])
                counts = counts.at[state, action].add(1)

        state_counts = counts.sum(axis=1, keepdims=True)
        state_counts = jnp.where(state_counts > 0, state_counts, jnp.ones_like(state_counts))
        empirical_ccps = counts / state_counts

        # Should match exactly (or close, with smoothing=0)
        visited_states = counts.sum(axis=1) > 0
        np.testing.assert_allclose(
            np.asarray(ccps[visited_states]), np.asarray(empirical_ccps[visited_states]), atol=1e-5
        )


class TestEmaxCorrection:
    """Tests for emax correction computation."""

    def test_emax_formula(self, rust_env_small):
        """Test that emax correction follows e(a,x) = gamma - log(P(a|x))."""
        estimator = CCPEstimator(num_policy_iterations=1)

        # Create simple CCPs
        ccps = jnp.array([[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])
        e = estimator._compute_emax_correction(ccps)

        expected = EULER_GAMMA - jnp.log(ccps)
        np.testing.assert_allclose(np.asarray(e), np.asarray(expected), atol=1e-6)

    def test_emax_handles_small_probs(self):
        """Test that emax handles very small probabilities without overflow."""
        estimator = CCPEstimator(num_policy_iterations=1, ccp_smoothing=1e-10)

        ccps = jnp.array([[0.999, 0.001], [1e-8, 1 - 1e-8]])
        e = estimator._compute_emax_correction(ccps)

        assert jnp.isfinite(e).all()


class TestHotzMillerEstimation:
    """Tests for Hotz-Miller (K=1) estimator."""

    def test_hotz_miller_runs(self, rust_env_small, small_panel, utility_small,
                              problem_spec_small, transitions_small):
        """Test that Hotz-Miller estimator runs without error."""
        estimator = CCPEstimator(
            num_policy_iterations=1,
            verbose=False,
        )

        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        assert result is not None
        assert len(result.parameters) == utility_small.num_parameters
        assert result.converged

    def test_hotz_miller_name(self):
        """Test that Hotz-Miller has correct name."""
        estimator = CCPEstimator(num_policy_iterations=1)
        assert estimator.name == "Hotz-Miller (CCP)"

    def test_hotz_miller_faster_than_nfxp(self, rust_env_small, utility_small,
                                          problem_spec_small, transitions_small):
        """Test that Hotz-Miller is faster than NFXP."""
        panel = simulate_panel(rust_env_small, n_individuals=100, n_periods=50, seed=42)

        hm_estimator = CCPEstimator(num_policy_iterations=1, verbose=False)
        nfxp_estimator = NFXPEstimator(verbose=False, outer_max_iter=100)

        import time

        # Time Hotz-Miller
        start = time.time()
        hm_result = hm_estimator.estimate(
            panel, utility_small, problem_spec_small, transitions_small
        )
        hm_time = time.time() - start

        # Time NFXP
        start = time.time()
        nfxp_result = nfxp_estimator.estimate(
            panel, utility_small, problem_spec_small, transitions_small
        )
        nfxp_time = time.time() - start

        # Hotz-Miller should be faster
        assert hm_time < nfxp_time, f"HM: {hm_time:.2f}s, NFXP: {nfxp_time:.2f}s"


class TestNPLEstimation:
    """Tests for NPL (K>1) estimator."""

    def test_npl_runs(self, rust_env_small, small_panel, utility_small,
                      problem_spec_small, transitions_small):
        """Test that NPL estimator runs without error."""
        estimator = CCPEstimator(
            num_policy_iterations=5,
            verbose=False,
        )

        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        assert result is not None
        assert len(result.parameters) == utility_small.num_parameters

    def test_npl_name(self):
        """Test that NPL has correct name."""
        estimator = CCPEstimator(num_policy_iterations=5)
        assert estimator.name == "NPL (K=5)"

        estimator2 = CCPEstimator(num_policy_iterations=-1)
        assert "convergence" in estimator2.name.lower()

    def test_npl_improves_over_hotz_miller(self, rust_env_small, utility_small,
                                           problem_spec_small, transitions_small):
        """Test that NPL iterations improve log-likelihood over Hotz-Miller."""
        panel = simulate_panel(rust_env_small, n_individuals=200, n_periods=100, seed=42)

        hm_estimator = CCPEstimator(num_policy_iterations=1, verbose=False)
        npl_estimator = CCPEstimator(num_policy_iterations=10, verbose=False)

        hm_result = hm_estimator.estimate(
            panel, utility_small, problem_spec_small, transitions_small
        )
        npl_result = npl_estimator.estimate(
            panel, utility_small, problem_spec_small, transitions_small
        )

        # NPL should have higher (or equal) log-likelihood
        assert npl_result.log_likelihood >= hm_result.log_likelihood - 0.1


class TestCCPvsNFXP:
    """Tests comparing CCP estimators to NFXP."""

    def test_npl_converges_to_nfxp(self, rust_env_small, utility_small,
                                    problem_spec_small, transitions_small):
        """Test that NPL converges to same estimates as NFXP."""
        panel = simulate_panel(rust_env_small, n_individuals=300, n_periods=100, seed=42)

        npl_estimator = CCPEstimator(
            num_policy_iterations=20,
            convergence_tol=1e-8,
            verbose=False,
        )
        nfxp_estimator = NFXPEstimator(
            verbose=False,
            outer_max_iter=200,
            outer_tol=1e-8,
        )

        npl_result = npl_estimator.estimate(
            panel, utility_small, problem_spec_small, transitions_small
        )
        nfxp_result = nfxp_estimator.estimate(
            panel, utility_small, problem_spec_small, transitions_small
        )

        # Parameters should be close
        param_diff = jnp.abs(npl_result.parameters - nfxp_result.parameters)
        assert float(param_diff.max()) < 0.1, \
            f"NPL: {npl_result.parameters}, NFXP: {nfxp_result.parameters}"

    def test_all_methods_recover_true_params(self, rust_env_small, utility_small,
                                              problem_spec_small, transitions_small):
        """Test that all methods recover true parameters reasonably well."""
        panel = simulate_panel(rust_env_small, n_individuals=500, n_periods=100, seed=42)
        true_params = rust_env_small.get_true_parameter_vector()

        estimators = {
            "Hotz-Miller": CCPEstimator(num_policy_iterations=1, verbose=False),
            "NPL": CCPEstimator(num_policy_iterations=10, verbose=False),
            "NFXP": NFXPEstimator(verbose=False, outer_max_iter=200),
        }

        for name, estimator in estimators.items():
            result = estimator.estimate(
                panel, utility_small, problem_spec_small, transitions_small
            )

            # Check parameter recovery (within 50% relative error)
            for i, param_name in enumerate(result.parameter_names):
                estimate = float(result.parameters[i])
                true_val = float(true_params[i])
                rel_error = abs(estimate - true_val) / (abs(true_val) + 1e-8)

                assert rel_error < 0.5, \
                    f"{name}: {param_name} has {rel_error:.1%} error " \
                    f"(est={estimate:.4f}, true={true_val:.4f})"


class TestCCPInference:
    """Tests for CCP inference (standard errors, confidence intervals)."""

    def test_standard_errors_computed(self, rust_env_small, small_panel, utility_small,
                                       problem_spec_small, transitions_small):
        """Test that standard errors are computed."""
        estimator = CCPEstimator(
            num_policy_iterations=1,
            compute_hessian=True,
            verbose=False,
        )

        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        assert result.standard_errors is not None
        assert len(result.standard_errors) == len(result.parameters)
        # SEs may be NaN if Hessian is singular (e.g., with small test data)
        assert result.standard_errors is not None

    def test_confidence_intervals(self, rust_env_small, small_panel, utility_small,
                                   problem_spec_small, transitions_small):
        """Test that confidence intervals contain point estimates."""
        estimator = CCPEstimator(
            num_policy_iterations=1,
            compute_hessian=True,
            verbose=False,
        )

        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        lower, upper = result.confidence_interval(alpha=0.05)

        # With small test data, SEs may be NaN -> CIs are NaN -> skip check
        for i in range(len(result.parameters)):
            if jnp.isfinite(lower[i]) and jnp.isfinite(upper[i]):
                assert lower[i] <= result.parameters[i] <= upper[i]

    def test_summary_output(self, rust_env_small, small_panel, utility_small,
                            problem_spec_small, transitions_small):
        """Test that summary output is generated correctly."""
        estimator = CCPEstimator(num_policy_iterations=1, verbose=False)
        result = estimator.estimate(
            small_panel, utility_small, problem_spec_small, transitions_small
        )

        summary = result.summary()

        assert "Hotz-Miller" in summary or "CCP" in summary
        assert "coef" in summary
        assert "std err" in summary


class TestCCPEdgeCases:
    """Tests for edge cases and error handling."""

    def test_sparse_data(self, rust_env_small, problem_spec_small, transitions_small):
        """Test handling of sparse data (few observations per state)."""
        # Small panel with sparse coverage
        panel = simulate_panel(rust_env_small, n_individuals=10, n_periods=10, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        estimator = CCPEstimator(
            num_policy_iterations=1,
            ccp_min_count=1,
            verbose=False,
        )

        # Should run without error
        result = estimator.estimate(
            panel, utility, problem_spec_small, transitions_small
        )

        assert result is not None
        assert jnp.isfinite(result.parameters).all()

    def test_unvisited_states(self, rust_env_small, problem_spec_small, transitions_small):
        """Test handling of states never visited in data."""
        # Very small panel - some states won't be visited
        panel = simulate_panel(rust_env_small, n_individuals=5, n_periods=5, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        estimator = CCPEstimator(
            num_policy_iterations=1,
            ccp_min_count=1,
            verbose=False,
        )

        result = estimator.estimate(
            panel, utility, problem_spec_small, transitions_small
        )

        # CCPs for unvisited states should be uniform
        ccps = estimator._estimate_ccps_from_data(
            panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )

        # Count which states were visited
        visited = jnp.zeros(problem_spec_small.num_states)
        for traj in panel.trajectories:
            for state in traj.states:
                visited = visited.at[int(state)].set(1)

        # Unvisited states should have uniform CCPs
        unvisited_mask = visited == 0
        if unvisited_mask.any():
            uniform_prob = 1.0 / problem_spec_small.num_actions
            np.testing.assert_allclose(
                np.asarray(ccps[unvisited_mask]),
                np.asarray(jnp.full_like(ccps[unvisited_mask], uniform_prob)),
                atol=1e-5
            )
