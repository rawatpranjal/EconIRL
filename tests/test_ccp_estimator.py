"""Tests for CCP-based estimators (Hotz-Miller and NPL).

Tests cover:
1. CCP estimation from data
2. Hotz-Miller (K=1) estimation
3. NPL (K>1) estimation
4. Comparison with NFXP
5. Standard errors and inference
"""

import warnings
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.optimizer import OptimizeResult
from econirl.estimation.ccp import EULER_GAMMA, CCPEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.linear import LinearUtility
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
        np.testing.assert_allclose(
            np.asarray(row_sums),
            np.asarray(jnp.ones_like(row_sums)),
            atol=1e-5,
        )

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

    def test_ccp_estimation_avoids_jax_scatter_dtype_warning(
        self,
        small_panel,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """Test that CCP fitting is dtype-clean under JAX scatter rules."""
        estimator = CCPEstimator(
            num_policy_iterations=1,
            compute_hessian=False,
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="scatter inputs have incompatible types.*",
                category=FutureWarning,
            )
            result = estimator.estimate(
                small_panel,
                utility_small,
                problem_spec_small,
                transitions_small,
            )

        assert result.converged


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

    def test_choice_probabilities_are_scale_equivariant(
        self,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """Scaling rewards and logit shocks together leaves choices unchanged."""
        estimator = CCPEstimator(num_policy_iterations=1)
        ccps = jnp.full(
            (problem_spec_small.num_states, problem_spec_small.num_actions),
            1.0 / problem_spec_small.num_actions,
        )
        parameters = jnp.array([0.25, 1.5])
        scaled_problem = replace(problem_spec_small, scale_parameter=2.0)

        values = estimator._compute_choice_specific_values(
            ccps,
            transitions_small,
            utility_small,
            parameters,
            problem_spec_small,
        )
        scaled_values = estimator._compute_choice_specific_values(
            ccps,
            transitions_small,
            utility_small,
            2.0 * parameters,
            scaled_problem,
        )

        np.testing.assert_allclose(
            np.asarray(scaled_values),
            2.0 * np.asarray(values),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(jax.nn.softmax(scaled_values / 2.0, axis=1)),
            np.asarray(jax.nn.softmax(values, axis=1)),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_emax_handles_small_probs(self):
        """Test that emax handles very small probabilities without overflow."""
        estimator = CCPEstimator(num_policy_iterations=1, ccp_smoothing=1e-10)

        ccps = jnp.array([[0.999, 0.001], [1e-8, 1 - 1e-8]])
        e = estimator._compute_emax_correction(ccps)

        assert jnp.isfinite(e).all()

    def test_euler_normalization_preserves_choice_probabilities(
        self,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """Removing Euler's constant should only fix the value-function level."""
        estimator = CCPEstimator(num_policy_iterations=1)
        ccps = jnp.full(
            (problem_spec_small.num_states, problem_spec_small.num_actions),
            1.0 / problem_spec_small.num_actions,
        )
        parameters = jnp.array([0.25, 1.5])
        normalized_values = estimator._compute_choice_specific_values(
            ccps,
            transitions_small,
            utility_small,
            parameters,
            problem_spec_small,
        )
        legacy_shift = (
            problem_spec_small.discount_factor
            * problem_spec_small.scale_parameter
            * EULER_GAMMA
            / (1.0 - problem_spec_small.discount_factor)
        )
        legacy_values = normalized_values + legacy_shift

        np.testing.assert_allclose(
            np.asarray(
                jax.nn.softmax(
                    normalized_values / problem_spec_small.scale_parameter,
                    axis=1,
                )
            ),
            np.asarray(
                jax.nn.softmax(
                    legacy_values / problem_spec_small.scale_parameter,
                    axis=1,
                )
            ),
            rtol=1e-6,
            atol=1e-6,
        )


class TestHotzMillerEstimation:
    """Tests for Hotz-Miller (K=1) estimator."""

    def test_hotz_miller_runs(
        self, rust_env_small, small_panel, utility_small, problem_spec_small, transitions_small
    ):
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
        assert result.metadata["mode"] == "one_step"
        assert result.metadata["final_ccps"].shape == (
            problem_spec_small.num_states,
            problem_spec_small.num_actions,
        )

    def test_hotz_miller_name(self):
        """Test that Hotz-Miller has correct name."""
        estimator = CCPEstimator(num_policy_iterations=1)
        assert estimator.name == "Hotz-Miller (CCP)"

    # A CCP-faster-than-NFXP wall-clock assertion was removed here. CCP's speed
    # advantage over NFXP is asymptotic in the state count; on a 90-state toy it
    # does not hold (CCP carries fixed JAX overhead, NFXP converges in a few BHHH
    # steps), so the ordering is not a meaningful correctness gate. The real
    # runtime comparison belongs in the cross-estimator runtime-vs-scale study.


class TestNPLEstimation:
    """Tests for NPL (K>1) estimator."""

    def test_npl_runs(
        self, rust_env_small, small_panel, utility_small, problem_spec_small, transitions_small
    ):
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
        assert result.converged
        assert result.metadata["termination_reason"] in {
            "fixed_k_complete",
            "fixed_point_converged",
        }
        assert result.metadata["inner_optimizer_succeeded"]

    def test_inner_optimizer_failure_is_visible(
        self,
        monkeypatch,
        small_panel,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """A failed pseudo-likelihood solve must not look like completion."""

        def fail_optimizer(_fun, x0, **_kwargs):
            return OptimizeResult(
                x=x0,
                fun=1.0,
                success=False,
                nit=1,
                nfev=1,
                message="Maximum iterations reached",
                grad_norm=1.0,
                projected_grad_norm=1.0,
                convergence_reason="max_iterations",
            )

        monkeypatch.setattr("econirl.estimation.ccp.minimize_lbfgsb", fail_optimizer)
        result = CCPEstimator(
            num_policy_iterations=3,
            compute_hessian=False,
        ).estimate(
            small_panel,
            utility_small,
            problem_spec_small,
            transitions_small,
        )

        assert not result.converged
        assert result.metadata["termination_reason"] == "inner_optimizer_failed"
        assert not result.metadata["inner_optimizer_succeeded"]
        assert result.metadata["inner_optimizer_history"][0]["accepted"] is False

    def test_parameter_stability_alone_does_not_claim_fixed_point(
        self,
        monkeypatch,
        small_panel,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """A stable parameter vector is insufficient when the policy still moves."""

        def unchanged_optimizer(_fun, x0, **_kwargs):
            return OptimizeResult(
                x=x0,
                fun=1.0,
                success=True,
                nit=1,
                nfev=1,
                message="Converged",
                grad_norm=0.0,
                projected_grad_norm=0.0,
                convergence_reason="projected_gradient",
            )

        monkeypatch.setattr(
            "econirl.estimation.ccp.minimize_lbfgsb",
            unchanged_optimizer,
        )
        estimator = CCPEstimator(
            num_policy_iterations=3,
            compute_hessian=False,
        )
        policy_updates = 0

        def moving_policy(_values, _scale):
            nonlocal policy_updates
            policy_updates += 1
            first_action_probability = 0.8 if policy_updates % 2 else 0.2
            return jnp.tile(
                jnp.array(
                    [first_action_probability, 1.0 - first_action_probability],
                ),
                (problem_spec_small.num_states, 1),
            )

        monkeypatch.setattr(estimator, "_update_ccps_from_values", moving_policy)
        result = estimator.estimate(
            small_panel,
            utility_small,
            problem_spec_small,
            transitions_small,
        )

        assert result.converged
        assert not result.metadata["npl_converged"]
        assert result.metadata["termination_reason"] == "fixed_k_complete"
        assert result.num_iterations == 3
        assert result.metadata["npl_parameter_residual"] == pytest.approx(0.0)
        assert result.metadata["npl_policy_residual"] > 0.0

    def test_fixed_k_early_joint_convergence_is_successful(
        self,
        monkeypatch,
        small_panel,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """A fixed-K run stops early only when both NPL residuals pass."""

        def unchanged_optimizer(_fun, x0, **_kwargs):
            return OptimizeResult(
                x=x0,
                fun=1.0,
                success=True,
                nit=1,
                nfev=1,
                message="Converged",
                grad_norm=0.0,
                projected_grad_norm=0.0,
                convergence_reason="projected_gradient",
            )

        uniform_policy = jnp.full(
            (problem_spec_small.num_states, problem_spec_small.num_actions),
            1.0 / problem_spec_small.num_actions,
        )
        monkeypatch.setattr(
            "econirl.estimation.ccp.minimize_lbfgsb",
            unchanged_optimizer,
        )
        estimator = CCPEstimator(
            num_policy_iterations=3,
            compute_hessian=False,
        )
        monkeypatch.setattr(
            estimator,
            "_estimate_ccps_from_data",
            lambda *_args: uniform_policy,
        )
        monkeypatch.setattr(
            estimator,
            "_update_ccps_from_values",
            lambda _values, _scale: uniform_policy,
        )

        result = estimator.estimate(
            small_panel,
            utility_small,
            problem_spec_small,
            transitions_small,
        )

        assert result.converged
        assert result.metadata["npl_converged"]
        assert result.metadata["termination_reason"] == "fixed_point_converged"
        assert result.num_iterations == 1
        assert result.metadata["npl_parameter_residual"] == pytest.approx(0.0)
        assert result.metadata["npl_policy_residual"] == pytest.approx(0.0)
        assert result.metadata["npl_residual_history"] == [
            {
                "policy_iteration": 1,
                "parameter_residual": 0.0,
                "policy_residual": 0.0,
            }
        ]

    def test_npl_name(self):
        """Test that NPL has correct name."""
        estimator = CCPEstimator(num_policy_iterations=5)
        assert estimator.name == "NPL (K=5)"

        estimator2 = CCPEstimator(num_policy_iterations=-1)
        assert "convergence" in estimator2.name.lower()

    def test_mode_aliases(self):
        """Test that named modes map to the documented estimators."""
        one_step = CCPEstimator(mode="one_step")
        npl = CCPEstimator(mode="npl")

        assert one_step.name == "Hotz-Miller (CCP)"
        assert "convergence" in npl.name.lower()

    def test_npl_improves_over_hotz_miller(
        self, rust_env_small, utility_small, problem_spec_small, transitions_small
    ):
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

    def test_npl_converges_to_nfxp(
        self, rust_env_small, utility_small, problem_spec_small, transitions_small
    ):
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
        assert float(param_diff.max()) < 0.1, (
            f"NPL: {npl_result.parameters}, NFXP: {nfxp_result.parameters}"
        )

    def test_all_methods_recover_true_params(
        self, rust_env_small, utility_small, problem_spec_small, transitions_small
    ):
        """Test that all methods recover true parameters reasonably well."""
        panel = simulate_panel(rust_env_small, n_individuals=500, n_periods=100, seed=42)
        true_params = rust_env_small.get_true_parameter_vector()

        estimators = {
            "Hotz-Miller": CCPEstimator(num_policy_iterations=1, verbose=False),
            "NPL": CCPEstimator(num_policy_iterations=10, verbose=False),
            "NFXP": NFXPEstimator(verbose=False, outer_max_iter=200),
        }

        for name, estimator in estimators.items():
            result = estimator.estimate(panel, utility_small, problem_spec_small, transitions_small)

            # Check parameter recovery (within 50% relative error)
            for i, param_name in enumerate(result.parameter_names):
                estimate = float(result.parameters[i])
                true_val = float(true_params[i])
                rel_error = abs(estimate - true_val) / (abs(true_val) + 1e-8)

                assert rel_error < 0.5, (
                    f"{name}: {param_name} has {rel_error:.1%} error "
                    f"(est={estimate:.4f}, true={true_val:.4f})"
                )


class TestCCPInference:
    """Tests for CCP inference (standard errors, confidence intervals)."""

    def test_linear_inference_matches_fixed_ccp_pseudo_likelihood(
        self,
        rust_env_small,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """Linear CCP inference uses one internally consistent criterion."""
        panel = simulate_panel(
            rust_env_small,
            n_individuals=300,
            n_periods=50,
            seed=2718,
        )
        estimator = CCPEstimator(
            num_policy_iterations=1,
            compute_hessian=True,
            verbose=False,
        )

        result = estimator.estimate(
            panel,
            utility_small,
            problem_spec_small,
            transitions_small,
        )

        assert result.metadata["se_method_detail"] == "fixed_ccp_pseudo_likelihood"
        assert result.hessian is not None
        eigenvalues = np.linalg.eigvalsh(np.asarray(result.hessian))
        assert np.all(eigenvalues < 0)
        assert np.isfinite(np.asarray(result.standard_errors)).all()
        assert (np.asarray(result.standard_errors) > 0).all()

    def test_standard_errors_computed(
        self, rust_env_small, small_panel, utility_small, problem_spec_small, transitions_small
    ):
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

    def test_confidence_intervals(
        self, rust_env_small, small_panel, utility_small, problem_spec_small, transitions_small
    ):
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

    def test_summary_output(
        self, rust_env_small, small_panel, utility_small, problem_spec_small, transitions_small
    ):
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

    def test_invalid_mode_raises(self):
        """Test that invalid mode names fail at construction."""
        with pytest.raises(ValueError, match="mode"):
            CCPEstimator(mode="bad")

    def test_invalid_policy_iteration_count_raises(self):
        """Test that invalid NPL iteration counts fail at construction."""
        with pytest.raises(ValueError, match="num_policy_iterations"):
            CCPEstimator(num_policy_iterations=0)

    def test_invalid_smoothing_raises(self):
        """Test that negative CCP smoothing fails at construction."""
        with pytest.raises(ValueError, match="ccp_smoothing"):
            CCPEstimator(ccp_smoothing=-1e-6)

    @pytest.mark.parametrize("num_policy_iterations", [1, 2])
    def test_full_likelihood_rejects_non_fixed_point_profiles(
        self,
        num_policy_iterations,
        small_panel,
        utility_small,
        problem_spec_small,
        transitions_small,
    ):
        """Joint inference is available only after unconstrained NPL convergence."""
        estimator = CCPEstimator(
            num_policy_iterations=num_policy_iterations,
            se_method="full_likelihood_bhhh",
            compute_hessian=True,
            verbose=False,
        )

        with pytest.raises(ValueError, match="num_policy_iterations=-1"):
            estimator.estimate(
                small_panel,
                utility_small,
                problem_spec_small,
                transitions_small,
            )

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
        result = estimator.estimate(panel, utility, problem_spec_small, transitions_small)

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

        estimator.estimate(panel, utility, problem_spec_small, transitions_small)

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
                atol=1e-5,
            )
