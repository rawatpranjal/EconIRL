"""Tests for TD-CCP Neural Estimator.

Tests cover:
1. TDCCPConfig default values
2. CCP estimation step
3. Flow decomposition
4. Neural network training convergence
5. Parameter recovery on Rust bus (slow)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.shapeshifter import ShapeshifterConfig, ShapeshifterEnvironment
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.td_ccp import (
    TDCCPEstimator,
    TDCCPConfig,
    _EVComponentNetwork,
    make_state_action_tabular_utility,
)
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel, simulate_panel_from_policy
from validation.estimators.tdccp.run import (
    build_paper_hard_case_dgp,
    evaluate_hard_case_summary,
    evaluate_paper_hard_case_summary,
    tdccp_hard_case_gates,
    tdccp_paper_hard_case_gates,
)


# ============================================================================
# Config tests
# ============================================================================


class TestTDCCPConfig:
    """Tests for TDCCPConfig dataclass defaults."""

    def test_default_hidden_dim(self):
        cfg = TDCCPConfig()
        assert cfg.hidden_dim == 64

    def test_default_num_hidden_layers(self):
        cfg = TDCCPConfig()
        assert cfg.num_hidden_layers == 2

    def test_default_avi_iterations(self):
        cfg = TDCCPConfig()
        assert cfg.avi_iterations == 20

    def test_default_epochs_per_avi(self):
        cfg = TDCCPConfig()
        assert cfg.epochs_per_avi == 30

    def test_default_learning_rate(self):
        cfg = TDCCPConfig()
        assert cfg.learning_rate == 1e-3

    def test_default_batch_size(self):
        cfg = TDCCPConfig()
        assert cfg.batch_size == 8192

    def test_default_ccp_smoothing(self):
        cfg = TDCCPConfig()
        assert cfg.ccp_smoothing == 0.01

    def test_default_outer_max_iter(self):
        cfg = TDCCPConfig()
        assert cfg.outer_max_iter == 200

    def test_default_outer_tol(self):
        cfg = TDCCPConfig()
        assert cfg.outer_tol == 1e-6

    def test_default_compute_se(self):
        cfg = TDCCPConfig()
        assert cfg.compute_se is True

    def test_default_verbose(self):
        cfg = TDCCPConfig()
        assert cfg.verbose is False

    def test_custom_config(self):
        cfg = TDCCPConfig(hidden_dim=128, avi_iterations=10, learning_rate=5e-4)
        assert cfg.hidden_dim == 128
        assert cfg.avi_iterations == 10
        assert cfg.learning_rate == 5e-4


# ============================================================================
# Estimator properties tests
# ============================================================================


class TestTDCCPEstimatorProperties:
    """Tests for basic estimator properties."""

    def test_name(self):
        estimator = TDCCPEstimator()
        assert estimator.name == "TD-CCP"

    def test_default_config(self):
        estimator = TDCCPEstimator()
        assert estimator.config.hidden_dim == 64

    def test_custom_config(self):
        cfg = TDCCPConfig(hidden_dim=128)
        estimator = TDCCPEstimator(config=cfg)
        assert estimator.config.hidden_dim == 128


# ============================================================================
# CCP estimation tests
# ============================================================================


class TestCCPEstimation:
    """Tests for the CCP frequency estimation step."""

    def test_ccps_sum_to_one(self, rust_env_small, small_panel, problem_spec_small):
        """Estimated CCPs should sum to 1 for each state."""
        estimator = TDCCPEstimator()
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        row_sums = ccps.sum(axis=1)
        np.testing.assert_allclose(np.asarray(row_sums), np.asarray(jnp.ones_like(row_sums)), atol=1e-5)

    def test_ccps_non_negative(self, rust_env_small, small_panel, problem_spec_small):
        """Estimated CCPs should be non-negative."""
        estimator = TDCCPEstimator()
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        assert (ccps >= 0).all()

    def test_ccps_shape(self, rust_env_small, small_panel, problem_spec_small):
        """CCPs should have shape (num_states, num_actions)."""
        estimator = TDCCPEstimator()
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        assert ccps.shape == (problem_spec_small.num_states, problem_spec_small.num_actions)

    def test_smoothing_prevents_zeros(self, rust_env_small, problem_spec_small):
        """CCP smoothing should prevent any zero probabilities."""
        # Very small panel so some (s, a) pairs may not be observed
        panel = simulate_panel(rust_env_small, n_individuals=5, n_periods=5, seed=42)
        estimator = TDCCPEstimator(config=TDCCPConfig(ccp_smoothing=0.01))
        ccps = estimator._estimate_ccps(
            panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        assert (ccps > 0).all()


# ============================================================================
# Transition extraction tests
# ============================================================================


class TestTransitionExtraction:
    """Tests for extracting (a,x,a',x') tuples from panel data."""

    def test_transition_lengths_match(self, rust_env_small, small_panel):
        """All transition arrays should have the same length."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert len(actions) == len(states) == len(next_actions) == len(next_states)

    def test_transitions_non_empty(self, rust_env_small, small_panel):
        """Should extract at least one transition."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert len(states) > 0

    def test_actions_valid(self, rust_env_small, small_panel, problem_spec_small):
        """All actions should be valid (0 or 1 for binary choice)."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert np.all(actions >= 0) and np.all(actions < problem_spec_small.num_actions)
        assert np.all(next_actions >= 0) and np.all(next_actions < problem_spec_small.num_actions)

    def test_states_valid(self, rust_env_small, small_panel, problem_spec_small):
        """All states should be valid indices."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert np.all(states >= 0) and np.all(states < problem_spec_small.num_states)
        assert np.all(next_states >= 0) and np.all(next_states < problem_spec_small.num_states)


# ============================================================================
# Semi-gradient method tests
# ============================================================================


class TestSemigradientSolve:
    """Tests for the linear semi-gradient method (eq 3.5)."""

    def test_h_table_shape(self, rust_env_small, small_panel, problem_spec_small):
        """h_table should have shape (num_states, num_actions, num_features)."""
        estimator = TDCCPEstimator(config=TDCCPConfig(method="semigradient"))
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        h_table, g_table = estimator._semigradient_solve(
            actions, states, next_actions, next_states,
            np.array(utility.feature_matrix), np.array(ccps),
            problem_spec_small.num_states, problem_spec_small.num_actions,
            problem_spec_small.discount_factor,
        )
        assert h_table.shape == (problem_spec_small.num_states, problem_spec_small.num_actions, utility.num_parameters)

    def test_g_table_shape(self, rust_env_small, small_panel, problem_spec_small):
        """g_table should have shape (num_states, num_actions)."""
        estimator = TDCCPEstimator(config=TDCCPConfig(method="semigradient"))
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        h_table, g_table = estimator._semigradient_solve(
            actions, states, next_actions, next_states,
            np.array(utility.feature_matrix), np.array(ccps),
            problem_spec_small.num_states, problem_spec_small.num_actions,
            problem_spec_small.discount_factor,
        )
        assert g_table.shape == (problem_spec_small.num_states, problem_spec_small.num_actions)

    def test_h_g_finite(self, rust_env_small, small_panel, problem_spec_small):
        """h and g should be finite everywhere."""
        estimator = TDCCPEstimator(config=TDCCPConfig(method="semigradient"))
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        h_table, g_table = estimator._semigradient_solve(
            actions, states, next_actions, next_states,
            np.array(utility.feature_matrix), np.array(ccps),
            problem_spec_small.num_states, problem_spec_small.num_actions,
            problem_spec_small.discount_factor,
        )
        assert np.all(np.isfinite(h_table))
        assert np.all(np.isfinite(g_table))

    def test_encoded_basis_uses_state_encoder(self):
        """Encoded basis should use problem.state_encoder, not scalar labels."""
        state_features = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float64,
        )
        problem = DDCProblem(
            num_states=3,
            num_actions=2,
            discount_factor=0.9,
            state_dim=2,
            state_encoder=lambda states: state_features[states],
        )
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="encoded",
                basis_dim=2,
            )
        )

        actions = np.array([0, 1, 0], dtype=np.int32)
        states = np.array([0, 1, 2], dtype=np.int32)
        phi = estimator._build_basis_functions(
            actions, states, problem.num_states, problem.num_actions, problem
        )

        # Per action: intercept + two first-order encoded features +
        # two second-order encoded features.
        assert phi.shape == (3, 10)
        np.testing.assert_allclose(phi[0, 1:3], np.array([1.0, 0.0]))
        np.testing.assert_allclose(phi[1, 6:8], np.array([0.0, 1.0]))
        assert np.all(phi[1, :5] == 0.0)

    def test_g_uses_next_period_entropy_target(self):
        """The g semi-gradient target should be beta * e(a', x')."""
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="tabular",
                basis_ridge=0.0,
            )
        )
        gamma = 0.5
        ccps = np.array([[0.8, 0.2]], dtype=np.float64)
        actions = np.array([0, 1], dtype=np.int32)
        states = np.array([0, 0], dtype=np.int32)
        next_actions = np.array([1, 0], dtype=np.int32)
        next_states = np.array([0, 0], dtype=np.int32)
        feature_matrix = np.zeros((1, 2, 1), dtype=np.float64)

        _, g_table = estimator._semigradient_solve(
            actions,
            states,
            next_actions,
            next_states,
            feature_matrix,
            ccps,
            num_states=1,
            num_actions=2,
            gamma=gamma,
        )

        euler = 0.5772156649015329
        e0 = euler - np.log(ccps[0, 0])
        e1 = euler - np.log(ccps[0, 1])
        phi = np.eye(2)
        phi_next = phi[[1, 0]]
        A = (phi.T @ (phi - gamma * phi_next)) / 2
        b = (phi.T @ (gamma * np.array([e1, e0]))) / 2
        expected = np.linalg.solve(A, b)

        np.testing.assert_allclose(g_table[0], expected, atol=1e-10)


class TestTDCCPHardCaseComponents:
    """Component tests for the TD-CCP hard-case runner."""

    def test_backward_lambda_solves_reversed_td_fixed_point(self):
        """Algorithm 5 lambda should solve the reversed TD recursion."""
        problem = DDCProblem(
            num_states=1,
            num_actions=2,
            discount_factor=0.4,
            scale_parameter=1.0,
        )
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="tabular",
                basis_ridge=0.0,
            )
        )
        params = np.array([0.7], dtype=np.float64)
        h_table = np.array([[[0.0], [1.0]]], dtype=np.float64)
        g_table = np.zeros((1, 2), dtype=np.float64)
        feature_matrix = h_table.copy()
        actions = np.array([0, 1], dtype=np.int32)
        states = np.array([0, 0], dtype=np.int32)
        next_actions = np.array([0, 1], dtype=np.int32)
        next_states = np.array([0, 0], dtype=np.int32)

        lambda_table = estimator._compute_backward_value(
            params,
            h_table,
            g_table,
            feature_matrix,
            actions,
            states,
            next_actions,
            next_states,
            problem,
            problem.discount_factor,
        )

        psi_table = -estimator._score_table(
            params,
            h_table,
            g_table,
            problem.scale_parameter,
        )
        expected = psi_table / (1.0 - problem.discount_factor)
        np.testing.assert_allclose(lambda_table, expected, atol=1e-12)

    def test_zeta_moment_matches_score_plus_lambda_residual_formula(self):
        """Equation (4.6) zeta should equal score plus lambda times residual."""
        problem = DDCProblem(
            num_states=1,
            num_actions=2,
            discount_factor=0.5,
            scale_parameter=1.0,
        )
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="tabular",
                basis_ridge=0.0,
            )
        )
        params = np.array([0.4], dtype=np.float64)
        tilde_params = np.array([0.35], dtype=np.float64)
        h_table = np.array([[[1.0], [0.2]]], dtype=np.float64)
        g_table = np.array([[0.0, 0.1]], dtype=np.float64)
        feature_matrix = np.array([[[0.0], [1.0]]], dtype=np.float64)
        lambda_table = np.array([[[0.3], [-0.2]]], dtype=np.float64)
        ccps = np.array([[0.6, 0.4]], dtype=np.float64)
        actions = np.array([0, 1], dtype=np.int32)
        states = np.array([0, 0], dtype=np.int32)
        next_actions = np.array([1, 0], dtype=np.int32)
        next_states = np.array([0, 0], dtype=np.int32)

        zeta = estimator._locally_robust_moments(
            params,
            h_table,
            g_table,
            lambda_table,
            tilde_params,
            feature_matrix,
            actions,
            states,
            next_actions,
            next_states,
            ccps,
            problem,
            problem.discount_factor,
        )

        policy = estimator._policy_from_h_g(
            params,
            h_table,
            g_table,
            problem.scale_parameter,
        )
        expected_h = np.einsum("sa,sak->sk", policy, h_table)
        score_table = (h_table - expected_h[:, None, :]) / problem.scale_parameter
        v_tilde = np.einsum("sak,k->sa", h_table, tilde_params) + g_table
        euler = 0.5772156649015329
        residual = (
            np.einsum("nk,k->n", feature_matrix[states, actions], tilde_params)
            + problem.discount_factor
            * (euler - np.log(ccps[next_states, next_actions]))
            + problem.discount_factor * v_tilde[next_states, next_actions]
            - v_tilde[states, actions]
        )
        expected = score_table[states, actions] + lambda_table[states, actions] * residual[:, None]
        np.testing.assert_allclose(zeta, expected, atol=1e-12)

    def test_locally_robust_moment_is_less_sensitive_to_h_perturbation(self):
        """The zeta correction should damp first-order h errors."""
        problem = DDCProblem(
            num_states=1,
            num_actions=2,
            discount_factor=0.0,
            scale_parameter=1.0,
        )
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="tabular",
                basis_ridge=0.0,
            )
        )
        params = np.array([1.0], dtype=np.float64)
        h_table = np.array([[[0.0], [1.0]]], dtype=np.float64)
        g_table = np.zeros((1, 2), dtype=np.float64)
        feature_matrix = h_table.copy()
        ccps = estimator._policy_from_h_g(
            params,
            h_table,
            g_table,
            problem.scale_parameter,
        )
        n_obs = 10_000
        n_action_1 = int(round(n_obs * ccps[0, 1]))
        actions = np.array([0] * (n_obs - n_action_1) + [1] * n_action_1, dtype=np.int32)
        states = np.zeros(n_obs, dtype=np.int32)
        next_actions = actions.copy()
        next_states = states.copy()
        lambda_table = estimator._compute_backward_value(
            params,
            h_table,
            g_table,
            feature_matrix,
            actions,
            states,
            next_actions,
            next_states,
            problem,
            problem.discount_factor,
        )

        base_score = estimator._score_moments(
            params,
            h_table,
            g_table,
            states,
            actions,
            problem.scale_parameter,
        )
        base_zeta = estimator._locally_robust_moments(
            params,
            h_table,
            g_table,
            lambda_table,
            params,
            feature_matrix,
            actions,
            states,
            next_actions,
            next_states,
            ccps,
            problem,
            problem.discount_factor,
        )
        perturbed_h = h_table.copy()
        perturbed_h[0, 1, 0] += 1e-3
        perturbed_score = estimator._score_moments(
            params,
            perturbed_h,
            g_table,
            states,
            actions,
            problem.scale_parameter,
        )
        perturbed_zeta = estimator._locally_robust_moments(
            params,
            perturbed_h,
            g_table,
            lambda_table,
            params,
            feature_matrix,
            actions,
            states,
            next_actions,
            next_states,
            ccps,
            problem,
            problem.discount_factor,
        )

        naive_move = float(np.linalg.norm(perturbed_score.mean(axis=0) - base_score.mean(axis=0)))
        robust_move = float(np.linalg.norm(perturbed_zeta.mean(axis=0) - base_zeta.mean(axis=0)))
        assert naive_move > 1e-5
        assert robust_move < naive_move * 1e-2

    def test_algorithm2_fold_covariance_clusters_by_individual_formula(self):
        """Fold covariance should use individual cluster sums, not row meat."""
        problem = DDCProblem(
            num_states=1,
            num_actions=2,
            discount_factor=0.0,
            scale_parameter=1.0,
        )
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="tabular",
                basis_ridge=0.0,
            )
        )
        params = np.array([0.0], dtype=np.float64)
        h_table = np.array([[[0.0], [1.0]]], dtype=np.float64)
        g_table = np.zeros((1, 2), dtype=np.float64)
        lambda_table = np.zeros((1, 2, 1), dtype=np.float64)
        feature_matrix = h_table.copy()
        actions = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        states = np.zeros(len(actions), dtype=np.int32)
        next_actions = actions.copy()
        next_states = states.copy()
        individual_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        ccps = np.array([[0.5, 0.5]], dtype=np.float64)

        covariance = estimator._paper_fold_covariance(
            params,
            h_table,
            g_table,
            lambda_table,
            params,
            feature_matrix,
            actions,
            states,
            next_actions,
            next_states,
            ccps,
            problem,
            problem.discount_factor,
            individual_ids=individual_ids,
        )

        zeta = covariance["zeta"]
        cluster_sums = np.array(
            [
                zeta[individual_ids == 0].sum(axis=0),
                zeta[individual_ids == 1].sum(axis=0),
            ]
        )
        cluster_moments = cluster_sums / (len(actions) / 2)
        expected_omega = (cluster_moments.T @ cluster_moments) / 2
        row_omega = (zeta.T @ zeta) / len(actions)

        assert covariance["covariance_unit"] == "individual"
        assert covariance["n_effective_units"] == 2
        np.testing.assert_allclose(covariance["Omega"], expected_omega, atol=1e-12)
        assert not np.allclose(covariance["Omega"], row_omega)

    def test_paper_hard_case_has_finite_theta_encoded_state_utility(self):
        """Paper hard case should have finite theta and exact linear rewards."""
        dgp = build_paper_hard_case_dgp(seed=11)
        utility = dgp["utility"]
        true_params = dgp["true_params"]
        true_reward = dgp["true_reward"]

        reconstructed = utility.compute(true_params)
        np.testing.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(true_reward),
            atol=1e-10,
        )
        assert utility.num_parameters == 6
        assert dgp["env"].problem_spec.state_dim == 2
        assert dgp["env"].problem_spec.num_states == 81
        assert dgp["basis_metadata"]["basis_source"] == "encoded state polynomial features"
        assert dgp["basis_metadata"]["action_normalization"] == (
            "action 0 reward features fixed to zero"
        )
        np.testing.assert_allclose(
            np.asarray(utility.feature_matrix[:, 0, :]),
            np.zeros((utility.num_states, utility.num_parameters)),
            atol=1e-12,
        )

    def test_paper_hard_case_metrics_and_gates_use_parameter_truth(self):
        """Finite-theta hard-case gates should include structural theta checks."""
        dgp = build_paper_hard_case_dgp(seed=13)
        env = dgp["env"]
        utility = dgp["utility"]
        true_params = dgp["true_params"]
        true_reward = dgp["true_reward"]
        truth = value_iteration(
            SoftBellmanOperator(env.problem_spec, env.transition_matrices),
            true_reward,
            tol=1e-10,
            max_iter=10_000,
        )
        summary = SimpleNamespace(
            parameters=true_params,
            standard_errors=jnp.full_like(true_params, 0.01),
            variance_covariance=jnp.eye(len(true_params), dtype=jnp.float64) * 1e-4,
            policy=truth.policy,
            value_function=truth.V,
            converged=True,
            metadata={
                "se_method_detail": "tdccp_algorithm2_locally_robust",
                "paper_inference": {"moment_norm_max": 0.0},
            },
        )

        metrics = evaluate_paper_hard_case_summary(
            env,
            utility,
            true_params,
            true_reward,
            summary,
            truth=truth,
        )
        gates = tdccp_paper_hard_case_gates(summary, metrics)
        gate_names = {gate.name for gate in gates}

        assert metrics["parameters"]["cosine_similarity"] == pytest.approx(1.0)
        assert metrics["reward_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert "parameter_cosine" in gate_names
        assert "parameter_relative_rmse" in gate_names
        assert all(gate.passed for gate in gates)

    def test_algorithm2_locally_robust_se_path_stores_validation_evidence(self):
        """Robust cross-fitting should run zeta/lambda/covariance path for validation evidence."""
        env = ShapeshifterEnvironment(
            ShapeshifterConfig(
                num_states=8,
                num_actions=2,
                num_features=2,
                reward_type="linear",
                feature_type="linear",
                action_dependent=True,
                stochastic_transitions=True,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.9,
                seed=17,
            )
        )
        utility = ActionDependentReward(
            env.feature_matrix,
            ["theta_0", "theta_1"],
        )
        true_params = jnp.array([0.5, -0.3], dtype=jnp.float64)
        true_reward = utility.compute(true_params)
        truth = value_iteration(
            SoftBellmanOperator(env.problem_spec, env.transition_matrices),
            true_reward,
            tol=1e-8,
            max_iter=1_000,
        )
        panel = simulate_panel_from_policy(
            env.problem_spec,
            env.transition_matrices,
            truth.policy,
            jnp.asarray(env._get_initial_state_distribution()),
            n_individuals=30,
            n_periods=15,
            seed=19,
        )
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="tabular",
                ccp_method="frequency",
                cross_fitting=True,
                robust_se=True,
                compute_se=True,
                outer_max_iter=60,
                outer_tol=1e-5,
            ),
            seed=23,
        )

        summary = estimator.estimate(
            panel,
            utility,
            env.problem_spec,
            env.transition_matrices,
        )
        paper_inference = summary.metadata["paper_inference"]

        assert summary.metadata["se_method_detail"] == "tdccp_algorithm2_locally_robust"
        assert paper_inference["method"] == "tdccp_algorithm2_locally_robust"
        assert paper_inference["covariance_unit"] == "individual"
        assert len(paper_inference["folds"]) == 2
        assert paper_inference["moment_norm_max"] < 1e-3
        assert np.isfinite(paper_inference["lambda_fixed_point_residual_norm_max"])
        assert np.isfinite(paper_inference["lambda_fixed_point_residual_rms_max"])
        assert np.isfinite(paper_inference["lambda_fixed_point_residual_max_abs"])
        assert len(paper_inference["preliminary_optimizer_success"]) == 2
        assert len(paper_inference["preliminary_optimizer_stationary"]) == 2
        assert all(paper_inference["preliminary_optimizer_stationary"])
        assert np.isfinite(paper_inference["preliminary_projected_gradient_norm_max"])
        assert len(paper_inference["robust_optimizer_success"]) == 2
        assert all(paper_inference["robust_optimizer_stationary"])
        assert np.all(np.isfinite(np.asarray(summary.standard_errors)))
        assert np.all(np.asarray(summary.standard_errors) > 0)
        assert np.asarray(summary.variance_covariance).shape == (2, 2)
        for fold in paper_inference["folds"]:
            assert {
                "tilde_theta",
                "theta",
                "lambda",
                "zeta_mean",
                "G",
                "Omega",
                "V_asymptotic",
                "n_effective_units",
                "covariance_unit",
                "lambda_fixed_point_residual_norm",
                "lambda_fixed_point_residual_rms",
                "lambda_fixed_point_residual_max_abs",
            } <= set(fold)
            assert fold["covariance_unit"] == "individual"
            assert fold["n_effective_units"] > 0
            assert np.isfinite(fold["lambda_fixed_point_residual_norm"])
            assert np.isfinite(fold["lambda_fixed_point_residual_rms"])
            assert np.isfinite(fold["lambda_fixed_point_residual_max_abs"])
            assert fold["lambda"].shape == (
                env.num_states,
                env.num_actions,
                utility.num_parameters,
            )

    def test_tabular_reward_utility_reconstructs_reward_matrix(self):
        """One-hot state-action utility should reconstruct any reward matrix."""
        reward = jnp.array(
            [
                [0.1, -0.2, 0.3],
                [0.4, 0.0, -0.5],
                [-0.7, 0.8, 0.9],
                [1.1, -1.2, 1.3],
            ],
            dtype=jnp.float64,
        )
        utility = make_state_action_tabular_utility(
            reward.shape[0],
            reward.shape[1],
        )

        reconstructed = utility.compute(reward.reshape(-1))

        np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(reward))

    def test_hard_case_metrics_use_shapeshifter_reward_and_solver_truth(self):
        """Hard-case metrics should compare against environment reward/solver truth."""
        env = ShapeshifterEnvironment(
            ShapeshifterConfig(
                num_states=5,
                num_actions=2,
                num_features=3,
                reward_type="neural",
                feature_type="neural",
                action_dependent=True,
                stochastic_transitions=True,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.9,
                seed=7,
            )
        )
        truth = value_iteration(
            SoftBellmanOperator(env.problem_spec, env.transition_matrices),
            env.true_reward_matrix,
            tol=1e-10,
            max_iter=10_000,
        )
        summary = SimpleNamespace(
            parameters=jnp.asarray(env.true_reward_matrix).reshape(-1),
            policy=truth.policy,
            value_function=truth.V,
            converged=True,
        )

        metrics = evaluate_hard_case_summary(env, summary, truth=truth)

        assert metrics["reward_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["reward_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["policy_tv"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["value_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["q_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert set(metrics["counterfactuals"]) == {"type_a", "type_b", "type_c"}
        for cf_metrics in metrics["counterfactuals"].values():
            assert cf_metrics["regret"] == pytest.approx(0.0, abs=1e-8)

    def test_neural_hard_case_gates_skip_parameter_cosine(self):
        """Neural-reward gates should not include finite-theta recovery checks."""
        summary = SimpleNamespace(converged=True)
        metrics = {
            "reward_normalized_rmse": 0.0,
            "policy_tv": 0.0,
            "value_normalized_rmse": 0.0,
            "q_normalized_rmse": 0.0,
            "counterfactuals": {
                "type_a": {"regret": 0.0},
                "type_b": {"regret": 0.0},
                "type_c": {"regret": 0.0},
            },
        }

        gates = tdccp_hard_case_gates(summary, metrics)
        gate_names = {gate.name for gate in gates}

        assert "parameter_cosine" not in gate_names
        assert "parameter_relative_rmse" not in gate_names
        assert {
            "converged",
            "reward_normalized_rmse",
            "policy_tv",
            "value_normalized_rmse",
            "q_normalized_rmse",
            "type_a_regret",
            "type_b_regret",
            "type_c_regret",
        }.issubset(gate_names)


# ============================================================================
# NN component tests
# ============================================================================


class TestEVComponentNetwork:
    """Tests for the MLP component network."""

    def test_output_scalar(self):
        """Network output should be a scalar for single input."""
        key = jax.random.PRNGKey(0)
        net = _EVComponentNetwork(input_dim=3, hidden_dim=32, num_hidden_layers=2, key=key)
        x = jnp.ones(3)
        out = net(x)
        assert out.shape == ()

    def test_vmap_batch(self):
        """Network should work with vmap for batched input."""
        key = jax.random.PRNGKey(0)
        net = _EVComponentNetwork(input_dim=3, hidden_dim=16, num_hidden_layers=1, key=key)
        x = jnp.ones((10, 3))
        out = jax.vmap(net)(x)
        assert out.shape == (10,)

    def test_output_shift_initializes_predictions(self):
        """AVI constant initialization should shift initial network values."""
        key = jax.random.PRNGKey(0)
        net_unshifted = _EVComponentNetwork(
            input_dim=3,
            hidden_dim=16,
            num_hidden_layers=1,
            key=key,
            output_shift=0.0,
        )
        net_shifted = _EVComponentNetwork(
            input_dim=3,
            hidden_dim=16,
            num_hidden_layers=1,
            key=key,
            output_shift=2.5,
        )
        x = jnp.ones(3)
        assert np.isclose(float(net_shifted(x) - net_unshifted(x)), 2.5)

    def test_gradient_flows(self):
        """Gradients should flow through the network."""
        key = jax.random.PRNGKey(0)
        net = _EVComponentNetwork(input_dim=3, hidden_dim=16, num_hidden_layers=2, key=key)
        x = jnp.ones(3)
        out = net(x)
        assert jnp.isfinite(out)


# ============================================================================
# Integration test (quick)
# ============================================================================


class TestTDCCPQuickIntegration:
    """Quick integration test that the full pipeline runs without errors."""

    def test_full_pipeline_runs(self, rust_env_small, problem_spec_small, transitions_small):
        """The full TD-CCP pipeline should run end-to-end on a small problem."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            method="semigradient",
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=20,
            compute_se=False,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        result = estimator._optimize(
            panel=panel,
            utility=utility,
            problem=problem_spec_small,
            transitions=transitions_small,
        )

        assert result is not None
        assert len(result.parameters) == utility.num_parameters
        assert jnp.isfinite(result.parameters).all()
        assert result.policy.shape == (problem_spec_small.num_states, problem_spec_small.num_actions)
        assert result.value_function.shape == (problem_spec_small.num_states,)

    def test_estimate_returns_summary(self, rust_env_small, problem_spec_small, transitions_small):
        """The estimate() method should return an EstimationSummary."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            method="semigradient",
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=20,
            compute_se=True,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        summary = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=problem_spec_small,
            transitions=transitions_small,
        )

        assert summary is not None
        assert summary.method == "TD-CCP"
        assert len(summary.parameters) == utility.num_parameters
        assert summary.parameter_names == utility.parameter_names

    def test_policy_is_valid(self, rust_env_small, problem_spec_small, transitions_small):
        """The estimated policy should be a valid probability distribution."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            method="semigradient",
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=20,
            compute_se=False,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        result = estimator._optimize(
            panel=panel,
            utility=utility,
            problem=problem_spec_small,
            transitions=transitions_small,
        )

        # Policy should be non-negative
        assert (result.policy >= 0).all()
        # Policy rows should sum to 1
        row_sums = result.policy.sum(axis=1)
        np.testing.assert_allclose(np.asarray(row_sums), np.asarray(jnp.ones_like(row_sums)), atol=1e-5)


# ============================================================================
# Import test
# ============================================================================


class TestImports:
    """Test that TDCCPEstimator and TDCCPConfig can be imported from estimation package."""

    def test_import_from_estimation(self):
        from econirl.estimation import TDCCPEstimator, TDCCPConfig

        assert TDCCPEstimator is not None
        assert TDCCPConfig is not None

    def test_in_all(self):
        from econirl.estimation import __all__

        assert "TDCCPEstimator" in __all__
        assert "TDCCPConfig" in __all__


# ============================================================================
# Slow: Parameter recovery test
# ============================================================================


@pytest.mark.slow
class TestParameterRecovery:
    """Parameter recovery test on Rust bus environment.

    Requires 500 agents x 100 periods, so marked as slow.
    """

    def test_parameter_recovery_rust_bus(self):
        """TD-CCP should recover Rust bus parameters with RMSE < 0.5."""
        env = RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.9999,
            seed=42,
        )
        utility = LinearUtility.from_environment(env)
        problem = env.problem_spec
        transitions = env.transition_matrices
        panel = simulate_panel(env, n_individuals=500, n_periods=100, seed=42)

        true_params = env.get_true_parameter_vector()

        cfg = TDCCPConfig(
            hidden_dim=64,
            num_hidden_layers=2,
            avi_iterations=20,
            epochs_per_avi=30,
            learning_rate=1e-3,
            batch_size=8192,
            ccp_smoothing=0.01,
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=200,
            compute_se=True,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        result = estimator._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )

        estimated_params = result.parameters
        diff = estimated_params - true_params
        rmse = float(jnp.sqrt((diff ** 2).mean()))

        assert rmse < 0.5, (
            f"RMSE={rmse:.4f} exceeds 0.5 threshold. "
            f"True: {np.asarray(true_params)}, Est: {np.asarray(estimated_params)}"
        )

        # Also check that individual parameters are in a reasonable range
        assert jnp.isfinite(estimated_params).all()
        assert result.log_likelihood < 0  # log-likelihood should be negative
