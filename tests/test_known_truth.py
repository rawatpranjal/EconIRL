"""Tests for the compact known-truth synthetic validation harness."""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from econirl.environments.shapeshifter import ShapeshifterConfig
from validation.known_truth import (
    DEFAULT_CELLS,
    ESTIMATOR_CONTRACTS,
    REQUIRED_ESTIMATORS,
    ContentHeterogeneityKnownTruthConfig,
    CounterfactualConfig,
    KnownTruthDGPConfig,
    RecoveryGateFailure,
    ShapeshifterKnownTruthConfig,
    SimulationConfig,
    build_counterfactual,
    build_known_truth_dgp,
    check_estimator_compatibility,
    evaluate_estimator_against_truth,
    get_cell,
    known_truth_initial_params,
    make_estimator,
    normalized_rmse,
    policy_divergence,
    recovery_gates,
    run_estimator,
    run_pre_estimation_diagnostics,
    simulate_known_truth_panel,
    solve_counterfactual_oracle,
    solve_known_truth,
)


def test_low_dim_action_dependent_dgp_solves_and_simulates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            num_regular_states=8,
            transition_noise=0.02,
            seed=10,
        )
    )

    assert dgp.transitions.shape == (3, 9, 9)
    assert dgp.feature_matrix.shape[:2] == (9, 3)
    assert jnp.allclose(dgp.transitions.sum(axis=2), 1.0)
    assert jnp.allclose(dgp.homogeneous_reward[:, dgp.config.exit_action], 0.0)
    assert jnp.allclose(dgp.homogeneous_reward[dgp.config.absorbing_state, :], 0.0)

    solution = solve_known_truth(dgp)
    assert solution.converged
    assert solution.policy.shape == (9, 3)
    assert jnp.allclose(solution.policy.sum(axis=1), 1.0, atol=1e-5)
    assert jnp.isclose(solution.state_occupancy.sum(), 1.0, atol=1e-5)

    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=25, n_periods=12, seed=11),
    )
    assert panel.num_individuals == 25
    assert panel.num_observations == 300

    diagnostics = run_pre_estimation_diagnostics(dgp, panel)
    assert diagnostics.passed
    assert diagnostics.feature_rank == diagnostics.num_features
    assert diagnostics.is_action_dependent


def test_state_only_and_high_dim_modes_are_distinct():
    state_only = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="state_only",
            reward_dim="low",
            num_regular_states=7,
            seed=12,
        )
    )
    state_only_diag = run_pre_estimation_diagnostics(state_only)
    assert state_only_diag.passed
    assert not state_only_diag.is_action_dependent

    high_dim = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            num_regular_states=9,
            high_state_dim=6,
            high_reward_features=10,
            seed=13,
        )
    )
    assert high_dim.state_features.shape == (10, 6)
    assert high_dim.feature_matrix.shape == (10, 3, 10)
    high_dim_diag = run_pre_estimation_diagnostics(high_dim)
    assert high_dim_diag.passed
    assert high_dim_diag.is_action_dependent


def test_canonical_high_dim_action_preset_is_well_conditioned():
    cell = get_cell("canonical_high_action")
    dgp = build_known_truth_dgp(cell.dgp_config)
    diagnostics = run_pre_estimation_diagnostics(dgp)

    assert dgp.state_features.shape == (81, 16)
    assert dgp.feature_matrix.shape == (81, 3, 32)
    assert diagnostics.passed
    assert diagnostics.feature_rank == diagnostics.num_features
    assert diagnostics.condition_number < 10.0
    assert cell.simulation_config.n_individuals == 2_000


def test_gladius_paper_high_state_cell_is_low_reward_high_state():
    cell = get_cell("gladius_paper_high_state")
    dgp = build_known_truth_dgp(cell.dgp_config)
    diagnostics = run_pre_estimation_diagnostics(dgp)

    assert cell.cell_id == "gladius_paper_high_state"
    assert dgp.config.state_mode == "high_dim"
    assert dgp.config.reward_dim == "low"
    assert dgp.state_features.shape == (21, 64)
    assert dgp.feature_matrix.shape == (21, 3, 4)
    assert diagnostics.passed
    assert diagnostics.anchor_valid
    assert diagnostics.is_action_dependent
    assert jnp.allclose(dgp.homogeneous_reward[:, dgp.config.exit_action], 0.0)
    assert cell.simulation_config.n_individuals == 1_000
    assert cell.simulation_config.n_periods == 100


def test_gladius_scaled_high_state_cell_doubles_nuisance_features():
    cell = get_cell("gladius_paper_high_state_scaled")
    dgp = build_known_truth_dgp(cell.dgp_config)
    diagnostics = run_pre_estimation_diagnostics(dgp)

    assert dgp.config.state_mode == "high_dim"
    assert dgp.config.reward_dim == "low"
    assert dgp.state_features.shape == (21, 128)
    assert dgp.feature_matrix.shape == (21, 3, 4)
    assert diagnostics.passed
    assert diagnostics.anchor_valid
    assert diagnostics.feature_rank == diagnostics.num_features
    assert cell.simulation_config.n_individuals == 1_000
    assert cell.simulation_config.n_periods == 100


def test_gladius_known_truth_estimator_receives_anchor_rewards():
    dgp = build_known_truth_dgp(get_cell("gladius_paper_high_state").dgp_config)
    estimator = make_estimator("GLADIUS", dgp, smoke=True)

    assert estimator.config.anchor_action == dgp.config.exit_action
    assert estimator.config.anchor_bellman_loss
    assert estimator.config.anchor_bellman_mode == "anchor_moment"
    assert estimator.config.anchor_rewards is not None
    assert len(estimator.config.anchor_rewards) == dgp.problem.num_states
    assert all(abs(value) < 1e-12 for value in estimator.config.anchor_rewards)


def test_latent_segment_dgp_tracks_segment_truth():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="latent_segments",
            num_regular_states=8,
            high_state_dim=5,
            high_reward_features=9,
            num_segments=2,
            seed=14,
        )
    )
    assert dgp.num_segments == 2
    assert dgp.true_parameters.shape == (2, 9)
    assert dgp.reward_matrix.shape == (2, 9, 3)

    sol0 = solve_known_truth(dgp, segment_index=0)
    sol1 = solve_known_truth(dgp, segment_index=1)
    assert sol0.converged
    assert sol1.converged

    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=15),
    )
    labels = panel.metadata["segment_labels"]
    assert len(labels) == 20
    assert set(labels).issubset({0, 1})


def test_type_a_b_c_counterfactual_oracles_change_truth():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=8, seed=16, transition_noise=0.0)
    )
    baseline = solve_known_truth(dgp)

    for kind in ("type_a", "type_b", "type_c"):
        oracle = solve_counterfactual_oracle(
            dgp,
            kind,
            config=CounterfactualConfig(type_b_skip=3, type_c_action=1),
        )
        assert oracle.baseline_solution.converged
        assert oracle.counterfactual_solution.converged
        divergence = policy_divergence(
            baseline.policy,
            oracle.counterfactual_solution.policy,
        )
        assert divergence.l1 >= 0.0

    cf = build_counterfactual(
        dgp,
        "type_c",
        CounterfactualConfig(type_c_action=1, type_c_penalty=-1_000.0),
    )
    type_c_oracle = solve_counterfactual_oracle(
        dgp,
        "type_c",
        config=CounterfactualConfig(type_c_action=1, type_c_penalty=-1_000.0),
    )
    assert cf.disabled_action == 1
    regular_policy = type_c_oracle.counterfactual_solution.policy[
        : dgp.config.num_regular_states, 1
    ]
    assert float(regular_policy.max()) < 1e-3


def test_estimator_contract_registry_has_required_estimators():
    assert len(REQUIRED_ESTIMATORS) == 13
    assert "BC" not in REQUIRED_ESTIMATORS
    assert "MCE-IRL Deep" in REQUIRED_ESTIMATORS
    assert "AIRL-Het" in REQUIRED_ESTIMATORS
    for name in REQUIRED_ESTIMATORS:
        contract = ESTIMATOR_CONTRACTS[name]
        assert contract.code_path
        assert contract.paper_paths
        assert contract.recovers


def test_estimator_factories_and_compatibility_reports_are_available():
    structural_dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=6, reward_mode="action_dependent")
    )
    state_only_dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=6, reward_mode="state_only")
    )
    hetero_dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="latent_segments",
            num_regular_states=6,
            high_state_dim=4,
            high_reward_features=8,
        )
    )

    for name in REQUIRED_ESTIMATORS:
        dgp = hetero_dgp if name == "AIRL-Het" else structural_dgp
        estimator = make_estimator(name, dgp, smoke=True)
        assert estimator is not None

    assert check_estimator_compatibility("NFXP", structural_dgp).compatible
    assert not check_estimator_compatibility("NFXP", state_only_dgp).compatible
    assert check_estimator_compatibility("MCE-IRL Deep", state_only_dgp).compatible
    assert check_estimator_compatibility("MCE-IRL Deep", structural_dgp).compatible
    assert check_estimator_compatibility("AIRL-Het", hetero_dgp).compatible


def test_nfxp_uses_universal_canonical_preset():
    cell_ids = {cell.cell_id for cell in DEFAULT_CELLS}
    assert {
        "canonical_low_action",
        "canonical_low_state_only",
        "airl_paper_identification",
        "f_irl_paper_state_marginal",
        "airl_het_paper_identification",
        "canonical_high_action",
        "gladius_paper_high_state",
        "gladius_paper_high_state_scaled",
        "canonical_latent_segments",
        "deep_mce_neural_reward",
        "deep_mce_neural_features",
        "deep_mce_neural_reward_features",
    }.issubset(cell_ids)

    dgp = build_known_truth_dgp(get_cell("canonical_low_action").dgp_config)
    assert check_estimator_compatibility("NFXP", dgp).compatible
    solution = solve_known_truth(dgp)
    action_mass = (solution.state_occupancy[:, None] * solution.policy).sum(axis=0)
    assert float(action_mass.min()) > 0.10

    state_only = build_known_truth_dgp(get_cell("canonical_low_state_only").dgp_config)
    high_action = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)
    latent = build_known_truth_dgp(get_cell("canonical_latent_segments").dgp_config)

    assert not check_estimator_compatibility("NFXP", state_only).compatible
    assert not check_estimator_compatibility("NFXP", high_action).compatible
    assert not check_estimator_compatibility("NFXP", latent).compatible


def test_airl_paper_identification_cell_matches_original_assumptions():
    cell = get_cell("airl_paper_identification")
    dgp = build_known_truth_dgp(cell.dgp_config)
    diagnostics = run_pre_estimation_diagnostics(dgp)

    assert cell.cell_id == "airl_paper_identification"
    assert dgp.config.reward_mode == "state_only"
    assert dgp.config.absorbing_state is None
    assert dgp.config.exit_action is None
    assert diagnostics.passed
    assert not diagnostics.is_action_dependent
    assert check_estimator_compatibility("AIRL", dgp).compatible
    assert jnp.allclose(
        dgp.homogeneous_reward,
        dgp.homogeneous_reward[:, :1],
    )
    estimator = make_estimator("AIRL", dgp, smoke=False)
    assert estimator.config.reward_type == "linear"
    assert estimator.config.reward_arg == "state"
    assert estimator.config.generator_reward == "f"
    assert estimator.config.min_rounds == 150
    transition_rows = dgp.transitions.sum(axis=2)
    transition_max = dgp.transitions.max(axis=2)
    assert jnp.allclose(transition_rows, 1.0)
    assert jnp.allclose(transition_max, 1.0)


def test_f_irl_paper_state_marginal_cell_matches_paper_assumptions():
    cell = get_cell("f_irl_paper_state_marginal")
    dgp = build_known_truth_dgp(cell.dgp_config)
    diagnostics = run_pre_estimation_diagnostics(dgp)
    estimator = make_estimator("f-IRL", dgp, smoke=True)

    assert cell.cell_id == "f_irl_paper_state_marginal"
    assert dgp.config.reward_mode == "state_only"
    assert dgp.config.absorbing_state is None
    assert dgp.config.exit_action is None
    assert diagnostics.passed
    assert not diagnostics.is_action_dependent
    assert check_estimator_compatibility("f-IRL", dgp).compatible
    assert jnp.allclose(dgp.homogeneous_reward, dgp.homogeneous_reward[:, :1])
    assert estimator._marginal_space == "state"
    assert estimator._reward_scope == "state"
    transition_rows = dgp.transitions.sum(axis=2)
    transition_max = dgp.transitions.max(axis=2)
    assert jnp.allclose(transition_rows, 1.0)
    assert jnp.allclose(transition_max, 1.0)


def test_airl_het_paper_identification_cell_matches_anchor_assumptions():
    cell = get_cell("airl_het_paper_identification")
    dgp = build_known_truth_dgp(cell.dgp_config)
    panel = simulate_known_truth_panel(dgp, cell.simulation_config)
    diagnostics = run_pre_estimation_diagnostics(dgp)

    assert cell.cell_id == "airl_het_paper_identification"
    assert isinstance(cell.dgp_config, ContentHeterogeneityKnownTruthConfig)
    assert dgp.config.heterogeneity == "latent_segments"
    assert dgp.config.reward_mode == "action_dependent"
    assert dgp.config.exit_action == 2
    assert dgp.config.absorbing_state == dgp.problem.num_states - 1
    assert dgp.num_segments == 2
    assert dgp.config.books_per_user == 4
    assert dgp.state_features.shape[1] == 18
    assert dgp.feature_matrix.shape[-1] == 20
    assert diagnostics.feature_rank == 20
    assert len(panel.trajectories) == (
        cell.simulation_config.n_individuals * dgp.config.books_per_user
    )
    assert len({traj.individual_id for traj in panel.trajectories}) == (
        cell.simulation_config.n_individuals
    )
    assert len(panel.metadata["segment_labels"]) == len(panel.trajectories)
    assert diagnostics.passed
    assert diagnostics.anchor_valid
    assert check_estimator_compatibility("AIRL-Het", dgp).compatible

    estimator = make_estimator("AIRL-Het", dgp, smoke=True)
    assert estimator.config.reward_type == "linear"
    assert estimator.config.generator_reward == "f"
    assert estimator.config.initialization == "behavioral_anchor"
    assert estimator.config.min_airl_rounds == 1


def test_legacy_cell_ids_are_aliases_not_separate_dgps():
    legacy = get_cell("low_state_action_reward")
    canonical = get_cell("canonical_low_action")

    assert legacy.cell_id == "low_state_action_reward"
    assert legacy.dgp_config == canonical.dgp_config


def test_shapeshifter_neural_reward_bridge_has_no_finite_theta_and_full_masks():
    dgp = build_known_truth_dgp(
        ShapeshifterKnownTruthConfig(
            env_config=ShapeshifterConfig(
                num_states=6,
                num_actions=2,
                num_features=3,
                reward_type="neural",
                feature_type="linear",
                action_dependent=False,
                stochastic_transitions=True,
                stochastic_rewards=False,
                discount_factor=0.9,
                state_dim=1,
                network_width=8,
                network_depth=1,
                seed=901,
            )
        )
    )

    assert dgp.config.reward_mode == "neural"
    assert dgp.config.absorbing_state is None
    assert dgp.config.exit_action is None
    assert dgp.homogeneous_parameters.shape == (0,)
    assert dgp.homogeneous_reward.shape == (
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )

    diagnostics = run_pre_estimation_diagnostics(dgp)
    assert diagnostics.passed
    truth = solve_known_truth(dgp)
    summary = SimpleNamespace(
        parameters=jnp.asarray([], dtype=jnp.float32),
        parameter_names=[],
        metadata={
            "reward_matrix": dgp.homogeneous_reward,
            "reward_validation_target": "raw_neural_reward_matrix",
            "counterfactual_reward_normalization": "affine",
            "occupancy_moment_residual": 0.0,
        },
        policy=truth.policy,
        value_function=truth.V,
        converged=True,
    )

    metrics = evaluate_estimator_against_truth(dgp, summary)
    assert metrics["parameters"] is None
    assert metrics["reward_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["policy"].tv == pytest.approx(0.0, abs=1e-10)
    assert metrics["value_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["q_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
    assert set(metrics["counterfactuals"]) == {"type_a", "type_b", "type_c"}

    gate_names = {
        gate.name
        for gate in recovery_gates("MCE-IRL Deep", summary, metrics, smoke=False)
    }
    assert "projected_parameter_cosine" not in gate_names
    assert "projected_parameter_relative_rmse" not in gate_names
    assert "occupancy_moment_residual" in gate_names
    assert all(
        gate.passed
        for gate in recovery_gates("MCE-IRL Deep", summary, metrics, smoke=False)
    )


def test_known_truth_initialization_is_deterministic_and_near_truth():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=6, transition_noise=0.02, seed=111)
    )

    init_a = known_truth_initial_params(dgp)
    init_b = known_truth_initial_params(dgp)
    truth = dgp.homogeneous_parameters

    assert jnp.allclose(init_a, init_b)
    assert init_a.shape == truth.shape
    assert not jnp.allclose(init_a, truth)
    assert float(jnp.linalg.norm(init_a - truth)) < 0.25 * float(jnp.linalg.norm(truth))


def test_nfxp_smoke_fit_produces_known_truth_metrics():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=112,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=113),
    )

    result = run_estimator("NFXP", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["optimizer"] == "BHHH"
    assert result.summary.metadata["inner_solver"] == "hybrid"
    assert result.summary.metadata["num_inner_iterations"] > 0

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_rmse"])
    assert math.isfinite(metrics["value_rmse"])
    assert math.isfinite(metrics["q_rmse"])
    assert metrics["policy"].tv >= 0.0
    assert set(metrics["counterfactuals"]) == {"type_a", "type_b", "type_c"}
    for cf_metrics in metrics["counterfactuals"].values():
        assert cf_metrics.policy.tv >= 0.0
        assert math.isfinite(cf_metrics.value_rmse)
        assert math.isfinite(cf_metrics.regret)


def test_ccp_smoke_fit_produces_known_truth_metrics_and_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=312,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=313),
    )

    result = run_estimator("CCP", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["num_policy_iterations"] == 3
    assert result.summary.metadata["npl_converged"] in {True, False}

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_rmse"])
    assert math.isfinite(metrics["value_rmse"])
    assert math.isfinite(metrics["q_rmse"])
    assert metrics["policy"].tv >= 0.0

    gate_names = {
        gate.name
        for gate in recovery_gates("CCP", result.summary, metrics, smoke=False)
    }
    assert {
        "npl_iterations",
        "standard_errors_finite",
        "parameter_cosine",
        "parameter_relative_rmse",
        "policy_tv",
        "value_rmse",
        "q_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }.issubset(gate_names)


def test_mpec_smoke_fit_produces_known_truth_metrics_and_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=412,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=413),
    )

    result = run_estimator("MPEC", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["method"] == "slsqp"
    assert math.isfinite(result.summary.metadata["final_constraint_violation"])

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_rmse"])
    assert math.isfinite(metrics["value_rmse"])
    assert math.isfinite(metrics["q_rmse"])
    assert metrics["policy"].tv >= 0.0

    gate_names = {
        gate.name
        for gate in recovery_gates("MPEC", result.summary, metrics, smoke=False)
    }
    assert {
        "converged",
        "constraint_violation",
        "standard_errors_finite",
        "parameter_cosine",
        "parameter_relative_rmse",
        "policy_tv",
        "value_rmse",
        "q_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }.issubset(gate_names)


def test_sees_smoke_fit_produces_known_truth_metrics_and_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=512,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=513),
    )

    result = run_estimator("SEES", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["basis_type"] == "bspline"
    assert result.summary.metadata["basis_dim"] == dgp.problem.num_states
    assert math.isfinite(result.summary.metadata["bellman_violation"])

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_rmse"])
    assert math.isfinite(metrics["value_rmse"])
    assert math.isfinite(metrics["q_rmse"])
    assert metrics["policy"].tv >= 0.0

    gate_names = {
        gate.name
        for gate in recovery_gates("SEES", result.summary, metrics, smoke=False)
    }
    assert {
        "bellman_violation",
        "standard_errors_finite",
        "parameter_cosine",
        "parameter_relative_rmse",
        "reward_rmse",
        "policy_tv",
        "value_rmse",
        "q_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }.issubset(gate_names)


def test_nnes_smoke_fit_produces_known_truth_metrics_and_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=612,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=613),
    )

    result = run_estimator("NNES", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["n_outer_iterations"] == 1
    assert len(result.summary.metadata["v_loss_per_outer"]) == 1
    assert math.isfinite(result.summary.metadata["v_loss_per_outer"][-1])
    assert result.summary.metadata["final_ccps"].shape == (
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_rmse"])
    assert math.isfinite(metrics["value_rmse"])
    assert math.isfinite(metrics["q_rmse"])
    assert metrics["policy"].tv >= 0.0

    gate_names = {
        gate.name
        for gate in recovery_gates("NNES", result.summary, metrics, smoke=False)
    }
    assert {
        "npl_outer_iterations",
        "final_v_loss",
        "parameter_cosine",
        "parameter_relative_rmse",
        "reward_rmse",
        "policy_tv",
        "value_rmse",
        "q_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }.issubset(gate_names)


def test_mce_irl_smoke_fit_produces_known_truth_metrics_and_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=712,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=713),
    )

    result = run_estimator("MCE-IRL", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["optimizer"] == "root"
    assert math.isfinite(result.summary.metadata["feature_difference"])
    assert math.isfinite(result.summary.metadata["occupancy_moment_residual"])

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["reward_normalized_rmse"])
    assert math.isfinite(metrics["value_normalized_rmse"])
    assert math.isfinite(metrics["q_normalized_rmse"])
    assert metrics["policy"].tv >= 0.0

    gate_names = {
        gate.name
        for gate in recovery_gates("MCE-IRL", result.summary, metrics, smoke=False)
    }
    assert {
        "converged",
        "feature_residual",
        "occupancy_moment_residual",
        "reward_normalized_rmse",
        "policy_tv",
        "value_normalized_rmse",
        "q_normalized_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }.issubset(gate_names)
    assert "parameter_cosine" not in gate_names
    assert "parameter_relative_rmse" not in gate_names


def test_mce_irl_deep_smoke_fit_produces_known_truth_metrics_and_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="state_only",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=812,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=813),
    )

    result = run_estimator("MCE-IRL Deep", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["estimator"] == "MCE-IRL Deep"
    assert result.summary.metadata["reward_type"] == "state"
    assert result.summary.metadata["projection_r2"] is not None
    assert math.isfinite(result.summary.metadata["feature_difference"])
    assert math.isfinite(result.summary.metadata["occupancy_moment_residual"])
    assert result.summary.metadata["raw_neural_reward_matrix"].shape == (
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_normalized_rmse"])
    assert math.isfinite(metrics["value_normalized_rmse"])
    assert math.isfinite(metrics["q_normalized_rmse"])
    assert metrics["policy"].tv >= 0.0

    gate_names = {
        gate.name
        for gate in recovery_gates("MCE-IRL Deep", result.summary, metrics, smoke=False)
    }
    assert {
        "converged",
        "occupancy_moment_residual",
        "projected_parameter_cosine",
        "projected_parameter_relative_rmse",
        "reward_normalized_rmse",
        "policy_tv",
        "value_normalized_rmse",
        "q_normalized_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }.issubset(gate_names)
    assert "parameter_cosine" not in gate_names
    assert "parameter_relative_rmse" not in gate_names


def test_mce_irl_deep_accepts_shapeshifter_neural_reward_smoke_case():
    dgp = build_known_truth_dgp(
        ShapeshifterKnownTruthConfig(
            env_config=ShapeshifterConfig(
                num_states=6,
                num_actions=2,
                num_features=3,
                reward_type="neural",
                feature_type="linear",
                action_dependent=False,
                stochastic_transitions=True,
                stochastic_rewards=False,
                discount_factor=0.9,
                state_dim=1,
                network_width=8,
                network_depth=1,
                seed=914,
            )
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=8, n_periods=6, seed=915),
    )

    result = run_estimator("MCE-IRL Deep", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.parameters.shape == (0,)
    assert result.metrics["parameters"] is None
    assert result.summary.metadata["reward_type"] == "state"
    assert result.summary.metadata["reward_validation_target"] == "raw_neural_reward_matrix"
    assert result.summary.metadata["reward_matrix"].shape == (
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )
    assert result.summary.policy.shape == (
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )
    assert math.isfinite(result.metrics["reward_normalized_rmse"])
    assert math.isfinite(result.metrics["value_normalized_rmse"])
    assert math.isfinite(result.metrics["q_normalized_rmse"])
    assert set(result.metrics["counterfactuals"]) == {"type_a", "type_b", "type_c"}


def test_airl_smoke_fit_produces_diagnostic_known_truth_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="state_only",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.0,
            seed=932,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=12, n_periods=6, seed=933),
    )

    result = run_estimator("AIRL", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.metadata["reward_arg"] == "state"
    assert result.summary.metadata["generator_reward"] == "recovered"
    assert result.summary.metadata["learned_shaping"]
    assert result.summary.policy.shape == (
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert math.isfinite(result.metrics["reward_normalized_rmse"])
    assert math.isfinite(result.metrics["value_normalized_rmse"])
    assert math.isfinite(result.metrics["q_normalized_rmse"])

    gate_names = {
        gate.name
        for gate in recovery_gates("AIRL", result.summary, result.metrics, smoke=False)
    }
    assert {
        "converged",
        "reward_normalized_rmse",
        "policy_tv",
        "value_normalized_rmse",
        "q_normalized_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    } == gate_names
    assert "parameter_cosine" not in gate_names


def test_f_irl_state_action_non_smoke_gates_are_diagnostic_only():
    summary = SimpleNamespace(
        converged=True,
        metadata={
            "occupancy_l1": 0.16,
            "reward_range": 1.20,
            "marginal_space": "state_action",
            "reward_scope": "state_action",
        },
    )

    gates = recovery_gates("f-IRL", summary, metrics={}, smoke=False)
    gate_names = {gate.name for gate in gates}

    assert gate_names == {"converged", "occupancy_l1", "reward_range"}
    assert "reward_normalized_rmse" not in gate_names
    assert "policy_tv" not in gate_names
    assert all(gate.passed for gate in gates)


def test_f_irl_paper_state_gates_require_structural_recovery():
    summary = SimpleNamespace(
        converged=True,
        metadata={
            "occupancy_l1": 0.01,
            "reward_range": 1.20,
            "marginal_space": "state",
            "reward_scope": "state",
        },
    )
    metrics = {
        "reward_normalized_rmse": 0.20,
        "policy": SimpleNamespace(tv=0.02),
        "value_normalized_rmse": 0.12,
        "q_normalized_rmse": 0.10,
        "counterfactuals": {
            "type_a": SimpleNamespace(regret=0.01),
            "type_b": SimpleNamespace(regret=0.02),
            "type_c": SimpleNamespace(regret=0.01),
        },
    }

    gates = recovery_gates("f-IRL", summary, metrics=metrics, smoke=False)
    gate_names = {gate.name for gate in gates}

    assert {
        "converged",
        "state_marginal_l1",
        "reward_range",
        "reward_normalized_rmse",
        "policy_tv",
        "value_normalized_rmse",
        "q_normalized_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    } == gate_names
    assert all(gate.passed for gate in gates)


def test_gladius_non_smoke_gates_require_structural_recovery():
    summary = SimpleNamespace(
        converged=True,
        metadata={"final_loss": 0.50},
    )
    metrics = {
        "parameters": SimpleNamespace(
            cosine_similarity=0.95,
            relative_rmse=0.20,
        ),
        "policy": SimpleNamespace(tv=0.05),
        "raw_bellman_reward_normalized_rmse": 0.10,
        "projected_reward_normalized_rmse": 0.10,
        "value_normalized_rmse": 0.10,
        "q_normalized_rmse": 0.10,
        "counterfactuals": {
            "type_a": SimpleNamespace(regret=0.02),
            "type_b": SimpleNamespace(regret=0.03),
            "type_c": SimpleNamespace(regret=0.04),
        },
    }

    gates = recovery_gates("GLADIUS", summary, metrics, smoke=False)
    gate_names = {gate.name for gate in gates}

    assert gate_names == {
        "converged",
        "final_loss",
        "parameter_cosine",
        "parameter_relative_rmse",
        "raw_bellman_reward_normalized_rmse",
        "projected_reward_normalized_rmse",
        "policy_tv",
        "value_normalized_rmse",
        "q_normalized_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }
    assert all(gate.passed for gate in gates)

    weak_metrics = dict(metrics)
    weak_metrics["projected_reward_normalized_rmse"] = 0.80
    weak_gates = recovery_gates("GLADIUS", summary, weak_metrics, smoke=False)
    projected_gate = next(
        gate for gate in weak_gates
        if gate.name == "projected_reward_normalized_rmse"
    )
    assert not projected_gate.passed


def test_iq_learn_non_smoke_gates_require_structural_recovery():
    summary = SimpleNamespace(
        converged=True,
        metadata={
            "expert_state_coverage": 1.0,
            "expert_state_action_coverage": 1.0,
        },
    )
    metrics = {
        "policy": SimpleNamespace(tv=0.04),
        "raw_bellman_reward_normalized_rmse": 0.05,
        "projected_reward_normalized_rmse": 0.05,
        "value_normalized_rmse": 0.05,
        "q_normalized_rmse": 0.05,
        "counterfactuals": {
            "type_a": SimpleNamespace(regret=0.01),
            "type_b": SimpleNamespace(regret=0.02),
            "type_c": SimpleNamespace(regret=0.01),
        },
    }

    gates = recovery_gates("IQ-Learn", summary, metrics, smoke=False)
    gate_names = {gate.name for gate in gates}

    assert gate_names == {
        "converged",
        "expert_state_coverage",
        "expert_state_action_coverage",
        "policy_tv",
        "raw_bellman_reward_normalized_rmse",
        "projected_reward_normalized_rmse",
        "value_normalized_rmse",
        "q_normalized_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }
    assert all(gate.passed for gate in gates)

    weak_metrics = dict(metrics)
    weak_metrics["raw_bellman_reward_normalized_rmse"] = 0.40
    weak_gates = recovery_gates("IQ-Learn", summary, weak_metrics, smoke=False)
    raw_gate = next(
        gate for gate in weak_gates
        if gate.name == "raw_bellman_reward_normalized_rmse"
    )
    assert not raw_gate.passed

    sparse_summary = SimpleNamespace(
        converged=True,
        metadata={
            "expert_state_coverage": 2 / 3,
            "expert_state_action_coverage": 0.90,
        },
    )
    sparse_gates = recovery_gates("IQ-Learn", sparse_summary, metrics, smoke=False)
    sparse_by_name = {gate.name: gate for gate in sparse_gates}

    assert not sparse_by_name["expert_state_coverage"].passed
    assert sparse_by_name["expert_state_coverage"].threshold == 1.0
    assert not sparse_by_name["expert_state_action_coverage"].passed
    assert sparse_by_name["expert_state_action_coverage"].threshold == 0.95


def test_iq_learn_known_truth_q_type_selection_matches_dgp_matrix():
    low = build_known_truth_dgp(get_cell("canonical_low_action").dgp_config)
    high = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)

    assert make_estimator("IQ-Learn", low, smoke=False).config.q_type == "tabular"
    assert make_estimator("IQ-Learn", low, smoke=True).config.q_type == "tabular"
    assert make_estimator("IQ-Learn", high, smoke=False).config.q_type == "neural"
    assert make_estimator("IQ-Learn", high, smoke=True).config.q_type == "tabular"


def test_anchor_projected_reward_diagnostic_removes_exit_action_gauge():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            num_regular_states=6,
            seed=943,
        )
    )
    state_offsets = jnp.linspace(-2.0, 2.0, dgp.problem.num_states)[:, None]
    estimated_reward = dgp.homogeneous_reward + state_offsets
    summary = SimpleNamespace(
        parameters=jnp.asarray([], dtype=jnp.float32),
        metadata={"reward_matrix": estimated_reward},
        policy=None,
        value_function=None,
    )

    metrics = evaluate_estimator_against_truth(dgp, summary)

    assert math.isfinite(metrics["reward_normalized_rmse"])
    assert metrics["anchor_projected_reward_normalized_rmse"] == pytest.approx(
        0.0,
        abs=1e-10,
    )
    assert metrics["anchor_projected_reward_rmse"] == pytest.approx(0.0, abs=1e-10)


def test_irl_normalized_rmse_removes_affine_reward_ambiguity_only():
    truth = jnp.array([0.0, 1.0, 2.0, 3.0])
    affine_same = 7.0 + 2.5 * truth
    sign_flipped = 7.0 - 2.5 * truth

    assert normalized_rmse(affine_same, truth) < 1e-8
    assert normalized_rmse(sign_flipped, truth) > 1.0


def test_nfxp_failed_non_smoke_recovery_raises_hard_gate():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=212,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=213),
    )

    with pytest.raises(RecoveryGateFailure):
        run_estimator("NFXP", dgp, panel, smoke=False)
