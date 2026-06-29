"""Validation evidence contract checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_archived_rust_benchmark_artifact_matches_source_csv() -> None:
    source = ROOT / "examples" / "rust-bus-engine" / "benchmark_results.csv"
    artifact = ROOT / "validation" / "results" / "rust_cross_estimator_benchmark_archived.json"

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(source.read_text(encoding="utf-8").splitlines()))

    assert payload["release_status"] == "archived_non_release_evidence"
    assert payload["source_csv"] == "examples/rust-bus-engine/benchmark_results.csv"
    assert payload["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["rows"] == rows
    assert "Do not use it as release certification" in payload["warning"]


def test_mpec_high_beta_smoke_artifact_uses_current_sqp_path() -> None:
    """MPEC high-beta local evidence must be scoped and solver-explicit."""

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    artifact = ROOT / "validation" / "results" / "mpec_high_beta_smoke.json"
    payload = json.loads(
        artifact.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    result = payload["result"]
    thresholds = payload["thresholds"]

    assert payload["artifact_name"] == "mpec_high_beta_smoke"
    assert payload["artifact_type"] == "local_smoke_guard"
    assert payload["release_status"] == "local_smoke_only_not_tier4_release_evidence"
    assert payload["does_not_replace"] == "tier4_high_gamma_mpec"
    assert payload["environment_config"]["discount_factor"] >= 0.999
    assert payload["environment_config"]["num_mileage_bins"] == 4
    assert payload["solver_config"]["solver"] == "sqp"
    assert payload["solver_config"]["compute_hessian"] is False
    assert result["passed"] is True
    assert result["converged"] is True
    assert result["method"] == "slsqp"
    assert result["final_constraint_violation"] < thresholds["final_constraint_violation_max"]
    assert result["num_iterations"] <= thresholds["num_iterations_max"]
    assert 0 < result["estimation_time"] < thresholds["estimation_time_max_seconds"]
    assert 0 < result["wall_time_seconds"] < thresholds["wall_time_max_seconds"]
    assert result["num_observations"] == (
        payload["simulation_config"]["n_individuals"] * payload["simulation_config"]["n_periods"]
    )
    assert result["num_observations"] > 0
    assert result["parameters"]
    assert all(math.isfinite(value) for value in result["parameters"])


def test_tdccp_primer_json_artifact_has_release_gates() -> None:
    """TD-CCP primer JSON should expose the paper-faithful release evidence."""

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    artifact = ROOT / "validation" / "results" / "tdccp.json"

    assert artifact.exists()

    payload = json.loads(
        artifact.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    result = payload["result"]
    summary = result["summary"]
    metrics = result["metrics"]
    metadata = summary["metadata"]
    paper_inference = summary["paper_inference"]
    monte_carlo = payload["monte_carlo"]
    gate_names = {gate["name"] for gate in result["gates"]}
    counterfactuals = metrics["counterfactuals"]

    assert {
        "estimator",
        "paper_target",
        "primary_cell_id",
        "release_status",
        "result",
        "monte_carlo",
    } <= set(payload)
    assert payload["estimator"] == "TD-CCP"
    assert payload["release_status"] == "Certified with caveat"
    assert payload["primary_cell_id"] == "shapeshifter_encoded_state_locally_robust"
    assert result["case_id"] == payload["primary_cell_id"]
    assert result["diagnostics"]["role"] == "certified"
    assert result["diagnostics"]["passed"] is True
    assert result["diagnostics"]["gate_fail_count"] == 0
    assert result["diagnostics"]["gate_pass_count"] == len(result["gates"])
    assert summary["converged"] is True
    assert summary["num_observations"] > 0
    assert summary["estimation_time"] > 0
    assert metadata["basis_type"] == "encoded"
    assert metadata["ccp_method"] == "logit"
    assert metadata["cross_fitting"] is True
    assert metadata["robust_se"] is True
    assert metadata["se_method_detail"] == "tdccp_algorithm2_locally_robust"
    assert paper_inference["method"] == "tdccp_algorithm2_locally_robust"
    assert paper_inference["covariance_unit"] == "individual"
    assert paper_inference["moment_norm_max"] < 1e-4
    assert math.isfinite(paper_inference["lambda_fixed_point_residual_norm_max"])
    assert math.isfinite(paper_inference["lambda_fixed_point_residual_rms_max"])
    assert math.isfinite(paper_inference["lambda_fixed_point_residual_max_abs"])
    assert math.isfinite(paper_inference["preliminary_projected_gradient_norm_max"])
    assert (
        paper_inference["preliminary_projected_gradient_norm_max"]
        <= (paper_inference["preliminary_stationarity_tol"])
    )
    assert len(paper_inference["preliminary_optimizer_success"]) == 2
    assert len(paper_inference["preliminary_optimizer_messages"]) == 2
    assert len(paper_inference["preliminary_optimizer_diagnostics"]) == 2
    assert len(paper_inference["preliminary_optimizer_stationary"]) == 2
    assert all(paper_inference["preliminary_optimizer_stationary"])
    assert len(paper_inference["robust_optimizer_success"]) == 2
    assert len(paper_inference["robust_optimizer_stationary"]) == 2
    assert all(paper_inference["robust_optimizer_success"])
    assert all(paper_inference["robust_optimizer_stationary"])
    assert len(paper_inference["folds"]) == 2
    assert len(summary["standard_errors"]) == len(summary["parameters"])
    assert set(summary["confidence_intervals_95"]) == set(summary["parameter_names"])
    assert all(se is not None and se > 0 for se in summary["standard_errors"])
    assert all(math.isfinite(x) and x > 0 for x in paper_inference["sample_covariance_diag"])
    for fold in paper_inference["folds"]:
        assert fold["covariance_unit"] == "individual"
        assert fold["n_effective_units"] > 0
        assert math.isfinite(fold["lambda_fixed_point_residual_norm"])
        assert math.isfinite(fold["lambda_fixed_point_residual_rms"])
        assert math.isfinite(fold["lambda_fixed_point_residual_max_abs"])
    assert metrics["parameters"]["cosine_similarity"] > 0.99
    assert metrics["parameters"]["relative_rmse"] < 0.15
    assert metrics["reward_normalized_rmse"] < 0.08
    assert metrics["policy_tv"] < 0.03
    assert metrics["value_normalized_rmse"] < 0.10
    assert metrics["q_normalized_rmse"] < 0.10
    assert all(gate["passed"] for gate in result["gates"])
    assert set(counterfactuals) == {"type_a", "type_b", "type_c"}
    for cf_metrics in counterfactuals.values():
        assert math.isfinite(cf_metrics["regret"])
        assert math.isfinite(cf_metrics["value_rmse"])
        assert math.isfinite(cf_metrics["policy_tv"])
        assert cf_metrics["regret"] < 0.05
        assert cf_metrics["policy_tv"] < 0.03
    assert {
        "converged",
        "algorithm2_locally_robust_path",
        "finite_positive_standard_errors",
        "zeta_moment_norm",
        "covariance_min_eigenvalue",
        "parameter_cosine",
        "parameter_relative_rmse",
        "reward_normalized_rmse",
        "policy_tv",
        "value_normalized_rmse",
        "q_normalized_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    } <= gate_names
    assert payload["raw_neural_diagnostic"]["diagnostics"]["role"] == "diagnostic"
    assert payload["raw_neural_diagnostic"]["case_id"] != payload["primary_cell_id"]
    assert monte_carlo["case_id"] == payload["primary_cell_id"]
    assert monte_carlo["purpose"] == "repeated_seed_algorithm2_standard_error_coverage"
    assert monte_carlo["n_replications"] >= 25
    assert monte_carlo["successful_replications"] == monte_carlo["n_replications"]
    assert 0 < monte_carlo["gate_passing_replications"] <= monte_carlo["n_replications"]
    assert monte_carlo["covariance_units"] == ["individual"]
    assert monte_carlo["robust_optimizer_all_success_count"] == monte_carlo["n_replications"]
    assert monte_carlo["robust_optimizer_all_stationary_count"] == monte_carlo["n_replications"]
    assert (
        monte_carlo["preliminary_optimizer_all_stationary_count"] == monte_carlo["n_replications"]
    )
    assert 0.0 <= monte_carlo["coverage_95_overall"] <= 1.0
    assert monte_carlo["zeta_moment_norm_max"] < 1e-4
    assert math.isfinite(monte_carlo["lambda_fixed_point_residual_norm_max"])
    assert math.isfinite(monte_carlo["lambda_fixed_point_residual_rms_max"])
    assert math.isfinite(monte_carlo["lambda_fixed_point_residual_max_abs"])
    assert math.isfinite(monte_carlo["preliminary_projected_gradient_norm_max"])
    assert (
        monte_carlo["preliminary_projected_gradient_norm_max"]
        <= (paper_inference["preliminary_stationarity_tol"])
    )
    assert len(monte_carlo["bias"]) == len(summary["parameters"])
    assert len(monte_carlo["rmse"]) == len(summary["parameters"])
    assert len(monte_carlo["empirical_sd"]) == len(summary["parameters"])
    assert len(monte_carlo["mean_standard_error"]) == len(summary["parameters"])
    assert len(monte_carlo["coverage_95_by_parameter"]) == len(summary["parameters"])
    assert all(math.isfinite(x) for x in monte_carlo["bias"])
    assert all(math.isfinite(x) and x >= 0 for x in monte_carlo["rmse"])
    assert all(math.isfinite(x) and x >= 0 for x in monte_carlo["empirical_sd"])
    assert all(math.isfinite(x) and x > 0 for x in monte_carlo["mean_standard_error"])
    assert all(0.0 <= x <= 1.0 for x in monte_carlo["coverage_95_by_parameter"])


def test_sees_validation_artifact_matches_explicit_harness_penalties() -> None:
    """SEES validation must not silently fall back to wrapper/default penalties."""

    from validation.known_truth import build_known_truth_dgp, get_cell, make_estimator

    artifact = ROOT / "validation" / "results" / "sees.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    metadata_by_cell = {row["cell_id"]: row["summary"]["metadata"] for row in payload["results"]}
    expected = {
        "canonical_low_action": {
            "basis_dim": 21,
            "penalty_weight": 100.0,
        },
        "canonical_high_action": {
            "basis_dim": 81,
            "penalty_weight": 10_000.0,
        },
    }

    for cell_id, fields in expected.items():
        dgp = build_known_truth_dgp(get_cell(cell_id).dgp_config)
        estimator = make_estimator("SEES", dgp, smoke=False)
        metadata = metadata_by_cell[cell_id]

        assert estimator.config.basis_dim == fields["basis_dim"]
        assert estimator.config.penalty_weight == fields["penalty_weight"]
        assert metadata["configured_basis_dim"] == fields["basis_dim"]
        assert metadata["basis_dim"] == fields["basis_dim"]
        assert metadata["penalty_weight"] == fields["penalty_weight"]


def test_iq_learn_artifact_keeps_support_gates_visible() -> None:
    """IQ-Learn release evidence must gate sparse expert support explicitly."""

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    artifact = ROOT / "validation" / "results" / "iq_learn.json"
    payload = json.loads(
        artifact.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )

    assert payload["estimator"] == "IQ-Learn"
    assert payload["status"] == "strict_structural_counterfactual_fail"
    assert payload["counterfactual_valid_certified"] is False
    assert "Low policy distance" in payload["interpretation"]
    assert "treated as counterfactual-valid" in payload["interpretation"]
    assert "certified as counterfactual-valid" not in payload["interpretation"]

    saw_state_action_stress = False
    for result in payload["results"]:
        metadata = result["summary"]["metadata"]
        gates = {gate["name"]: gate for gate in result["gates"]}

        assert {
            "expert_state_coverage",
            "expert_state_action_coverage",
        } <= set(gates)
        assert gates["expert_state_coverage"]["operator"] == ">="
        assert gates["expert_state_coverage"]["threshold"] == 1.0
        assert gates["expert_state_action_coverage"]["operator"] == ">="
        assert gates["expert_state_action_coverage"]["threshold"] == 0.95
        assert gates["expert_state_coverage"]["value"] == metadata["expert_state_coverage"]
        assert (
            gates["expert_state_action_coverage"]["value"]
            == metadata["expert_state_action_coverage"]
        )
        assert 0.0 <= metadata["expert_state_coverage"] <= 1.0
        assert 0.0 <= metadata["expert_state_action_coverage"] <= 1.0

        if metadata["expert_state_action_coverage"] < 1.0:
            saw_state_action_stress = True
            assert result["counterfactual_valid_certified"] is False

    assert saw_state_action_stress


def test_iq_learn_sparse_support_guard_records_support_failure() -> None:
    """The local IQ-Learn support guard must fail sparse support explicitly."""

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    artifact = ROOT / "validation" / "results" / "iq_learn_sparse_support_guard.json"
    payload = json.loads(
        artifact.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    result = payload["result"]
    metadata = result["metadata"]
    gates = {gate["name"]: gate for gate in payload["gates"]}

    assert payload["artifact_name"] == "iq_learn_sparse_support_guard"
    assert payload["artifact_type"] == "local_sparse_support_guard"
    assert payload["estimator"] == "IQ-Learn"
    assert payload["generated_by"] == "validation/estimators/iq_learn/sparse_support_guard.py"
    assert payload["release_status"] == "local_support_guard_not_counterfactual_release_evidence"
    assert payload["counterfactual_valid_certified"] is False
    assert result["counterfactual_valid_certified"] is False
    assert result["treated_as_counterfactual_valid"] is False
    assert result["support_gate_passed"] is False
    assert result["support_gates_passed"] is False
    assert result["non_support_gates_passed"] is True
    assert "certified as counterfactual-valid" not in payload["purpose"]

    assert payload["problem_config"]["num_states"] == 3
    assert payload["problem_config"]["num_actions"] == 2
    assert payload["panel_config"]["observed_states"] == [0]
    assert payload["panel_config"]["observed_state_actions"] == [[0, 0]]
    assert payload["thresholds"]["expert_state_coverage_min"] == 1.0
    assert payload["thresholds"]["expert_state_action_coverage_min"] == 0.95

    assert math.isclose(metadata["expert_state_coverage"], 1 / 3)
    assert math.isclose(metadata["expert_state_action_coverage"], 1 / 6)
    assert 0 < metadata["expert_state_coverage"] < 1.0
    assert 0 < metadata["expert_state_action_coverage"] < 0.95

    state_gate = gates["expert_state_coverage"]
    state_action_gate = gates["expert_state_action_coverage"]
    assert state_gate["operator"] == ">="
    assert state_gate["threshold"] == 1.0
    assert state_gate["value"] == metadata["expert_state_coverage"]
    assert state_gate["passed"] is False
    assert state_action_gate["operator"] == ">="
    assert state_action_gate["threshold"] == 0.95
    assert state_action_gate["value"] == metadata["expert_state_action_coverage"]
    assert state_action_gate["passed"] is False

    assert set(result["failed_support_gates"]) == {
        "expert_state_coverage",
        "expert_state_action_coverage",
    }
    assert set(result["failed_gates"]) == set(result["failed_support_gates"])


def test_airl_artifact_keeps_action_dependent_cell_diagnostic() -> None:
    """Plain AIRL must not claim the action-dependent diagnostic as support."""

    artifact = ROOT / "validation" / "results" / "airl.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["estimator"] == "AIRL"
    assert payload["primary_cell_id"] == "airl_paper_identification"
    assert "failed diagnostic" in payload["interpretation"]

    by_cell = {row["cell_id"]: row for row in payload["results"]}
    paper_cell = by_cell["airl_paper_identification"]
    diagnostic = by_cell["airl_anchor_action_dependent"]

    assert paper_cell["diagnostics"]["is_action_dependent"] is False
    assert paper_cell["summary"]["metadata"]["reward_arg"] == "state"
    assert all(gate["passed"] for gate in paper_cell["gates"])

    assert diagnostic["diagnostics"]["is_action_dependent"] is True
    assert diagnostic["summary"]["metadata"]["reward_arg"] == "state_action"
    assert diagnostic["summary"]["converged"] is False
    assert all(not gate["passed"] for gate in diagnostic["gates"])


def test_airl_fu_section_71_transfer_artifact_locks_identified_route() -> None:
    """Fu Section 7.1 evidence must use fitted AIRL rows, not oracle rows."""

    artifact = ROOT / "validation" / "results" / "airl_fu2018_71.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["meta"]["n_states"] == 16
    assert payload["meta"]["n_actions"] == 4
    assert payload["meta"]["reward"] == "1.0 at state 0"
    assert payload["meta"]["start_state"] == 1
    assert payload["meta"]["transition_mode"] == "deterministic"
    assert payload["meta"]["paper_transition_target"] == "random transition matrices"
    assert payload["meta"]["release_status"] == "claim_reproduction_not_paper_exact"
    assert payload["meta"]["probe_status"] == "deterministic_identified_diagnostic"
    assert (
        payload["meta"]["paper_fidelity"]
        == "partial_claim_reproduction_deterministic_transition_variant"
    )
    assert payload["meta"]["n_individuals"] == 1500
    assert payload["meta"]["n_periods"] == 40
    assert payload["meta"]["estimator_config"]["negative_history"] == 0
    assert payload["meta"]["estimator_config"]["discriminator_data_mode"] == "sampled"
    assert payload["meta"]["estimator_config"]["policy_sample_mode"] == "chain"
    assert payload["meta"]["estimator_config"]["shaping_bellman_penalty"] == 0.0
    assert payload["meta"]["estimator_config"]["estimator_mode"] == "airl"
    assert payload["meta"]["estimator_config"]["state_initializer"] == "zeros"
    assert payload["meta"]["estimator_config"]["state_generator_reward"] == "f"
    assert payload["meta"]["estimator_config"]["action_generator_reward"] == "recovered"
    assert "randomly drawn transition matrices" in payload["meta"]["paper_target"]
    assert "reward maps and a transfer curve" in payload["meta"]["numeric_target"]

    assert "identified_metric_source" not in payload["meta"]
    assert 'AIRL(version="state_only")' not in payload["summary"]
    assert 'AIRLEstimator(reward_arg="state_action") diagnostic' not in payload["summary"]

    state_only = payload["summary"]["AIRL-1 (state-only)"]
    unanchored_state_action = payload["summary"]["AIRL-2 (state-action)"]

    assert 0.95 < state_only["reward_affine_r2"] < 1.0
    assert state_only["transfer_frac_optimal"] > 0.99
    assert "metric_source" not in payload["records"][0]["AIRL-1 (state-only)"]
    assert unanchored_state_action["reward_affine_r2"] < 0.5
    assert unanchored_state_action["transfer_frac_optimal"] < 0.0


def test_airl_fu_stochastic_probe_records_open_paper_gap() -> None:
    """The dense random-transition stress probe is not a successful replication."""

    artifact = ROOT / "validation" / "results" / "airl_fu2018_71_stochastic_probe.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    meta = payload["meta"]

    assert meta["transition_mode"] == "stochastic"
    assert meta["paper_transition_target"] == "random transition matrices"
    assert meta["release_status"] == "claim_reproduction_not_paper_exact"
    assert meta["paper_fidelity"] == ("paper_transition_style_exploratory_not_figure_calibrated")
    assert meta["probe_status"] == "open_gap_state_only_transfer_probe_failed"
    assert meta["n_individuals"] == 250
    assert meta["n_periods"] == 20
    assert meta["estimator_config"]["negative_history"] == 20
    assert meta["estimator_config"]["discriminator_data_mode"] == "sampled"
    assert meta["estimator_config"]["policy_sample_mode"] == "chain"
    assert meta["estimator_config"]["shaping_bellman_penalty"] == 0.0
    assert meta["estimator_config"]["estimator_mode"] == "airl"
    assert meta["estimator_config"]["state_initializer"] == "zeros"
    assert meta["estimator_config"]["state_generator_reward"] == "log_odds"
    assert meta["estimator_config"]["action_generator_reward"] == "log_odds"
    assert "reward maps and a transfer curve" in meta["numeric_target"]

    state_only = payload["summary"]["AIRL-1 (state-only)"]
    unanchored_state_action = payload["summary"]["AIRL-2 (state-action)"]

    assert state_only["reward_affine_r2"] < 0.10
    assert state_only["transfer_frac_optimal"] < 0.0
    assert unanchored_state_action["reward_affine_r2"] < 0.10
    assert abs(unanchored_state_action["transfer_frac_optimal"]) < 0.10


def test_airl_fu_sparse_original_probe_recovers_state_only_reward() -> None:
    """The public-code-style sparse random MDP recovers the state-only reward."""

    artifact = ROOT / "validation" / "results" / "airl_fu2018_71_sparse_original.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    meta = payload["meta"]

    assert meta["transition_mode"] == "sparse_original"
    assert meta["paper_transition_target"] == "random transition matrices"
    assert meta["release_status"] == "claim_reproduction_not_paper_exact"
    assert meta["paper_fidelity"] == (
        "original_code_style_sparse_transition_probe_not_figure_calibrated"
    )
    assert meta["probe_status"] == "state_only_sparse_original_airl_passed"
    assert "exp(rand) weights" in meta["transition_mode_note"]
    source = meta["source_code_reference"]
    assert source["repository"] == "justinjfu/inverse_rl"
    assert source["path"] == "tabular_maxent_irl/simple_env.py"
    assert source["function"] == "random_env"
    assert source["default_t_sparsity"] == 0.75
    assert any("t_sparsity" in step for step in source["recipe"])
    assert meta["n_individuals"] == 1500
    assert meta["n_periods"] == 40
    assert meta["estimator_config"]["estimator_mode"] == "airl"
    assert meta["estimator_config"]["state_initializer"] == "zeros"
    assert meta["estimator_config"]["discriminator_data_mode"] == "sampled"
    assert meta["estimator_config"]["policy_sample_mode"] == "rollout"
    assert meta["estimator_config"]["negative_history"] == 20
    assert meta["estimator_config"]["state_generator_reward"] == "recovered"

    state_only = payload["summary"]["AIRL-1 (state-only)"]
    state_action = payload["summary"]["AIRL-2 (state-action)"]

    assert state_only["reward_affine_r2"] > 0.90
    assert state_only["transfer_frac_optimal"] > 0.99
    assert state_action["reward_affine_r2"] < 0.10


def test_airl_fu_stochastic_exact_occupancy_airl_records_partial_gap() -> None:
    """Exact tabular occupancy removes sampler noise but still misses paper recovery."""

    artifact = ROOT / "validation" / "results" / "airl_fu2018_71_stochastic_airl_occupancy.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    meta = payload["meta"]

    assert meta["transition_mode"] == "stochastic"
    assert meta["paper_transition_target"] == "random transition matrices"
    assert meta["release_status"] == "claim_reproduction_not_paper_exact"
    assert meta["probe_status"] == "open_gap_exact_occupancy_airl_failed"
    assert meta["estimator_config"]["estimator_mode"] == "airl"
    assert meta["estimator_config"]["state_initializer"] == "zeros"
    assert meta["estimator_config"]["discriminator_data_mode"] == "occupancy"
    assert meta["estimator_config"]["shaping_bellman_penalty"] == 0.0
    assert meta["estimator_config"]["negative_history"] == 0
    assert meta["n_individuals"] == 5000
    assert meta["n_periods"] == 80

    state_only = payload["summary"]["AIRL-1 (state-only)"]
    sampled = json.loads(
        (ROOT / "validation" / "results" / "airl_fu2018_71_stochastic_probe.json").read_text(
            encoding="utf-8"
        )
    )["summary"]["AIRL-1 (state-only)"]

    assert 0.55 < state_only["reward_affine_r2"] < 0.90
    assert 0.80 < state_only["transfer_frac_optimal"] < 0.90
    assert state_only["reward_affine_r2"] > sampled["reward_affine_r2"] + 0.50
    assert state_only["transfer_frac_optimal"] > sampled["transfer_frac_optimal"] + 1.0


def test_airl_fu_stochastic_inverse_bellman_ceiling_passes() -> None:
    """The stochastic MDP is identifiable with enough data; AIRL training is the gap."""

    artifact = ROOT / "validation" / "results" / "airl_fu2018_71_stochastic_ccp_ceiling.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    meta = payload["meta"]

    assert meta["transition_mode"] == "stochastic"
    assert meta["paper_transition_target"] == "random transition matrices"
    assert meta["release_status"] == "claim_reproduction_not_paper_exact"
    assert meta["probe_status"] == "state_only_inverse_bellman_ceiling_passed"
    assert meta["estimator_config"]["estimator_mode"] == "ccp_inverse"
    assert meta["estimator_config"]["state_initializer"] == "zeros"
    assert meta["estimator_config"]["discriminator_data_mode"] == "sampled"
    assert meta["n_individuals"] == 5000
    assert meta["n_periods"] == 80

    ceiling = payload["summary"]["finite-sample state-only inverse Bellman ceiling"]
    assert ceiling["reward_affine_r2"] > 0.95
    assert ceiling["transfer_frac_optimal"] > 0.98

    record = payload["records"][0]["finite-sample state-only inverse Bellman ceiling"]
    assert record["linear_system_rank"] == 16
    assert record["linear_system_condition_number"] < 25
    assert record["min_state_count"] > 10_000
    assert record["state0_count"] > 30_000


def test_airl_fu_stochastic_ccp_initialized_airl_continuation_passes() -> None:
    """Warm-started AIRL is useful evidence, but not a from-zero paper replication."""

    artifact = ROOT / "validation" / "results" / "airl_fu2018_71_stochastic_airl_ccp_init.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    meta = payload["meta"]

    assert meta["transition_mode"] == "stochastic"
    assert meta["paper_transition_target"] == "random transition matrices"
    assert meta["release_status"] == "claim_reproduction_not_paper_exact"
    assert meta["probe_status"] == "state_only_ccp_initialized_airl_continuation_passed"
    assert "not the paper's from-zero AIRL run" in meta["initialization_note"]
    assert meta["estimator_config"]["estimator_mode"] == "airl"
    assert meta["estimator_config"]["state_initializer"] == "ccp_inverse"
    assert meta["estimator_config"]["discriminator_data_mode"] == "sampled"
    assert meta["estimator_config"]["policy_sample_mode"] == "rollout"
    assert meta["estimator_config"]["shaping_bellman_penalty"] == 0.0
    assert meta["estimator_config"]["negative_history"] == 20
    assert meta["estimator_config"]["max_rounds"] == 5
    assert meta["n_individuals"] == 5000
    assert meta["n_periods"] == 80

    state_only = payload["summary"]["AIRL-1 (state-only)"]
    assert state_only["reward_affine_r2"] > 0.95
    assert state_only["transfer_frac_optimal"] > 0.98

    record = payload["records"][0]["AIRL-1 (state-only)"]
    assert record["num_iterations"] == 5
    assert record["initial_reward_affine_r2"] > 0.95
    assert record["initial_transfer_frac_optimal"] > 0.98
    assert record["initial_linear_system_rank"] == 16
    assert record["initial_min_state_count"] > 10_000


def test_airl_fu_reference_match_reproduces_section_71_recovery_and_transfer() -> None:
    """Section 7.1 (Figures 1 and 2) matched to the authors' reference code.

    Section 7.1 uses MaxEnt IRL (the precursor), implemented in the package by
    MCEIRLEstimator. The reference oracle is a port of justinjfu/inverse_rl. Figure 1
    (recovery): the state-only reward recovers the ground truth, the state-action reward
    is a shaped advantage. Figure 2 (transfer, averaged over fresh dynamics): the
    state-only reward transfers to optimal, the state-action reward stays marginal.
    """

    artifact = ROOT / "validation" / "results" / "airl_fu2018_71_reference_match.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    meta = payload["meta"]
    summary = payload["summary"]

    assert meta["probe_status"] == (
        "section_71_recovery_and_state_only_transfer_reference_match_passed"
    )
    assert meta["bar"] == "reference_implementation_equivalence"
    assert meta["n_states"] == 16
    assert meta["n_actions"] == 4
    assert meta["n_transfer_envs"] >= 20
    source = meta["source_code_reference"]
    assert source["repository"] == "justinjfu/inverse_rl"
    assert source["reference_config"]["seed"] == 1
    assert source["reference_config"]["t_sparsity"] == 0.8
    assert source["transfer_experiment_in_public_repo"] is False

    # Figure 1: recovery.
    ref_so = summary["reference"]["state_only"]
    assert ref_so["reward_affine_r2"] >= 0.999
    assert ref_so["reward_inf_norm_error"] < 1e-6

    ref_sa = summary["reference"]["state_action"]
    # Shaped advantage: uncorrelated with the truth, yet reproduces the expert.
    assert ref_sa["reward_affine_r2_train"] <= 0.05
    assert ref_sa["shaping_policy_tv_train"] <= 0.05

    pkg_so = summary["package"]["state_only"]
    assert pkg_so["reward_affine_r2"] >= 0.95

    # Figure 2: gate ONLY the method-intrinsic fact. The recovered state-only reward is
    # the true reward, so it is optimal under any dynamics: transfer 1.0 with ~0 variance.
    assert ref_so["transfer_frac_optimal"] >= 0.95
    assert ref_so["transfer_frac_std"] <= 0.05
    assert pkg_so["transfer_frac_optimal"] >= 0.95
    # The state-action transfer is DESCRIPTIVE only (NOT gated): it is high variance
    # (std ~0.4) and representative-dependent, so we lock only that its spread is recorded,
    # not the fragile mean or the contrast (which is just 1 minus that same mean).
    assert "transfer_frac_optimal" in ref_sa
    assert ref_sa["transfer_frac_std"] >= 0.1
    assert "transfer_contrast_state_only_minus_state_action" in summary

    # Package soft-Bellman matches the reference soft value iteration (primitive
    # equivalence), and the two recovered state-only rewards agree up to affine.
    assert summary["soft_policy_max_gap"] < 1e-5
    assert summary["package_vs_reference_state_only_r2"] >= 0.95


def test_airl_fu_internal_replication_note_does_not_overclaim_exact_match() -> None:
    """The private Fu memo must agree with the scoped JSON artifact status."""

    note_path = ROOT / "docs" / "research" / "replications" / "airl_fu2018.md"
    if not note_path.exists():
        return

    note = note_path.read_text(encoding="utf-8")

    assert "fitted claim reproduction, not an exact paper-number replication" in note
    assert "claim_reproduction_not_paper_exact" in note
    assert "AIRL-1 state-only fitted reward | 0.977 | 1.000" in note
    assert "Do not mark Fu Section 7.1 as exact paper replication" in note

    assert "Recovery (exact Section 7.1 target)" not in note
    assert "reward affine R2 1.000 and transfer fraction" not in note
    assert '`AIRL(version="state_only")` | 1.000 | 1.000' not in note
    assert "exact identified\n  known-truth target" not in note


def test_airl_anchored_heterogeneity_artifact_is_fitted_mechanism_validation() -> None:
    """AIRL-Het must not report oracle-perfect empirical replication."""

    artifact = ROOT / "validation" / "results" / "aairl.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["status"] == "pass"
    assert payload["primary_cell_id"] == "airl_het_paper_identification"
    assert payload["paper_target"]["replication_status"] == (
        "blocked_without_proprietary_platform_data"
    )
    assert payload["paper_target"]["num_segments"] == 4
    assert "over one million users" in payload["paper_target"]["data_requirement"]
    assert any(
        "Table 5 marginal effects" in target
        for target in payload["paper_target"]["empirical_targets"]
    )
    assert "Table 6 aggregate" in payload["paper_target"]["empirical_targets"][-1]
    assert payload["paper_target"]["training_schedule"]["epochs"] == 1000
    assert payload["paper_target"]["training_schedule"]["trajectories_per_epoch"] == 16_384
    assert payload["paper_target"]["training_schedule"]["posterior_fixed_after_epoch"] == 500
    assert payload["local_validation_scope"]["scope"] == ("synthetic_mechanism_validation_only")
    assert payload["local_validation_scope"]["not_empirical_replication"] is True
    assert payload["local_validation_scope"]["num_segments"] == 2

    result = payload["results"][0]
    assert result["diagnostics"]["anchor_valid"] is True
    assert result["dgp_config"]["exit_action"] == 2
    assert result["dgp_config"]["absorbing_state"] == 60
    assert "replication_mode" not in result["summary"]["metadata"]
    assert result["summary"]["metadata"]["num_segments"] == 2
    assert result["summary"]["metadata"]["generator_reward"] == "f"
    assert result["summary"]["metadata"]["initialization"] == "behavioral_anchor"
    assert result["summary"]["num_iterations"] > 0
    assert result["summary"]["log_likelihood"] is not None
    assert all(gate["passed"] for gate in result["gates"])

    oracle_reference = result["oracle_assignment_reference"]
    assert oracle_reference["num_trajectories"] == 3200
    assert oracle_reference["num_users"] == 800
    assert math.isclose(
        oracle_reference["trajectory_map_accuracy"],
        0.7159375,
    )
    assert math.isclose(
        oracle_reference["user_pooled_map_accuracy"],
        0.90125,
    )

    metrics = result["metrics"]
    assert math.isclose(metrics["segment_assignment_accuracy"], 0.895)
    assert metrics["segment_assignment_accuracy"] < oracle_reference["user_pooled_map_accuracy"]
    assert math.isclose(metrics["segment_prior_l1"], 0.043477813304278856)
    assert math.isclose(metrics["max_segment_policy_tv"], 0.05913144436746753)
    assert math.isclose(
        metrics["max_segment_reward_normalized_rmse"],
        0.2650011487814588,
    )
    assert math.isclose(
        metrics["max_segment_value_normalized_rmse"],
        0.14197337719868644,
    )
    assert math.isclose(
        metrics["max_segment_q_normalized_rmse"],
        0.21138937605956745,
    )
    assert math.isclose(
        metrics["max_segment_counterfactual_regret"]["type_a"],
        0.014492342845929596,
    )
    assert math.isclose(
        metrics["max_segment_counterfactual_regret"]["type_b"],
        0.11893310781701327,
    )
    assert math.isclose(
        metrics["max_segment_counterfactual_regret"]["type_c"],
        0.00686809129514554,
    )


def test_f_irl_artifact_keeps_action_dependent_row_diagnostic() -> None:
    """f-IRL's action-dependent DDC row is a negative-control diagnostic."""

    artifact = ROOT / "validation" / "results" / "f_irl.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["estimator"] == "f-IRL"
    assert payload["primary_cell_id"] == "f_irl_paper_state_marginal"
    assert "diagnostic negative control" in payload["interpretation"]

    by_cell = {row["cell_id"]: row for row in payload["results"]}
    paper_cell = by_cell["f_irl_paper_state_marginal"]
    diagnostic = by_cell["canonical_low_action"]

    assert paper_cell["summary"]["metadata"]["marginal_space"] == "state"
    assert paper_cell["summary"]["metadata"]["reward_scope"] == "state"
    assert all(gate["passed"] for gate in paper_cell["gates"])

    assert diagnostic["diagnostics"]["is_action_dependent"] is True
    assert diagnostic["summary"]["metadata"]["marginal_space"] == "state_action"
    assert diagnostic["summary"]["metadata"]["reward_scope"] == "state_action"
    assert {gate["name"] for gate in diagnostic["gates"]} == {
        "converged",
        "occupancy_l1",
        "reward_range",
    }
    assert "reward_normalized_rmse" not in {gate["name"] for gate in diagnostic["gates"]}
    assert not all(gate["passed"] for gate in diagnostic["gates"])


def test_gladius_artifacts_keep_failed_structural_gates_visible() -> None:
    """GLADIUS projected diagnostics must not hide raw/value gate failures."""

    artifacts = [
        ROOT / "validation" / "results" / "gladius.json",
        ROOT / "validation" / "results" / "gladius_scaled.json",
    ]

    for artifact in artifacts:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        failed_gates = set(payload["failed_gates"])
        gates = {gate["name"]: gate for gate in payload["gates"]}

        assert payload["estimator"] == "GLADIUS"
        assert payload["status"] == "strict_structural_counterfactual_fail"
        assert payload["counterfactual_valid_certified"] is False
        assert {
            "raw_bellman_reward_normalized_rmse",
            "value_normalized_rmse",
        } <= failed_gates
        assert gates["projected_reward_normalized_rmse"]["passed"] is True
        assert gates["raw_bellman_reward_normalized_rmse"]["passed"] is False
        assert gates["value_normalized_rmse"]["passed"] is False
        assert all(gates[f"{kind}_regret"]["passed"] for kind in ["type_a", "type_b", "type_c"])


def test_structural_estimator_docs_topology_matches_nfxp_ccp() -> None:
    """Structural estimator pages should keep the shared reader path.

    The locked academic template (NFXP reference, 2026-06-23) folds the old
    ``context`` and ``under_the_hood`` subpages into the main page, leaving five
    subpages in a fixed order. The canonical reader order is the hidden toctree at
    the foot of the page, so order is checked there rather than by first mention in
    the body (estimator pages cross-link their own subpages inline, which is not the
    reader path).
    """

    expected = [
        "quick_start",
        "pre_estimation",
        "validation",
        "counterfactuals",
        "rust_bus",
    ]

    for slug in ["nfxp", "ccp", "tdccp"]:
        docs_dir = ROOT / "docs" / "estimators" / slug
        actual_files = sorted(path.stem for path in docs_dir.glob("*.md"))
        missing = sorted(set(expected) - set(actual_files))
        assert not missing, (slug, missing)

        parent = (ROOT / "docs" / "estimators" / f"{slug}.md").read_text(encoding="utf-8")
        toctree_start = parent.find("{toctree}")
        assert toctree_start != -1, (slug, "no toctree")
        toctree = parent[toctree_start:]
        cursor = -1
        for item in expected:
            needle = f"{slug}/{item}"
            position = toctree.find(needle)
            assert position > cursor, (slug, needle)
            cursor = position
