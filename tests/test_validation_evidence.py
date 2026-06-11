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
    artifact = (
        ROOT
        / "validation"
        / "results"
        / "rust_cross_estimator_benchmark_archived.json"
    )

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

    artifact = (
        ROOT
        / "validation"
        / "results"
        / "mpec_high_beta_smoke.json"
    )
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
    assert result["final_constraint_violation"] < thresholds[
        "final_constraint_violation_max"
    ]
    assert result["num_iterations"] <= thresholds["num_iterations_max"]
    assert 0 < result["estimation_time"] < thresholds["estimation_time_max_seconds"]
    assert 0 < result["wall_time_seconds"] < thresholds["wall_time_max_seconds"]
    assert result["num_observations"] == (
        payload["simulation_config"]["n_individuals"]
        * payload["simulation_config"]["n_periods"]
    )
    assert result["num_observations"] > 0
    assert result["parameters"]
    assert all(math.isfinite(value) for value in result["parameters"])


def test_tdccp_primer_json_artifact_has_release_gates() -> None:
    """TD-CCP primer JSON should expose the paper-faithful release evidence."""

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    artifact = (
        ROOT
        / "validation"
        / "results"
        / "tdccp.json"
    )

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
    assert paper_inference["preliminary_projected_gradient_norm_max"] <= (
        paper_inference["preliminary_stationarity_tol"]
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
    assert monte_carlo["preliminary_optimizer_all_stationary_count"] == monte_carlo["n_replications"]
    assert 0.0 <= monte_carlo["coverage_95_overall"] <= 1.0
    assert monte_carlo["zeta_moment_norm_max"] < 1e-4
    assert math.isfinite(monte_carlo["lambda_fixed_point_residual_norm_max"])
    assert math.isfinite(monte_carlo["lambda_fixed_point_residual_rms_max"])
    assert math.isfinite(monte_carlo["lambda_fixed_point_residual_max_abs"])
    assert math.isfinite(monte_carlo["preliminary_projected_gradient_norm_max"])
    assert monte_carlo["preliminary_projected_gradient_norm_max"] <= (
        paper_inference["preliminary_stationarity_tol"]
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

    artifact = (
        ROOT
        / "validation"
        / "results"
        / "sees.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    metadata_by_cell = {
        row["cell_id"]: row["summary"]["metadata"] for row in payload["results"]
    }
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

    artifact = (
        ROOT
        / "validation"
        / "results"
        / "iq_learn.json"
    )
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
        assert gates["expert_state_coverage"]["value"] == metadata[
            "expert_state_coverage"
        ]
        assert gates["expert_state_action_coverage"]["value"] == metadata[
            "expert_state_action_coverage"
        ]
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

    artifact = (
        ROOT
        / "validation"
        / "results"
        / "iq_learn_sparse_support_guard.json"
    )
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
    assert (
        payload["generated_by"]
        == "validation/estimators/iq_learn/sparse_support_guard.py"
    )
    assert (
        payload["release_status"]
        == "local_support_guard_not_counterfactual_release_evidence"
    )
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

    artifact = (
        ROOT
        / "validation"
        / "results"
        / "airl.json"
    )
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


def test_f_irl_artifact_keeps_action_dependent_row_diagnostic() -> None:
    """f-IRL's action-dependent DDC row is a negative-control diagnostic."""

    artifact = (
        ROOT
        / "validation"
        / "results"
        / "f_irl.json"
    )
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
    assert "reward_normalized_rmse" not in {
        gate["name"] for gate in diagnostic["gates"]
    }
    assert not all(gate["passed"] for gate in diagnostic["gates"])


def test_gladius_artifacts_keep_failed_structural_gates_visible() -> None:
    """GLADIUS projected diagnostics must not hide raw/value gate failures."""

    artifacts = [
        ROOT
        / "validation"
        / "results"
        / "gladius.json",
        ROOT
        / "validation"
        / "results"
        / "gladius_scaled.json",
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
        assert all(
            gates[f"{kind}_regret"]["passed"]
            for kind in ["type_a", "type_b", "type_c"]
        )


def test_structural_estimator_docs_topology_matches_nfxp_ccp() -> None:
    """Structural estimator pages should keep the shared reader path."""

    expected = [
        "context",
        "quick_start",
        "under_the_hood",
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

        parent = (ROOT / "docs" / "estimators" / f"{slug}.md").read_text(
            encoding="utf-8"
        )
        cursor = -1
        for item in expected:
            needle = f"{slug}/{item}"
            position = parent.find(needle)
            assert position > cursor, needle
            cursor = position
