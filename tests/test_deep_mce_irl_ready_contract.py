"""Contracts for Neural MCE-IRL completion evidence."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("validation/results")


def read_result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_controlled_recovery_cells_pass() -> None:
    payload = read_result("deep_mce_irl.json")
    assert payload["primary_cell"] == "deep_mce_neural_reward"
    assert len(payload["records"]) == 4
    assert all(gate["passed"] for record in payload["records"] for gate in record["gates"])


def test_repeated_stability_receipt_is_ready() -> None:
    payload = read_result("deep_mce_irl_ready.json")
    assert payload["status"] == "ready"
    assert payload["inference"].startswith("not supported")
    assert payload["configuration"]["n_fits"] >= 300
    summary = payload["summary"]
    assert summary["n_successful"] == summary["n_requested"]
    assert summary["n_converged"] == summary["n_requested"]
    assert summary["reward_nrmse"]["median"] <= 0.15
    assert summary["reward_nrmse"]["p95"] <= 0.30
    assert summary["policy_tv"]["median"] <= 0.05
    assert summary["policy_tv"]["p95"] <= 0.10
    assert summary["training_seed_policy_tv"]["median"] <= 0.02
    assert summary["training_seed_policy_tv"]["p95"] <= 0.05
    assert all(check["passed"] for check in payload["checks"])


def test_counterfactual_stability_has_six_passing_checks() -> None:
    payload = read_result("deep_mce_irl_ready.json")
    counterfactual_checks = [
        check
        for check in payload["checks"]
        if check["name"].endswith(("_median_regret", "_p95_regret"))
    ]
    assert len(counterfactual_checks) == 6
    assert all(check["passed"] for check in counterfactual_checks)
    design_checks = [
        check for check in payload["checks"] if check["name"].endswith("_oracle_policy_changes")
    ]
    assert len(design_checks) == 3
    assert all(check["passed"] for check in design_checks)


def test_wulfmeier_shaped_study_is_ready() -> None:
    payload = read_result("deep_mce_irl_wulfmeier.json")
    assert payload["status"] == "ready"
    assert payload["paper_replication"] is False
    assert payload["configuration"]["grid_size"] == 32
    assert payload["configuration"]["demo_counts"] == [8, 16, 32, 64, 128]
    assert payload["configuration"]["panel_seeds"] == 5
    assert payload["configuration"]["training_seeds"] == 3
    assert payload["summary"]["n_successful"] == payload["summary"]["n_requested"]
    assert 0 <= payload["summary"]["n_converged"] <= payload["summary"]["n_requested"]
    assert (
        sum(payload["summary"]["termination_counts"].values()) == payload["summary"]["n_requested"]
    )
    for cell_name in ("objectworld:128", "binaryworld:64", "binaryworld:128"):
        cell = payload["summary"]["cells"][cell_name]
        assert cell["linear_joint_solution_pass"] == cell["linear_fits"] == 5
    assert all(check["passed"] for check in payload["checks"])
