"""Pure aggregation tests for the NFXP readiness manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.estimators.nfxp.ready import (
    hard_problem_summary,
    inference_summary,
    readiness_gates,
)


def _record(rep: int, estimate: list[float], standard_error: list[float]) -> dict:
    return {
        "problem": "test",
        "rep": rep,
        "parameters": estimate,
        "standard_errors": standard_error,
        "converged": True,
        "runtime_seconds": 1.0,
        "policy_tv": 0.01,
        "counterfactuals": {
            "reward": {"policy_tv": 0.01, "regret": 0.01},
            "transition": {"policy_tv": 0.01, "regret": 0.01},
        },
        "error": None,
    }


def test_inference_summary_keeps_left_and_right_tail_errors_separate() -> None:
    truth = np.array([0.0])
    estimates = [-3.0, -2.0, 0.0, 2.0, 3.0]
    records = [
        _record(index, [estimate], [0.5])
        for index, estimate in enumerate(estimates)
    ]

    result = inference_summary(records, truth, ["theta"])

    assert result["parameters"]["theta"]["left_miss"] == 0.4
    assert result["parameters"]["theta"]["right_miss"] == 0.4
    assert result["parameters"]["theta"]["coverage_95"] == pytest.approx(0.2)


def test_failed_fit_reduces_usable_rate_instead_of_disappearing() -> None:
    records = [_record(0, [1.0], [0.1]), _record(1, [1.0], [0.1])]
    records[1] = {
        "problem": "test",
        "rep": 1,
        "converged": False,
        "error": "RuntimeError: failed",
    }

    result = inference_summary(records, np.array([1.0]), ["theta"])

    assert result["n_total"] == 2
    assert result["n_usable"] == 1
    assert result["usable_rate"] == 0.5


def test_smoke_gates_are_computed_but_not_enforced() -> None:
    records = [
        _record(index, [1.0, -0.8, 0.6], [0.1, 0.1, 0.1])
        for index in range(2)
    ]
    hard = hard_problem_summary(
        records,
        np.array([1.0, -0.8, 0.6]),
        ["theta_0", "theta_1", "theta_2"],
    )
    inference = {
        "usable_rate": 0.0,
        "parameters": {},
    }

    gates = readiness_gates(inference, hard, final_run=False)

    assert gates
    assert all(gate["enforced"] is False for gate in gates)


def test_release_manifest_records_a_passing_full_run() -> None:
    path = Path("validation/results/nfxp_ready.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "ready"
    assert payload["paper_target"] is None
    assert payload["inference"]["n_total"] == 1000
    assert payload["inference"]["n_usable"] == 1000
    assert payload["hard_problem"]["n_total"] == 20
    assert payload["hard_problem"]["n_usable"] == 20
    assert all(gate["passed"] for gate in payload["gates"] if gate["enforced"])
    assert "[3] TRANSITION MODEL" in payload["summary_report"]
    assert "Converged:   yes" in payload["summary_report"]
