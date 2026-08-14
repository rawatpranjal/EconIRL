"""Pure aggregation tests for the NFXP readiness manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.estimators.nfxp.ready import (
    _out_of_sample_scores,
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
        "out_of_sample": {
            "negative_log_likelihood": 0.4,
            "brier_score": 0.2,
            "excess_negative_log_likelihood": 0.01,
            "excess_brier_score": 0.005,
        },
        "counterfactuals": {
            "reward": {"policy_tv": 0.01, "regret": 0.01},
            "transition": {"policy_tv": 0.01, "regret": 0.01},
        },
        "error": None,
    }


def test_inference_summary_keeps_left_and_right_tail_errors_separate() -> None:
    truth = np.array([0.0])
    estimates = [-3.0, -2.0, 0.0, 2.0, 3.0]
    records = [_record(index, [estimate], [0.5]) for index, estimate in enumerate(estimates)]

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


def test_out_of_sample_scores_compare_same_heldout_choices() -> None:
    class PanelStub:
        def get_all_states(self) -> np.ndarray:
            return np.array([0, 1])

        def get_all_actions(self) -> np.ndarray:
            return np.array([0, 1])

    oracle = np.array([[0.8, 0.2], [0.3, 0.7]])
    estimated = np.array([[0.7, 0.3], [0.4, 0.6]])

    scores = _out_of_sample_scores(PanelStub(), estimated, oracle)

    assert scores["n_observations"] == 2
    assert scores["negative_log_likelihood"] > scores["oracle_negative_log_likelihood"]
    assert scores["brier_score"] > scores["oracle_brier_score"]


def test_smoke_gates_are_computed_but_not_enforced() -> None:
    records = [_record(index, [1.0, -0.8, 0.6], [0.1, 0.1, 0.1]) for index in range(2)]
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
    assert payload["paper_target"] == {
        "receipt": "docs/replications.md",
        "relationship": "separate exact paper replication",
        "runner": "src/econirl/replication/rust1987/table_ix.py",
        "source": "Rust (1987), Table IX, Group 4",
    }
    assert payload["provenance"]["git_commit"] == ("d7fa16fb101b39ac7d5977ec2203d00ab30e4c2c")
    assert payload["inference"]["n_total"] == 1000
    assert payload["inference"]["n_usable"] == 1000
    assert payload["hard_problem"]["n_total"] == 20
    assert payload["hard_problem"]["n_usable"] == 20
    assert payload["hard_problem"]["out_of_sample"]["excess_negative_log_likelihood_mean"] <= 0.02
    assert payload["records_parquet"].endswith("ready_records.parquet")
    assert all(gate["passed"] for gate in payload["gates"] if gate["enforced"])
    expected_sections = (
        "Estimator\n",
        "\nData\n",
        "\nModel\n",
        "\nPre-estimation checks\n",
        "\nFit\n",
        "\nOutcome\n",
        "\nUncertainty\n",
        "\nLimitations\n",
    )
    assert all(section in payload["summary_report"] for section in expected_sections)
    assert "Converged: yes" in payload["summary_report"]
