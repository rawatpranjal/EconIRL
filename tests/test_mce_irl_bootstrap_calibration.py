"""Frozen-design and committed-result checks for MCE-IRL bootstrap calibration."""

from __future__ import annotations

import json
from pathlib import Path

from validation.estimators.mce_irl.bootstrap_calibration import (
    CALIBRATION_REPS,
    N_BOOTSTRAP,
    gates,
    summarize,
)


def test_bootstrap_design_is_frozen() -> None:
    assert CALIBRATION_REPS == 50
    assert N_BOOTSTRAP == 99


def test_bootstrap_aggregation_reports_coverage_tails_and_width() -> None:
    records = [
        {
            "error": None,
            "success_fraction": 1.0,
            "n_successful": 9,
            "n_requested": 9,
            "failure_count": 0,
            "covered": True,
            "lower_miss": False,
            "upper_miss": False,
            "interval_width": 0.4,
        },
        {
            "error": None,
            "success_fraction": 1.0,
            "n_successful": 9,
            "n_requested": 9,
            "failure_count": 0,
            "covered": False,
            "lower_miss": True,
            "upper_miss": False,
            "interval_width": 0.6,
        },
    ]
    summary = summarize(records)
    assert summary["coverage_95"] == 0.5
    assert summary["lower_tail_miss_rate"] == 0.5
    assert summary["upper_tail_miss_rate"] == 0.0
    assert summary["mean_interval_width"] == 0.5
    assert all("enforced" in gate for gate in gates(summary, final_run=True))


def test_release_bootstrap_receipt_is_ready() -> None:
    path = Path("validation/results/mce_irl_bootstrap_calibration.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["design"]["n_calibration_panels"] >= CALIBRATION_REPS
    assert payload["design"]["n_bootstrap"] >= N_BOOTSTRAP
    assert payload["summary"]["n_usable"] >= 0.95 * CALIBRATION_REPS
    assert payload["program_check"]["passed"]
    assert all(gate["passed"] for gate in payload["gates"] if gate["enforced"])
