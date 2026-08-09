"""Pure aggregation tests for NFXP bootstrap calibration."""

from __future__ import annotations

import pytest

from validation.estimators.nfxp.bootstrap_calibration import (
    calibration_gates,
    summarize_bootstrap,
)


def _record(rep: int, covered: list[bool], widths: list[float]) -> dict:
    return {
        "rep": rep,
        "success_fraction": 0.98,
        "covered": covered,
        "interval_widths": widths,
        "failure_count": 2,
        "error": None,
    }


def test_summary_retains_coverage_widths_and_failures() -> None:
    records = [
        _record(0, [True, False], [0.4, 0.6]),
        _record(1, [True, True], [0.2, 0.8]),
    ]

    summary = summarize_bootstrap(records, ["theta_0", "theta_1"])

    assert summary["usable_rate"] == 1.0
    assert summary["minimum_success_fraction"] == 0.98
    assert summary["total_failed_draws"] == 4
    assert summary["parameters"]["theta_0"]["coverage_95"] == 1.0
    assert summary["parameters"]["theta_1"]["coverage_95"] == 0.5
    assert summary["parameters"]["theta_0"]["mean_interval_width"] == pytest.approx(0.3)


def test_failed_panel_reduces_usable_rate() -> None:
    records = [_record(0, [True], [0.4]), {"rep": 1, "error": "failed"}]

    summary = summarize_bootstrap(records, ["theta"])

    assert summary["n_total"] == 2
    assert summary["n_usable"] == 1
    assert summary["usable_rate"] == 0.5


def test_smoke_gates_are_reported_but_not_enforced() -> None:
    summary = summarize_bootstrap([_record(0, [True], [0.4])], ["theta"])

    gates = calibration_gates(summary, final_run=False)

    assert gates
    assert all(gate["enforced"] is False for gate in gates)
