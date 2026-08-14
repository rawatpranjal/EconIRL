"""Frozen-design and committed-result checks for TD-CCP qualification."""

from __future__ import annotations

import json
from pathlib import Path

from validation.estimators.tdccp import (
    bootstrap_calibration,
    inference_calibration,
    neural_avi,
    ready,
)


def _payload(name: str) -> dict:
    return json.loads(Path(f"validation/results/{name}.json").read_text(encoding="utf-8"))


def test_qualification_designs_are_frozen() -> None:
    assert ready.FINAL_REPLICATIONS == 20
    assert ready.N_INDIVIDUALS == 2000
    assert ready.N_PERIODS == 60
    assert inference_calibration.FINAL_REPLICATIONS == 1000
    assert inference_calibration.N_INDIVIDUALS == 3000
    assert inference_calibration.N_PERIODS == 30
    assert bootstrap_calibration.CALIBRATION_REPS == 50
    assert bootstrap_calibration.N_BOOTSTRAP == 99
    assert neural_avi.FINAL_REPLICATIONS == 30


def test_exact_table_e1_result_matches_every_published_quantity() -> None:
    payload = _payload("tdccp_table_e1")
    assert payload["completed_replications"] == 1000
    assert payload["exact_replication_passed"] is True
    for mode in ("nonrobust", "robust"):
        assert payload["modes"][mode]["n_replications"] == 1000
        assert payload["modes"][mode]["matches_four_significant_figures"] is True


def test_ready_neural_and_highdim_results_pass_all_gates() -> None:
    readiness = _payload("tdccp_ready")
    assert readiness["summary"]["status"] == "ready"
    assert readiness["summary"]["completed_replications"] == 20
    assert all(gate["passed"] for gate in readiness["summary"]["gates"])

    neural = _payload("tdccp_neural_avi")
    assert neural["status"] == "ready"
    assert neural["summary"]["completed_replications"] == 30
    assert all(gate["passed"] for gate in neural["summary"]["gates"])

    highdim = _payload("tdccp_highdim")
    assert highdim["status"] == "ready"
    assert highdim["seeds"] == 30
    assert all(gate["passed"] for gate in highdim["gates"])


def test_inference_result_passes_every_frozen_gate() -> None:
    payload = _payload("tdccp_inference")
    assert payload["status"] == "ready"
    assert payload["summary"]["completed_replications"] == 1000
    assert all(gate["passed"] for gate in payload["summary"]["gates"])
