"""Release contracts for MCE-IRL repeated-run and road validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_mce_irl_repeated_run_receipt_is_ready() -> None:
    payload = json.loads(Path("validation/results/mce_irl_ready.json").read_text(encoding="utf-8"))

    assert payload["status"] == "ready"
    assert payload["n_replications"] >= 300
    repeated = payload["repeated_run_inference"]
    assert repeated["n_successful"] == payload["n_replications"]
    assert repeated["coverage_selfcheck_passed"]
    assert 0.91 <= repeated["coverage"] <= 0.99
    assert abs(repeated["bias"]) <= 0.05
    assert repeated["max_stationarity_residual"] <= 1e-6
    assert payload["standard_error_check"]["passed"]
    assert payload["intervention_check"]["passed"]
    assert all(gate["passed"] for gate in payload["gates"])


def test_ziebart_synthetic_receipt_separates_shape_from_replication() -> None:
    payload = json.loads(
        Path("validation/results/mce_irl_ziebart_synthetic.json").read_text(encoding="utf-8")
    )

    assert payload["passed"]
    assert "does not reproduce" in payload["claim"]
    assert payload["network"]["num_states"] > 300_000
    assert payload["network"]["num_action_slots"] > 900_000
    assert len(payload["network"]["feature_names"]) == 22
    assert payload["data"]["train_trips"] == 1_851
    assert payload["data"]["test_trips"] == 7_403
    assert payload["fit"]["converged"]
    assert payload["fit"]["stationarity_residual"] <= 0.02
    assert np.isfinite(list(payload["synthetic_metrics"].values())).all()
    assert payload["paper"]["reported_table_1_targets"] == {
        "distance_match_percent": 78.79,
        "routes_at_least_90_percent": 52.98,
        "average_log_probability": -6.85,
    }
