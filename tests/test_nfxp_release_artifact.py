"""Release artifact checks for the NFXP primer."""

from __future__ import annotations

import json
import math
from pathlib import Path


def test_nfxp_primer_json_artifact_has_release_gates() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "results"
        / "nfxp.json"
    )

    assert artifact.exists()

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    result = payload["result"]
    summary = result["summary"]
    metrics = result["metrics"]
    gate_names = {gate["name"] for gate in result["gates"]}
    counterfactuals = metrics["counterfactuals"]

    assert payload["estimator"] == "NFXP"
    assert payload["release_status"] == "Certified"
    assert payload["primary_cell_id"] == "canonical_low_action"
    assert summary["converged"] is True
    assert summary["num_observations"] > 0
    assert summary["estimation_time"] > 0
    assert metrics["parameters"]["relative_rmse"] < 0.15
    assert metrics["policy"]["tv"] < 0.03
    assert all(gate["passed"] for gate in result["gates"])
    assert set(counterfactuals) == {"type_a", "type_b", "type_c"}
    for cf_metrics in counterfactuals.values():
        assert math.isfinite(cf_metrics["regret"])
        assert math.isfinite(cf_metrics["value_rmse"])
        assert math.isfinite(cf_metrics["policy"]["tv"])
        assert cf_metrics["regret"] < 0.05
        assert cf_metrics["policy"]["tv"] < 0.03
    assert {
        "converged",
        "parameter_cosine",
        "parameter_relative_rmse",
        "policy_tv",
        "value_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    } <= gate_names
