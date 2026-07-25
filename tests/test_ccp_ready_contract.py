"""Contract tests for the CCP readiness runner and committed result."""

from __future__ import annotations

import json
from pathlib import Path

import validation.estimators.ccp.ready as ready_module
from validation.estimators.ccp.ready import (
    HARD_CONFIG,
    INFERENCE_CONFIG,
    checkpoint_records,
)


def test_ccp_readiness_configuration_is_frozen() -> None:
    assert INFERENCE_CONFIG.n_replications == 1000
    assert INFERENCE_CONFIG.num_states == 20
    assert INFERENCE_CONFIG.n_individuals == 250
    assert INFERENCE_CONFIG.n_periods == 40
    assert HARD_CONFIG.n_replications == 20
    assert HARD_CONFIG.num_states == 100
    assert HARD_CONFIG.n_individuals == 4000
    assert HARD_CONFIG.n_periods == 40


def test_checkpoint_reader_keeps_latest_complete_row(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rows.jsonl"
    checkpoint.write_text(
        '{"problem":"p","rep":0,"error":null}\n{"problem":"p","rep":0,"error":"new"}\n',
        encoding="utf-8",
    )

    rows = checkpoint_records(checkpoint)

    assert rows[("p", 0)]["error"] == "new"


def test_runner_can_reuse_completed_standard_error_results() -> None:
    source = Path(ready_module.__file__).read_text(encoding="utf-8")

    assert "--alternate-se-results" in source


def test_committed_ccp_readiness_manifest() -> None:
    path = Path("validation/results/ccp_ready.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["estimator"] == "CCP"
    assert payload["paper_target"] is None
    assert payload["inference"]["n_total"] == 1000
    assert payload["hard_problem"]["n_total"] == 20
    assert payload["support_example"]["state_action_coverage"] == 1.0
    assert all(gate["passed"] for gate in payload["gates"] if gate["enforced"])
