"""Frozen contracts for Neural MCE-IRL bootstrap calibration evidence."""

from __future__ import annotations

import json
from pathlib import Path

from validation.estimators.deep_mce_irl.bootstrap_calibration import (
    load_checkpoint,
    run_panels,
)


def test_checkpoint_rejects_records_from_a_different_draw_count(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mixed.jsonl"
    records = [
        {"panel_index": 0, "n_requested": 5, "usable": True},
        {"panel_index": 1, "n_requested": 99, "usable": True},
    ]
    checkpoint.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert load_checkpoint(checkpoint, n_draws=99) == {1: records[1]}


def test_resume_compacts_incompatible_smoke_records(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mixed.jsonl"
    smoke = {"panel_index": 0, "n_requested": 5, "usable": True}
    full = {"panel_index": 0, "n_requested": 99, "usable": True}
    checkpoint.write_text(
        json.dumps(smoke) + "\n" + json.dumps(full) + "\n",
        encoding="utf-8",
    )

    records = run_panels(
        1,
        99,
        smoke=False,
        checkpoint=checkpoint,
        quiet=True,
    )

    assert records == [full]
    assert checkpoint.read_text(encoding="utf-8").splitlines() == [json.dumps(full)]


def test_full_bootstrap_calibration_is_ready() -> None:
    payload = json.loads(
        Path("validation/results/deep_mce_irl_bootstrap_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["mode"] == "full"
    assert payload["status"] == "ready"
    assert payload["paper_replication"] is False
    assert payload["design"]["n_panels"] == 50
    assert payload["design"]["bootstrap_draws_per_panel"] == 99
    assert payload["design"]["bootstrap_unit"] == "individual_trajectory"
    assert payload["design"]["individuals_per_panel"] == 300
    assert payload["design"]["periods_per_individual"] == 1
    assert payload["design"]["target_states"] == [0, 1, 2]
    assert payload["design"]["anchor_action"] == 0
    assert payload["summary"]["n_usable"] >= 48
    assert payload["summary"]["bootstrap_draws_requested"] == 4_950
    assert payload["summary"]["bootstrap_draws_successful"] >= 4_703
    assert all(
        targets["mean_policy_tv"] >= payload["frozen_thresholds"]["minimum_oracle_policy_tv"]
        for targets in payload["summary"]["oracle_counterfactual_targets"].values()
    )
    assert all(check["passed"] for check in payload["checks"])
