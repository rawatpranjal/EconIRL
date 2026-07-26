"""Contract checks for the CCP Rust Table IX receipt."""

from __future__ import annotations

import json
from pathlib import Path

from validation.estimators.ccp.rust_table_ix import (
    EXPECTED_RAW_SHA256,
    verify_receipt,
)


def test_committed_ccp_table_ix_receipt_has_all_gates_passing() -> None:
    receipt_path = Path("validation/results/ccp_rust_table_ix.json")
    verify_receipt(receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert payload["status"] == "pass"
    assert payload["source"]["sha256"] == EXPECTED_RAW_SHA256
    assert payload["source"]["path"] == ("downloads/nfxp_unzip/nfxp/dat/a530875.asc")
    assert payload["source"]["sample_observations"] == 4292
    assert payload["specification"]["transition_orientation"] == ("(n_actions, n_states, n_states)")
    assert payload["ccp"]["joint_parameter_names"] == [
        "operating_cost",
        "replacement_cost",
        "transition_p0",
        "transition_p1",
    ]
    assert payload["ccp"]["npl_parameter_residual"] <= 1e-12
    assert payload["ccp"]["npl_policy_residual"] <= 1e-12
    assert payload["ccp"]["bellman_policy_residual"] <= 1e-8
    assert all(gate["passed"] for gate in payload["gates"].values())
