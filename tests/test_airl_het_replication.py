"""Lock the AIRL-Het identification replication (Lee-Sudhir-Wang 2026).

The paper proves (Theorems 1-3) that an exit-action reward anchor plus an
absorbing-state anchor make the action-dependent AIRL discriminator uniquely
recover the true reward and value, with an EM layer recovering latent segments.
The package reproduces this on a controlled two-segment problem.

This locks the recovery from the saved result, so it is fast. Regenerate with
``python validation/estimators/aairl/run.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "validation" / "results" / "aairl.json"


def _cell():
    data = json.loads(RESULT.read_text())
    assert data["estimator"] == "AIRL-Het", data.get("estimator")
    return data["results"][0]


def test_anchored_segment_recovery():
    """Two segments, membership and prior recovered, reward and policy in bound."""
    m = _cell()["metrics"]
    # latent segment membership and sizes
    assert m["segment_assignment_accuracy"] >= 0.7, m["segment_assignment_accuracy"]
    assert m["segment_prior_l1"] <= 0.35, m["segment_prior_l1"]
    # segment-specific reward and policy recovered under the anchors
    assert m["max_segment_reward_normalized_rmse"] <= 0.30, m["max_segment_reward_normalized_rmse"]
    assert m["max_segment_policy_tv"] <= 0.12, m["max_segment_policy_tv"]
    assert len(m["segment_reward_normalized_rmse"]) == 2, m["segment_reward_normalized_rmse"]


def test_all_gates_pass():
    """Every recovery gate on the anchored cell passes."""
    cell = _cell()
    gates = cell.get("gates", [])
    assert gates, "no gates recorded"
    failed = [g["name"] for g in gates if not g.get("passed")]
    assert not failed, failed
