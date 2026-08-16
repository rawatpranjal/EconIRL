"""Contract for the Neural MCE-IRL serialization receipt."""

from __future__ import annotations

import json
from pathlib import Path


def test_serialization_receipt_passes_every_check() -> None:
    path = Path("validation/results/deep_mce_irl_serialization.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["estimator"] == "MCEIRLNeural"
    assert payload["status"] == "passed"
    assert payload["econirl_version"] == "0.0.10"
    assert payload["fresh_process"] is True
    assert payload["summary_equal"] is True
    assert payload["confidence_intervals_equal"] is True
    assert payload["counterfactual_intervals_equal"] is True
    assert all(gap <= payload["threshold"] for gap in payload["maximum_absolute_gaps"].values())
