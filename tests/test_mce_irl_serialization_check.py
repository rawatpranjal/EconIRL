"""Fresh-process serialization checks for the MCE-IRL workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_serialization_runner_uses_a_fresh_process(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            "validation/estimators/mce_irl/serialization_check.py",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["fresh_process"] is True
    assert payload["summary_equal"] is True
    assert all(gap <= 1e-12 for gap in payload["maximum_absolute_gaps"].values())


def test_release_receipt_records_wheel_process_parity() -> None:
    path = Path("validation/results/mce_irl_serialization.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["fresh_process"] is True
    assert payload["summary_equal"] is True
    assert payload["module_outside_checkout"] is True
    assert payload["wheel_origin_required"] is True
    assert not Path(payload["econirl_module"]).is_relative_to(Path.cwd())
    assert all(gap <= payload["threshold"] for gap in payload["maximum_absolute_gaps"].values())
