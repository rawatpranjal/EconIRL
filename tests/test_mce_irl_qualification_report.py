"""Fail-closed contracts for the MCE-IRL qualification report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation.estimators.mce_irl.qualification_report import (
    DEFAULT_INPUTS,
    failures,
    load_receipts,
    render,
)


def test_qualification_report_accepts_all_current_receipts() -> None:
    receipts = load_receipts(DEFAULT_INPUTS)
    assert failures(receipts) == []
    report = render(receipts)
    assert "300/300 fits" in report
    assert "4950/4950 draws" in report
    assert "welfare levels withheld" in report
    assert "fresh wheel process" in report


def test_qualification_report_rejects_one_failed_gate() -> None:
    receipts = load_receipts(DEFAULT_INPUTS)
    broken = copy.deepcopy(receipts)
    broken["bootstrap"]["gates"][0]["passed"] = False
    assert failures(broken) == ["bootstrap:usable_rate"]


def test_qualification_report_requires_every_receipt(tmp_path: Path) -> None:
    paths = {name: tmp_path / path.name for name, path in DEFAULT_INPUTS.items()}
    one_name = next(iter(paths))
    paths[one_name].write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing qualification receipt"):
        load_receipts(paths)
