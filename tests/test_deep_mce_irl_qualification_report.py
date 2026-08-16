"""Fail-closed contracts for the Neural MCE-IRL qualification report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation.estimators.deep_mce_irl.qualification_report import (
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
    bootstrap_summary = receipts["bootstrap"]["summary"]
    assert f"{bootstrap_summary['n_usable']}/50 usable panels" in report
    assert "fresh wheel process" in report
    assert "not a paper-number replication" in report


def test_qualification_report_rejects_one_failed_check() -> None:
    receipts = load_receipts(DEFAULT_INPUTS)
    broken = copy.deepcopy(receipts)
    broken["bootstrap"]["checks"][0]["passed"] = False
    assert failures(broken)[0] == "bootstrap:usable_panels"


def test_qualification_report_rejects_changed_bootstrap_design() -> None:
    receipts = load_receipts(DEFAULT_INPUTS)
    broken = copy.deepcopy(receipts)
    broken["bootstrap"]["design"]["bootstrap_draws_per_panel"] = 98
    assert "bootstrap:draws_per_panel=98" in failures(broken)


def test_qualification_report_requires_every_receipt(tmp_path: Path) -> None:
    paths = {name: tmp_path / path.name for name, path in DEFAULT_INPUTS.items()}
    one_name = next(iter(paths))
    paths[one_name].write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing qualification receipt"):
        load_receipts(paths)
