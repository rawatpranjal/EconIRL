"""Fail-closed contracts for the GLADIUS qualification report."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from validation.estimators.gladius.qualification_report import (
    failures,
    load_receipts,
    render,
)


def _ready_receipts() -> dict:
    return {
        "known_truth": {
            "status": "strict_structural_counterfactual_pass",
            "gates": [{"name": f"gate_{index}", "passed": True} for index in range(12)],
            "metrics": {
                "raw_bellman_reward_normalized_rmse": 0.2,
                "q_normalized_rmse": 0.1,
                "value_normalized_rmse": 0.2,
            },
        },
        "bootstrap": {
            "design": {"panels": 20, "draws_per_panel": 19},
            "usable_panels": 20,
            "reward": {"coverage": 0.9},
            "policy": {"coverage": 0.9},
            "gates": {"coverage": True, "exact_seeded_reproducibility": True},
            "all_passed": True,
        },
        "paper": {
            "cells": [
                {"n_traj": size, "mean_mape": 0.1} for size in (50, 250, 500, 1000, 2500, 5000)
            ],
            "gates": {"full_6x20_design": True},
            "all_passed": True,
            "selection_boundary": (
                "simulation-only true-held-out-MAPE selection; not used by public fit"
            ),
        },
        "serialization": {
            "status": "ready",
            "fresh_process": True,
            "module_outside_checkout": True,
            "wheel_origin_required": True,
            "summary_equal": True,
            "confidence_intervals_equal": True,
            "maximum_absolute_gaps": {"reward": 0.0},
        },
        "notebook": {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "outputs": [
                        {"output_type": "stream", "text": ["Installed package import: True\n"]}
                    ],
                }
            ]
        },
    }


def test_qualification_report_accepts_complete_evidence() -> None:
    receipts = _ready_receipts()

    assert failures(receipts) == []
    report = render(receipts)
    assert "12/12" in report
    assert "20/20 usable panels" in report
    assert "fresh wheel process" in report


def test_qualification_report_rejects_one_failed_gate() -> None:
    receipts = copy.deepcopy(_ready_receipts())
    receipts["paper"]["gates"]["full_6x20_design"] = False

    assert failures(receipts) == ["paper:full_6x20_design"]


def test_qualification_report_requires_every_receipt(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in ("known", "bootstrap")}
    notebook = tmp_path / "workflow.ipynb"
    with pytest.raises(FileNotFoundError, match="missing qualification receipt"):
        load_receipts(paths, notebook)
