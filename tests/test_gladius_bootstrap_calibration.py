"""Contracts for the frozen GLADIUS trajectory-bootstrap design."""

from __future__ import annotations

from validation.estimators.gladius.bootstrap_calibration import (
    BOOTSTRAP_DRAWS,
    CALIBRATION_PANELS,
    summarize,
)


def _record(replication: int) -> dict:
    cells = []
    for family, truth in (("reward", -1.0), ("policy", 0.25)):
        for state in (0, 1, 3):
            cells.append(
                {
                    "family": family,
                    "name": f"{family}[{state}]",
                    "truth": truth,
                    "lower": truth - 0.1,
                    "upper": truth + 0.1,
                    "width": 0.2,
                    "covered": True,
                    "lower_miss": False,
                    "upper_miss": False,
                }
            )
    return {
        "replication": replication,
        "n_requested": BOOTSTRAP_DRAWS,
        "n_successful": BOOTSTRAP_DRAWS,
        "success_rate": 1.0,
        "cells": cells,
    }


def test_bootstrap_summary_accepts_only_the_frozen_final_design() -> None:
    records = [_record(replication) for replication in range(CALIBRATION_PANELS)]

    payload = summarize(records, final_run=True)

    assert payload["all_passed"] is True
    assert payload["design"]["panels"] == 20
    assert payload["design"]["draws_per_panel"] == 19


def test_bootstrap_summary_does_not_promote_a_smoke_run() -> None:
    payload = summarize([_record(0), _record(1)], final_run=False)

    assert payload["all_passed"] is False
    assert payload["gates"]["final_design"] is False
