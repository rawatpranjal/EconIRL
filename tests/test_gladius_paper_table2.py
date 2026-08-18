"""Fail-closed contracts for the GLADIUS Table 2 replication receipt."""

from __future__ import annotations

from validation.estimators.gladius.paper_table2_mape import (
    PAPER_GLADIUS,
    TARGET_REPETITIONS,
    TARGET_SIZES,
    summarize_records,
)


def _passing_records() -> list[dict]:
    return [
        {
            "n_traj": n_traj,
            "seed": seed,
            "mape": PAPER_GLADIUS[n_traj],
            "nfxp_oracle_mape": 1.0,
            "optimization": {"oracle_selected": True},
        }
        for n_traj in TARGET_SIZES
        for seed in range(TARGET_REPETITIONS)
    ]


def test_table2_summary_accepts_only_the_frozen_6x20_design() -> None:
    summary = summarize_records(_passing_records())

    assert summary["all_passed"] is True
    assert summary["gates"]["full_6x20_design"] is True
    assert len(summary["cells"]) == 6


def test_table2_summary_rejects_a_missing_replication() -> None:
    summary = summarize_records(_passing_records()[:-1])

    assert summary["all_passed"] is False
    assert summary["gates"]["full_6x20_design"] is False


def test_table2_summary_rejects_a_cell_above_paper_mean_plus_2se() -> None:
    records = _passing_records()
    for record in records:
        if record["n_traj"] == 5000:
            record["mape"] = 1.0

    summary = summarize_records(records)

    assert summary["all_passed"] is False
    assert summary["gates"]["all_cells_within_paper_mean_plus_2se"] is False
    assert summary["gates"]["post_250_non_deterioration"] is False
