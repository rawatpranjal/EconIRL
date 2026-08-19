"""Fail-closed contracts for the GLADIUS Table 2 replication receipt."""

from __future__ import annotations

from validation.estimators.gladius.paper_table2_mape import (
    PAPER_GLADIUS,
    PAPER_RECIPE_RECEIPT,
    TARGET_REPETITIONS,
    TARGET_SIZES,
    qualification_batch_size,
    qualification_min_q_updates,
    run_one,
    summarize_records,
)


def _passing_records() -> list[dict]:
    return [
        {
            "n_traj": n_traj,
            "seed": seed,
            "mape": PAPER_GLADIUS[n_traj],
            "nfxp_oracle_mape": 1.0,
            "optimization": {
                "oracle_selected": True,
                "batch_size": qualification_batch_size(n_traj),
                "q_updates_per_epoch": qualification_min_q_updates(n_traj),
                **PAPER_RECIPE_RECEIPT,
            },
        }
        for n_traj in TARGET_SIZES
        for seed in range(TARGET_REPETITIONS)
    ]


def test_table2_summary_accepts_only_the_frozen_6x20_design() -> None:
    assert PAPER_RECIPE_RECEIPT["seed"] == 2

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


def test_table2_summary_rejects_a_mislabeled_training_recipe() -> None:
    records = _passing_records()
    records[0]["optimization"]["network_mode"] = "separate"

    summary = summarize_records(records)

    assert summary["all_passed"] is False
    assert summary["gates"]["paper_recipe_disclosed"] is False

    records = _passing_records()
    records[0]["optimization"]["batch_size"] = 32

    summary = summarize_records(records)

    assert summary["all_passed"] is False
    assert summary["gates"]["paper_recipe_disclosed"] is False


def test_every_table2_cell_gets_enough_q_updates_to_recover_reward() -> None:
    """Protect the bad-seed failure that fixed batch size 32 concealed.

    The floor is stated in Q updates per epoch, not in trajectories. A fixed
    small batch also fails, the other way: it leaves the largest cell
    gradient-noise limited. So the batch never drops below what the floor needs
    and never rises above the author's own 32.
    """
    batch_size = qualification_batch_size(50)

    for n_traj in TARGET_SIZES:
        assert qualification_min_q_updates(n_traj) >= 10, n_traj
        assert 2 <= qualification_batch_size(n_traj) <= 32, n_traj
    assert qualification_batch_size(5000) == 32

    record = run_one(
        50,
        3,
        max_epochs=200,
        batch_size=batch_size,
        oracle=False,
    )

    assert batch_size == 2
    assert record["mape"] <= 6.0, record
