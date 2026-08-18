#!/usr/bin/env python3
"""Calibrate GLADIUS whole-trajectory bootstrap intervals on known truth."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl import GLADIUS  # noqa: E402
from econirl.core.reward_spec import RewardSpec  # noqa: E402
from econirl.core.types import Panel, Trajectory  # noqa: E402

OUTPUT = ROOT / "validation" / "results" / "gladius_bootstrap_calibration.json"
CALIBRATION_PANELS = 20
BOOTSTRAP_DRAWS = 19
BASE_SEED = 81_000
SELECTED_STATES = (0, 1, 3)
SELECTED_ACTION = 1
N_STATES = 4
N_ACTIONS = 2
FIT_MAX_EPOCHS = 300
FIT_PATIENCE = 20


def _controlled_case(seed: int) -> tuple[Panel, RewardSpec, np.ndarray, np.ndarray, np.ndarray]:
    """Return a small identified DGP with state-independent choice contrasts."""
    rng = np.random.default_rng(seed)
    reward = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    reward[:, 1] = -1.0
    policy = np.repeat(
        np.asarray([[1.0, np.exp(-1.0)]]) / (1.0 + np.exp(-1.0)),
        N_STATES,
        axis=0,
    )
    transitions = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=float)
    for action in range(N_ACTIONS):
        for state in range(N_STATES):
            transitions[action, state, (state + 1) % N_STATES] = 1.0
    trajectories = []
    for individual in range(100):
        state = individual % N_STATES
        states = []
        actions = []
        next_states = []
        for _ in range(30):
            action = int(rng.choice(N_ACTIONS, p=policy[state]))
            next_state = (state + 1) % N_STATES
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            state = next_state
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
                individual_id=individual,
            )
        )
    feature_matrix = np.zeros((N_STATES, N_ACTIONS, 1), dtype=np.float32)
    feature_matrix[:, 1, 0] = -1.0
    features = RewardSpec(jnp.asarray(feature_matrix), names=["action_one_cost"])
    return Panel(trajectories=trajectories), features, transitions, reward, policy


def fit_panel(replication: int, n_bootstrap: int) -> dict[str, Any]:
    seed = BASE_SEED + replication
    panel, features, transitions, true_reward, true_policy = _controlled_case(seed)
    model = GLADIUS(
        n_actions=2,
        discount=0.95,
        q_hidden_dim=8,
        q_num_layers=1,
        ev_hidden_dim=8,
        ev_num_layers=1,
        batch_size=64,
        max_epochs=FIT_MAX_EPOCHS,
        patience=FIT_PATIENCE,
        lr_decay_rate=5e-4,
        anchor_action=0,
        anchor_rewards=tuple([0.0] * N_STATES),
        compute_se=True,
        n_bootstrap=n_bootstrap,
        seed=seed,
        se_seed=seed + 100_000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        model.fit(
            panel,
            features=features,
            transitions=transitions,
        )
    bootstrap = model.bootstrap_
    if bootstrap is None:
        raise RuntimeError("public GLADIUS bootstrap result was not populated")

    reward_draws = np.asarray(bootstrap.reward_draws, dtype=float)
    policy_draws = np.asarray(bootstrap.policy_draws, dtype=float)

    cells = []
    for state in SELECTED_STATES:
        reward_values = reward_draws[:, state, SELECTED_ACTION]
        policy_values = policy_draws[:, state, SELECTED_ACTION]
        cells.extend(
            [
                _cell_record(
                    family="reward",
                    name=f"reward[{state},{SELECTED_ACTION}]",
                    truth=float(true_reward[state, SELECTED_ACTION]),
                    draws=reward_values,
                ),
                _cell_record(
                    family="policy",
                    name=f"policy[{state},{SELECTED_ACTION}]",
                    truth=float(true_policy[state, SELECTED_ACTION]),
                    draws=policy_values,
                ),
            ]
        )
    return {
        "replication": replication,
        "panel_seed": seed,
        "bootstrap_seed": model.se_seed,
        "n_requested": n_bootstrap,
        "n_successful": int(bootstrap.n_successful),
        "success_rate": float(bootstrap.n_successful / n_bootstrap),
        "failures": list(bootstrap.failures),
        "base_fit_converged": bool(model.converged_),
        "base_fit_iterations": int(model.n_iter_),
        "base_fit_termination": model.termination_reason_,
        "cells": cells,
    }


def _cell_record(
    *,
    family: str,
    name: str,
    truth: float,
    draws: np.ndarray,
) -> dict[str, Any]:
    lower, upper = np.quantile(
        draws,
        [0.025, 0.975],
        method="inverted_cdf",
    )
    return {
        "family": family,
        "name": name,
        "truth": truth,
        "lower": float(lower),
        "upper": float(upper),
        "width": float(upper - lower),
        "covered": bool(lower <= truth <= upper),
        "lower_miss": bool(truth < lower),
        "upper_miss": bool(truth > upper),
    }


def _family_summary(records: list[dict[str, Any]], family: str) -> dict[str, Any]:
    cells = [cell for record in records for cell in record["cells"] if cell["family"] == family]
    widths = np.asarray([cell["width"] for cell in cells], dtype=float)
    median_width = float(np.median(widths))
    return {
        "n_intervals": len(cells),
        "coverage": float(np.mean([cell["covered"] for cell in cells])),
        "lower_tail_miss_rate": float(np.mean([cell["lower_miss"] for cell in cells])),
        "upper_tail_miss_rate": float(np.mean([cell["upper_miss"] for cell in cells])),
        "median_width": median_width,
        "p95_width": float(np.percentile(widths, 95)),
        "p95_to_median_width": (
            float(np.percentile(widths, 95) / median_width) if median_width > 0 else float("inf")
        ),
        "all_widths_positive": bool(np.all(widths > 0)),
    }


def summarize(records: list[dict[str, Any]], *, final_run: bool) -> dict[str, Any]:
    usable = [record for record in records if record["success_rate"] >= 0.95]
    reward = _family_summary(usable, "reward")
    policy = _family_summary(usable, "policy")
    gates = {
        "final_design": final_run,
        "usable_panel_rate": len(usable) / len(records) >= 0.95,
        "minimum_draw_success": min(record["success_rate"] for record in records) >= 0.95,
        "base_fit_convergence": all(record["base_fit_converged"] for record in records),
        "reward_coverage": reward["coverage"] >= 0.85,
        "policy_coverage": policy["coverage"] >= 0.85,
        "reward_tail_misses": max(reward["lower_tail_miss_rate"], reward["upper_tail_miss_rate"])
        <= 0.10,
        "policy_tail_misses": max(policy["lower_tail_miss_rate"], policy["upper_tail_miss_rate"])
        <= 0.10,
        "positive_widths": reward["all_widths_positive"] and policy["all_widths_positive"],
        "reward_width_stability": reward["p95_to_median_width"] <= 4.0,
        "policy_width_stability": policy["p95_to_median_width"] <= 4.0,
    }
    return {
        "protocol_history": {
            "initial_final_run": {
                "fit_max_epochs": 80,
                "fit_patience": 81,
                "reward_coverage": 0.6833333333333333,
                "policy_coverage": 0.7166666666666667,
                "all_passed": False,
            },
            "remediation": (
                "require converged base and bootstrap refits; allow max 300 epochs "
                "with patience 20; frozen coverage and tail thresholds unchanged"
            ),
        },
        "design": {
            "panels": len(records),
            "draws_per_panel": records[0]["n_requested"],
            "resampling_unit": "individual trajectory",
            "selected_states": list(SELECTED_STATES),
            "selected_action": SELECTED_ACTION,
            "fit_max_epochs": FIT_MAX_EPOCHS,
            "fit_patience": FIT_PATIENCE,
        },
        "usable_panels": len(usable),
        "reward": reward,
        "policy": policy,
        "gates": gates,
        "all_passed": all(gates.values()),
        "records": records,
    }


def reproducibility_check(n_bootstrap: int) -> dict[str, Any]:
    first = fit_panel(10_000, n_bootstrap)
    second = fit_panel(10_000, n_bootstrap)
    exact = first == second
    return {
        "exact": exact,
        "replication": 10_000,
        "n_bootstrap": n_bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=int, default=CALIBRATION_PANELS)
    parser.add_argument("--start-panel", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-reproducibility", action="store_true")
    parser.add_argument("--merge-shard", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.merge_shard:
        by_replication: dict[int, dict[str, Any]] = {}
        for shard in args.merge_shard:
            payload = json.loads(shard.read_text(encoding="utf-8"))
            for record in payload["records"]:
                replication = int(record["replication"])
                if replication in by_replication and by_replication[replication] != record:
                    raise ValueError(f"conflicting bootstrap shard for panel {replication}")
                by_replication[replication] = record
        records = [by_replication[key] for key in sorted(by_replication)]
        final_run = (
            len(records) >= CALIBRATION_PANELS and records[0]["n_requested"] >= BOOTSTRAP_DRAWS
        )
        output = args.output
    else:
        records = []
        panels = min(args.panels, 2) if args.smoke else args.panels
        n_bootstrap = min(args.n_bootstrap, 3) if args.smoke else args.n_bootstrap
        for replication in range(args.start_panel, args.start_panel + panels):
            record = fit_panel(replication, n_bootstrap)
            records.append(record)
            print(
                f"bootstrap panel {replication + 1}: "
                f"{record['n_successful']}/{record['n_requested']} draws",
                flush=True,
            )
        final_run = (
            args.start_panel == 0
            and panels >= CALIBRATION_PANELS
            and n_bootstrap >= BOOTSTRAP_DRAWS
        )
        output = Path("/tmp/gladius_bootstrap_smoke.json") if args.smoke else args.output

    payload = summarize(records, final_run=final_run)
    if args.skip_reproducibility:
        payload["reproducibility"] = {"exact": None, "skipped": True}
    else:
        payload["reproducibility"] = reproducibility_check(3)
    payload["gates"]["exact_seeded_reproducibility"] = bool(payload["reproducibility"]["exact"])
    payload["all_passed"] = all(payload["gates"].values())
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote: {output}")
    if final_run and not payload["all_passed"]:
        raise SystemExit("GLADIUS bootstrap calibration gates failed")


if __name__ == "__main__":
    main()
