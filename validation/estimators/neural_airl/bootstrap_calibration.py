#!/usr/bin/env python3
"""Repeated-panel trajectory-bootstrap evidence for NeuralAIRL functionals."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "neural_airl_bootstrap.json"
DEFAULT_CHECKPOINT = ROOT / "validation" / "results" / "neural_airl_bootstrap.jsonl"
FULL_PANELS = 10
FULL_DRAWS = 9

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


THRESHOLDS = {
    "minimum_reward_coverage": 0.75,
    "minimum_policy_coverage": 0.80,
    "maximum_reward_width_cv": 0.80,
    "maximum_policy_width_cv": 0.80,
    "minimum_oracle_policy_shift": 0.01,
}


def identifiable_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the nonlinear reward under deterministic decomposable dynamics."""
    from validation.estimators.neural_airl.nonlinear_recovery import problem

    inputs, _, _, true_reward = problem()
    transitions = np.zeros((2, 9, 9), dtype=np.float64)
    changed = np.zeros_like(transitions)
    for state in range(9):
        transitions[0, state, (state + 1) % 9] = 1.0
        transitions[1, state, (state - 1) % 9] = 1.0
        changed[0, state, (state + 2) % 9] = 1.0
        changed[1, state, (state - 1) % 9] = 1.0
    return inputs, transitions, changed, true_reward


def fit_panel(panel_index: int, *, draws: int, smoke: bool) -> dict[str, Any]:
    from econirl import NeuralAIRL
    from econirl.evaluation.selfcheck import assert_effect
    from validation.estimators.neural_airl.nonlinear_recovery import (
        policy_tv,
        simulate,
        solve,
    )

    inputs, transitions, changed, true_reward = identifiable_problem()
    panel = simulate(28_000 + panel_index, smoke=smoke, transitions=transitions)
    model = NeuralAIRL(
        n_states=9,
        n_actions=2,
        discount=0.9,
        feature_matrix=inputs,
        reward_hidden_dim=32,
        reward_num_layers=2,
        shaping_hidden_dim=32,
        policy_hidden_dim=32,
        policy_steps=15,
        discriminator_steps=3,
        max_rounds=120 if smoke else 160,
        min_rounds=55 if smoke else 70,
        policy_step_size=0.1,
        compute_se=True,
        n_bootstrap=draws,
        seed=29_000 + panel_index,
        se_seed=30_000 + panel_index,
    ).fit(panel, transitions=transitions)
    assert model.bootstrap_ is not None
    intervals = model.conf_int()
    centered_truth = true_reward - true_reward.mean()
    oracle = solve(true_reward, transitions)
    changed_oracle = solve(true_reward, changed)

    reward_hits = 0
    reward_widths: list[float] = []
    for state in range(9):
        lower, upper = intervals[f"reward[{state}]"]
        reward_hits += lower <= centered_truth[state] <= upper
        reward_widths.append(upper - lower)

    policy_hits = 0
    policy_widths: list[float] = []
    for state in range(9):
        for action in range(2):
            lower, upper = intervals[f"policy[{state},{action}]"]
            policy_hits += lower <= oracle.policy[state, action] <= upper
            policy_widths.append(upper - lower)

    counterfactual = model.counterfactual(transitions=changed)
    oracle_shift = policy_tv(changed_oracle.policy, oracle.policy)
    assert_effect(oracle_shift, min_abs=THRESHOLDS["minimum_oracle_policy_shift"])
    if not reward_widths or not policy_widths:
        raise RuntimeError("bootstrap self-check found no interval widths")
    assert_effect(float(np.mean(reward_widths)), min_abs=1e-5)
    assert_effect(float(np.mean(policy_widths)), min_abs=1e-5)
    return {
        "panel": panel_index,
        "panel_seed": 28_000 + panel_index,
        "training_seed": 29_000 + panel_index,
        "bootstrap_seed": 30_000 + panel_index,
        "n_observations": panel.num_observations,
        "converged": bool(model.converged_),
        "bootstrap_draws_requested": draws,
        "bootstrap_draws_successful": int(model.bootstrap_.n_successful),
        "reward_hits": int(reward_hits),
        "reward_total": 9,
        "reward_mean_width": float(np.mean(reward_widths)),
        "policy_hits": int(policy_hits),
        "policy_total": 18,
        "policy_mean_width": float(np.mean(policy_widths)),
        "oracle_policy_shift": oracle_shift,
        "counterfactual_policy_tv": policy_tv(
            counterfactual.counterfactual_policy,
            changed_oracle.policy,
        ),
        "counterfactual_bootstrap_intervals": counterfactual.metadata.get("bootstrap_intervals"),
    }


def read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["panel"])] = record
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def coefficient_of_variation(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / array.mean()) if len(array) > 1 else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    reward_hits = sum(record["reward_hits"] for record in records)
    reward_total = sum(record["reward_total"] for record in records)
    policy_hits = sum(record["policy_hits"] for record in records)
    policy_total = sum(record["policy_total"] for record in records)
    reward_widths = [record["reward_mean_width"] for record in records]
    policy_widths = [record["policy_mean_width"] for record in records]
    return {
        "n_requested": len(records),
        "n_converged": sum(record["converged"] for record in records),
        "n_with_counterfactual_intervals": sum(
            record["counterfactual_bootstrap_intervals"] is not None for record in records
        ),
        "bootstrap_draws_requested": sum(record["bootstrap_draws_requested"] for record in records),
        "bootstrap_draws_successful": sum(
            record["bootstrap_draws_successful"] for record in records
        ),
        "reward_coverage": reward_hits / reward_total,
        "policy_coverage": policy_hits / policy_total,
        "reward_width_mean": float(np.mean(reward_widths)),
        "reward_width_cv": coefficient_of_variation(reward_widths),
        "policy_width_mean": float(np.mean(policy_widths)),
        "policy_width_cv": coefficient_of_variation(policy_widths),
        "oracle_policy_shift_minimum": min(record["oracle_policy_shift"] for record in records),
        "counterfactual_policy_tv_p95": float(
            np.percentile([record["counterfactual_policy_tv"] for record in records], 95)
        ),
    }


def checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    specifications = {
        "all_fits_converged": (
            summary["n_converged"],
            "==",
            summary["n_requested"],
        ),
        "all_counterfactual_intervals_present": (
            summary["n_with_counterfactual_intervals"],
            "==",
            summary["n_requested"],
        ),
        "all_bootstrap_draws_successful": (
            summary["bootstrap_draws_successful"],
            "==",
            summary["bootstrap_draws_requested"],
        ),
        "reward_coverage": (
            summary["reward_coverage"],
            ">=",
            THRESHOLDS["minimum_reward_coverage"],
        ),
        "policy_coverage": (
            summary["policy_coverage"],
            ">=",
            THRESHOLDS["minimum_policy_coverage"],
        ),
        "reward_width_cv": (
            summary["reward_width_cv"],
            "<=",
            THRESHOLDS["maximum_reward_width_cv"],
        ),
        "policy_width_cv": (
            summary["policy_width_cv"],
            "<=",
            THRESHOLDS["maximum_policy_width_cv"],
        ),
        "oracle_policy_shift": (
            summary["oracle_policy_shift_minimum"],
            ">=",
            THRESHOLDS["minimum_oracle_policy_shift"],
        ),
    }
    output: list[dict[str, Any]] = []
    for name, (value, operator, threshold) in specifications.items():
        passed = (
            value == threshold
            if operator == "=="
            else value <= threshold
            if operator == "<="
            else value >= threshold
        )
        output.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--panels", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    n_panels = args.panels or (2 if args.smoke else FULL_PANELS)
    draws = args.draws or (3 if args.smoke else FULL_DRAWS)
    existing = {} if args.no_resume else read_checkpoint(args.checkpoint)
    for panel_index in range(n_panels):
        if panel_index in existing:
            continue
        record = fit_panel(panel_index, draws=draws, smoke=args.smoke)
        append_checkpoint(args.checkpoint, record)
        existing[panel_index] = record
        print(
            f"panel {panel_index + 1}/{n_panels}: "
            f"{record['bootstrap_draws_successful']}/{draws} draws"
        )
        jax.clear_caches()
        gc.collect()
    records = [existing[index] for index in range(n_panels)]
    summary = summarize(records)
    gate_results = checks(summary)
    receipt = {
        "estimator": "NeuralAIRL",
        "status": "ready" if all(check["passed"] for check in gate_results) else "failed",
        "mode": "smoke" if args.smoke else "full",
        "paper_replication": False,
        "design": {
            "n_panels": n_panels,
            "bootstrap_draws_per_panel": draws,
            "bootstrap_unit": "individual_trajectory",
            "dynamics": "deterministic_decomposable",
            "interval_method": "normal_bootstrap_standard_error",
        },
        "frozen_thresholds": THRESHOLDS,
        "summary": summary,
        "checks": gate_results,
        "records": records,
        "environment": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("econirl", "jax", "jaxlib", "numpy", "optax")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "summary": summary}, indent=2))
    print(f"wrote {args.output}")
    return 0 if receipt["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
