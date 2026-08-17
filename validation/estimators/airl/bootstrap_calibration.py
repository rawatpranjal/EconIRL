#!/usr/bin/env python3
"""Calibrate AIRL trajectory-bootstrap intervals on an identified state reward."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "airl_bootstrap_calibration.json"
DEFAULT_CHECKPOINT = ROOT / "validation" / "results" / "airl_bootstrap_calibration.jsonl"
FULL_PANELS = 20
FULL_DRAWS = 19

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


FROZEN_THRESHOLDS = {
    "minimum_usable_panel_rate": 0.95,
    "minimum_reward_coverage": 0.85,
    "minimum_policy_coverage": 0.85,
    "maximum_tail_miss": 0.10,
    "maximum_reward_absolute_bias": 0.03,
    "maximum_policy_absolute_bias": 0.02,
    "maximum_p95_median_width_ratio": 4.0,
}


def build_problem():
    """Return the small decomposable state-only calibration problem."""
    from econirl.environments.shapeshifter import ShapeshifterConfig
    from validation.known_truth import ShapeshifterKnownTruthConfig, build_known_truth_dgp

    return build_known_truth_dgp(
        ShapeshifterKnownTruthConfig(
            ShapeshifterConfig(
                num_states=6,
                num_actions=2,
                num_features=2,
                reward_type="linear",
                feature_type="linear",
                action_dependent=False,
                stochastic_transitions=False,
                stochastic_rewards=False,
                discount_factor=0.9,
                reward_scale=3.0,
                seed=221,
            )
        )
    )


def fit_panel(panel_index: int, draws: int) -> dict[str, Any]:
    """Fit one simulated panel and retain interval-level coverage records."""
    from econirl import AIRL, RewardSpec
    from validation.known_truth import (
        SimulationConfig,
        simulate_known_truth_panel,
        solve_known_truth,
    )

    dgp = build_problem()
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(
            n_individuals=160,
            n_periods=40,
            seed=30_000 + panel_index,
        ),
    )
    reward = RewardSpec.state_dependent(
        dgp.feature_matrix[:, 0, :],
        dgp.parameter_names,
        dgp.problem.num_actions,
    )
    try:
        model = AIRL(
            n_states=dgp.problem.num_states,
            n_actions=dgp.problem.num_actions,
            discount=dgp.problem.discount_factor,
            reward_lr=0.1,
            max_rounds=45,
            min_rounds=45,
            discriminator_steps=3,
            compute_se=True,
            n_bootstrap=draws,
            seed=40_000 + panel_index,
            se_seed=50_000 + panel_index,
        ).fit(panel, transitions=np.asarray(dgp.transitions), reward=reward)
        intervals = model.conf_int()
        truth_reward = np.asarray(dgp.homogeneous_reward[:, 0], dtype=float)
        truth_reward -= truth_reward.mean()
        truth_policy = np.asarray(solve_known_truth(dgp).policy, dtype=float)
        reward_records = _interval_records(
            "reward",
            truth_reward,
            np.asarray(model.reward_),
            intervals,
        )
        policy_records = _interval_records(
            "policy",
            truth_policy,
            np.asarray(model.policy_),
            intervals,
        )
        assert model.bootstrap_ is not None
        return {
            "panel_index": panel_index,
            "panel_seed": 30_000 + panel_index,
            "successful_draws": model.bootstrap_.n_successful,
            "requested_draws": draws,
            "reward": reward_records,
            "policy": policy_records,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are calibration evidence
        return {
            "panel_index": panel_index,
            "panel_seed": 30_000 + panel_index,
            "requested_draws": draws,
            "successful_draws": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _interval_records(
    family: str,
    truth: np.ndarray,
    estimate: np.ndarray,
    intervals: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Flatten one functional family into auditable scalar records."""
    records = []
    for index in np.ndindex(truth.shape):
        suffix = ",".join(str(value) for value in index)
        name = f"{family}[{suffix}]"
        lower, upper = intervals[name]
        target = float(truth[index])
        fitted = float(estimate[index])
        records.append(
            {
                "name": name,
                "truth": target,
                "estimate": fitted,
                "lower": lower,
                "upper": upper,
                "covered": bool(lower <= target <= upper),
                "lower_miss": bool(target < lower),
                "upper_miss": bool(target > upper),
                "width": float(upper - lower),
                "bias": float(fitted - target),
            }
        )
    return records


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        int(record["panel_index"]): record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


def run_panels(
    checkpoint: Path,
    n_panels: int,
    draws: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Run missing panels in spawn-safe workers and checkpoint each result."""
    completed = load_checkpoint(checkpoint)
    missing = [index for index in range(n_panels) if index not in completed]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fit_panel, index, draws): index for index in missing}
            for future in as_completed(futures):
                record = future.result()
                completed[int(record["panel_index"])] = record
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                print(
                    f"panel {record['panel_index'] + 1}/{n_panels}: "
                    f"{record.get('error') or str(record['successful_draws']) + ' draws'}"
                )
    return [completed[index] for index in range(n_panels)]


def summarize_family(records: list[dict[str, Any]], family: str) -> dict[str, Any]:
    rows = [row for record in records if record.get("error") is None for row in record[family]]
    widths = np.asarray([row["width"] for row in rows], dtype=float)
    biases = np.asarray([row["bias"] for row in rows], dtype=float)
    median_width = float(np.median(widths))
    return {
        "n_intervals": len(rows),
        "coverage": float(np.mean([row["covered"] for row in rows])),
        "lower_tail_miss": float(np.mean([row["lower_miss"] for row in rows])),
        "upper_tail_miss": float(np.mean([row["upper_miss"] for row in rows])),
        "mean_absolute_bias": float(np.mean(np.abs(biases))),
        "minimum_width": float(widths.min()),
        "median_width": median_width,
        "p95_width": float(np.percentile(widths, 95)),
        "p95_median_width_ratio": float(np.percentile(widths, 95) / median_width),
    }


def checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    reward = summary["families"]["reward"]
    policy = summary["families"]["policy"]
    thresholds = FROZEN_THRESHOLDS
    values = {
        "usable_panel_rate": summary["usable_panel_rate"],
        "reward_coverage": reward["coverage"],
        "policy_coverage": policy["coverage"],
        "reward_lower_tail_miss": reward["lower_tail_miss"],
        "reward_upper_tail_miss": reward["upper_tail_miss"],
        "policy_lower_tail_miss": policy["lower_tail_miss"],
        "policy_upper_tail_miss": policy["upper_tail_miss"],
        "reward_absolute_bias": reward["mean_absolute_bias"],
        "policy_absolute_bias": policy["mean_absolute_bias"],
        "reward_width_stability": reward["p95_median_width_ratio"],
        "policy_width_stability": policy["p95_median_width_ratio"],
    }
    limits = {
        "usable_panel_rate": (">=", thresholds["minimum_usable_panel_rate"]),
        "reward_coverage": (">=", thresholds["minimum_reward_coverage"]),
        "policy_coverage": (">=", thresholds["minimum_policy_coverage"]),
        "reward_lower_tail_miss": ("<=", thresholds["maximum_tail_miss"]),
        "reward_upper_tail_miss": ("<=", thresholds["maximum_tail_miss"]),
        "policy_lower_tail_miss": ("<=", thresholds["maximum_tail_miss"]),
        "policy_upper_tail_miss": ("<=", thresholds["maximum_tail_miss"]),
        "reward_absolute_bias": ("<=", thresholds["maximum_reward_absolute_bias"]),
        "policy_absolute_bias": ("<=", thresholds["maximum_policy_absolute_bias"]),
        "reward_width_stability": (
            "<=",
            thresholds["maximum_p95_median_width_ratio"],
        ),
        "policy_width_stability": (
            "<=",
            thresholds["maximum_p95_median_width_ratio"],
        ),
    }
    return [
        {
            "name": name,
            "value": value,
            "operator": limits[name][0],
            "threshold": limits[name][1],
            "passed": bool(
                value >= limits[name][1] if limits[name][0] == ">=" else value <= limits[name][1]
            ),
        }
        for name, value in values.items()
    ]


def environment() -> dict[str, Any]:
    packages = ("econirl", "jax", "jaxlib", "numpy", "optax")
    return {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    n_panels = 2 if args.smoke else FULL_PANELS
    draws = 5 if args.smoke else FULL_DRAWS
    records = run_panels(args.checkpoint, n_panels, draws, args.workers)
    usable = [record for record in records if record.get("error") is None]
    summary = {
        "n_requested": n_panels,
        "n_usable": len(usable),
        "usable_panel_rate": len(usable) / n_panels,
        "bootstrap_draws_requested": n_panels * draws,
        "bootstrap_draws_successful": sum(record["successful_draws"] for record in usable),
        "families": {family: summarize_family(records, family) for family in ("reward", "policy")},
    }
    gate_results = checks(summary)
    receipt = {
        "estimator": "AIRL",
        "status": "ready" if all(gate["passed"] for gate in gate_results) else "failed",
        "paper_replication": False,
        "mode": "smoke" if args.smoke else "full",
        "design": {
            "n_panels": n_panels,
            "bootstrap_draws_per_panel": draws,
            "bootstrap_unit": "individual_trajectory",
            "individuals_per_panel": 160,
            "periods_per_individual": 40,
            "states": 6,
            "actions": 2,
            "reward_target": "centered state-only reward and induced policy",
        },
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "summary": summary,
        "checks": gate_results,
        "records": records,
        "environment": environment(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0 if receipt["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
