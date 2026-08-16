#!/usr/bin/env python3
"""Calibrate Neural MCE-IRL whole-trajectory bootstrap intervals."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "deep_mce_irl_bootstrap_calibration.json"
DEFAULT_CHECKPOINT = ROOT / "validation" / "results" / "deep_mce_irl_bootstrap_calibration.jsonl"
FULL_PANELS = 50
FULL_DRAWS = 99
TARGET_STATES = (0, 1, 2)
MIN_USABLE_RATE = 0.95
MIN_DRAW_RATE = 0.95
MIN_COVERAGE = 0.86
MAX_TAIL_MISS = 0.10
MAX_WIDTH_RATIO = 2.0
MIN_ORACLE_POLICY_TV = 1e-4

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl import MCEIRLNeural  # noqa: E402
from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.core.types import DDCProblem, Panel, Trajectory  # noqa: E402


def transition_tensor() -> np.ndarray:
    """Return the frozen (actions, states, next_states) transition tensor."""
    transitions = np.zeros((2, 3, 3), dtype=np.float32)
    for state in range(3):
        transitions[:, state, state] = 1.0
    return transitions


def changed_transition_tensor() -> np.ndarray:
    """Return a less persistent action-one kernel for the transition exercise."""
    changed = transition_tensor().copy()
    for state in range(3):
        previous = max(state - 1, 0)
        changed[1, state] = 0.0
        changed[1, state, state] += 0.45
        changed[1, state, previous] += 0.55
    return cast(np.ndarray, changed)


def true_reward() -> np.ndarray:
    """Anchored nonlinear reward, with action zero fixed to zero."""
    reward = np.zeros((3, 2), dtype=np.float32)
    reward[:, 1] = np.array([0.75, -0.55, 0.90], dtype=np.float32)
    return reward


def encode_states(states: jnp.ndarray) -> jnp.ndarray:
    """Use a fixed one-hot state encoding for the calibration problem."""
    return jnn.one_hot(jnp.asarray(states, dtype=jnp.int32), 3)


def solve(
    reward: np.ndarray,
    transitions: np.ndarray,
    action_mask: np.ndarray | None = None,
) -> Any:
    """Solve one known reward and environment exactly."""
    mask = np.ones_like(reward, dtype=bool) if action_mask is None else action_mask
    result = value_iteration(
        SoftBellmanOperator(
            DDCProblem(
                num_states=3,
                num_actions=2,
                discount_factor=0.5,
                scale_parameter=1.0,
            ),
            jnp.asarray(transitions),
        ),
        jnp.asarray(np.where(mask, reward, -1e9)),
        tol=1e-10,
        max_iter=10_000,
    )
    if not result.converged:
        raise RuntimeError("known-truth Bellman solve did not converge")
    return result


def simulate_panel(panel_index: int, *, smoke: bool) -> Panel:
    """Simulate one panel from the frozen known-truth policy."""
    rng = np.random.default_rng(41_000 + panel_index)
    transitions = transition_tensor()
    policy = np.asarray(solve(true_reward(), transitions).policy)
    n_individuals = 90 if smoke else 300
    n_periods = 1
    trajectories = []
    for individual in range(n_individuals):
        state = individual % 3
        states: list[int] = []
        actions: list[int] = []
        next_states: list[int] = []
        for _ in range(n_periods):
            action = int(rng.choice(2, p=policy[state]))
            next_state = int(rng.choice(3, p=transitions[action, state]))
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
    return Panel(trajectories)


def model_for(panel_index: int, n_draws: int, *, smoke: bool) -> MCEIRLNeural:
    """Return the frozen estimator configuration for one calibration panel."""
    return MCEIRLNeural(
        n_states=3,
        n_actions=2,
        discount=0.5,
        reward_hidden_dim=8,
        reward_num_layers=1,
        max_epochs=35,
        lr=0.10,
        occupancy_tol=0.08 if smoke else 0.03,
        patience=60,
        improvement_tol=0.0,
        inner_solver="value",
        inner_tol=1e-8,
        inner_max_iter=2_000,
        state_encoder=encode_states,
        state_dim=3,
        anchor_action=0,
        se_method="bootstrap",
        n_bootstrap=n_draws,
        se_seed=63_000 + panel_index,
        seed=52,
    )


def interval_record(interval: tuple[float, float], truth: float, target: str) -> dict[str, Any]:
    """Record coverage, tail direction, and width for one scalar interval."""
    lower, upper = (float(interval[0]), float(interval[1]))
    return {
        "target": target,
        "lower": lower,
        "upper": upper,
        "truth": float(truth),
        "covered": bool(lower <= truth <= upper),
        "lower_tail_miss": bool(truth < lower),
        "upper_tail_miss": bool(truth > upper),
        "width": upper - lower,
    }


def counterfactual_design() -> dict[str, dict[str, Any]]:
    """Return the three frozen counterfactual interventions and their truths."""
    reward = true_reward()
    transitions = transition_tensor()
    baseline = solve(reward, transitions)
    reward_delta = np.zeros_like(reward)
    reward_delta[:, 1] = np.array([0.10, 0.20, 0.30])
    changed_transitions = changed_transition_tensor()
    action_mask = np.ones_like(reward, dtype=bool)
    action_mask[2, 1] = False
    designs = {
        "reward_counterfactual": {
            "kwargs": {"reward_delta": reward_delta},
            "solution": solve(reward + reward_delta, transitions),
        },
        "transition_counterfactual": {
            "kwargs": {"transitions": changed_transitions},
            "solution": solve(reward, changed_transitions),
        },
        "action_mask_counterfactual": {
            "kwargs": {"action_mask": action_mask},
            "solution": solve(reward, transitions, action_mask),
        },
    }
    for design in designs.values():
        solution = design["solution"]
        design["truth"] = {
            "mean_policy_tv": float(
                0.5
                * np.abs(np.asarray(solution.policy) - np.asarray(baseline.policy))
                .sum(axis=1)
                .mean()
            ),
            "mean_value_change": float(np.mean(np.asarray(solution.V - baseline.V))),
        }
    return designs


def fit_panel(panel_index: int, n_draws: int, *, smoke: bool) -> dict[str, Any]:
    """Fit one panel and evaluate all frozen interval targets."""
    started = time.perf_counter()
    panel = simulate_panel(panel_index, smoke=smoke)
    model = model_for(panel_index, n_draws, smoke=smoke)
    try:
        model.fit(panel, transitions=transition_tensor())
        assert model.bootstrap_ is not None
        intervals = model.conf_int()
        truth_solution = solve(true_reward(), transition_tensor())
        families: dict[str, list[dict[str, Any]]] = {"reward": [], "policy": []}
        for state in TARGET_STATES:
            families["reward"].append(
                interval_record(
                    intervals[f"reward[s={state},a=1]"],
                    float(true_reward()[state, 1]),
                    f"state_{state}",
                )
            )
            families["policy"].append(
                interval_record(
                    intervals[f"policy[s={state},a=1]"],
                    float(truth_solution.policy[state, 1]),
                    f"state_{state}",
                )
            )
        for family, design in counterfactual_design().items():
            result = model.counterfactual(**design["kwargs"])
            receipt = result.metadata["bootstrap_intervals"]
            families[family] = [
                interval_record(receipt[target], truth, target)
                for target, truth in design["truth"].items()
            ]
        required_draws = math.ceil(MIN_DRAW_RATE * n_draws)
        return {
            "panel_index": panel_index,
            "panel_seed": 41_000 + panel_index,
            "bootstrap_seed": 63_000 + panel_index,
            "point_converged": model.converged_ is True,
            "termination_reason": model.termination_reason_,
            "n_requested": model.bootstrap_.n_requested,
            "n_successful": model.bootstrap_.n_successful,
            "usable": bool(model.converged_ and model.bootstrap_.n_successful >= required_draws),
            "families": families,
            "failures": list(model.bootstrap_.failures),
            "runtime_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are calibration evidence
        return {
            "panel_index": panel_index,
            "panel_seed": 41_000 + panel_index,
            "bootstrap_seed": 63_000 + panel_index,
            "point_converged": False,
            "usable": False,
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_checkpoint(path: Path, *, n_draws: int) -> dict[int, dict[str, Any]]:
    """Load compatible records, rejecting smoke or stale draw counts."""
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if int(record.get("n_requested", -1)) != n_draws:
                continue
            records[int(record["panel_index"])] = record
    return records


def run_panels(
    n_panels: int,
    n_draws: int,
    *,
    smoke: bool,
    checkpoint: Path | None,
    quiet: bool,
    jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run or resume the panel-level calibration sweep."""
    completed = load_checkpoint(checkpoint, n_draws=n_draws) if checkpoint is not None else {}
    records = []
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        compact_path = checkpoint.with_suffix(checkpoint.suffix + ".compact")
        compact_path.write_text(
            "".join(
                json.dumps(completed[index], allow_nan=False) + "\n" for index in sorted(completed)
            ),
            encoding="utf-8",
        )
        compact_path.replace(checkpoint)
    missing = [panel_index for panel_index in range(n_panels) if panel_index not in completed]
    if jobs < 1:
        raise ValueError("jobs must be positive")
    new_by_panel: dict[int, dict[str, Any]] = {}

    def persist(record: dict[str, Any]) -> None:
        panel_index = int(record["panel_index"])
        new_by_panel[panel_index] = record
        if checkpoint is not None:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        if not quiet:
            print(
                f"panel {panel_index + 1}/{n_panels}: "
                f"usable={record['usable']} "
                f"draws={record.get('n_successful', 0)}/{n_draws}",
                flush=True,
            )

    if jobs == 1:
        for panel_index in missing:
            persist(fit_panel(panel_index, n_draws, smoke=smoke))
            jax.clear_caches()
            gc.collect()
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
            futures = {
                executor.submit(_fit_panel_job, (panel_index, n_draws, smoke)): panel_index
                for panel_index in missing
            }
            for future in as_completed(futures):
                persist(future.result())
    for panel_index in range(n_panels):
        if panel_index in completed:
            records.append(completed[panel_index])
            continue
        record = new_by_panel[panel_index]
        records.append(record)
    return records


def _fit_panel_job(args: tuple[int, int, bool]) -> dict[str, Any]:
    """Spawn-safe entry point for one independent calibration panel."""
    panel_index, n_draws, smoke = args
    try:
        return fit_panel(panel_index, n_draws, smoke=smoke)
    finally:
        jax.clear_caches()
        gc.collect()


def family_summary(records: list[dict[str, Any]], family: str) -> dict[str, Any]:
    """Aggregate coverage, tail misses, and interval widths for one family."""
    intervals = [
        interval
        for record in records
        if record.get("usable")
        for interval in record["families"][family]
    ]
    widths = np.asarray([item["width"] for item in intervals], dtype=float)
    median_width = float(np.median(widths)) if widths.size else None
    p95_width = float(np.percentile(widths, 95)) if widths.size else None
    per_target_widths = {}
    for target in sorted({item["target"] for item in intervals}):
        target_widths = np.asarray(
            [item["width"] for item in intervals if item["target"] == target],
            dtype=float,
        )
        target_median = float(np.median(target_widths))
        target_p95 = float(np.percentile(target_widths, 95))
        per_target_widths[target] = {
            "median_width": target_median,
            "p95_width": target_p95,
            "p95_median_width_ratio": (target_p95 / target_median if target_median > 0 else None),
        }
    target_ratios = [
        item["p95_median_width_ratio"]
        for item in per_target_widths.values()
        if item["p95_median_width_ratio"] is not None
    ]
    return {
        "n_intervals": len(intervals),
        "coverage": float(np.mean([item["covered"] for item in intervals])) if intervals else 0.0,
        "lower_tail_miss": (
            float(np.mean([item["lower_tail_miss"] for item in intervals])) if intervals else 1.0
        ),
        "upper_tail_miss": (
            float(np.mean([item["upper_tail_miss"] for item in intervals])) if intervals else 1.0
        ),
        "minimum_width": float(np.min(widths)) if widths.size else None,
        "median_width": median_width,
        "p95_width": p95_width,
        "p95_median_width_ratio": max(target_ratios) if target_ratios else None,
        "per_target_widths": per_target_widths,
    }


def digest(records: list[dict[str, Any]]) -> str:
    """Hash deterministic statistical output, excluding runtime."""
    stable = [
        {key: value for key, value in row.items() if key != "runtime_seconds"} for row in records
    ]
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def aggregate(
    records: list[dict[str, Any]],
    *,
    n_panels: int,
    n_draws: int,
    smoke_reproducible: bool,
) -> dict[str, Any]:
    """Apply the frozen calibration gates without changing thresholds."""
    usable = [record for record in records if record.get("usable")]
    families = {
        name: family_summary(records, name)
        for name in (
            "reward",
            "policy",
            "reward_counterfactual",
            "transition_counterfactual",
            "action_mask_counterfactual",
        )
    }
    required_usable = math.ceil(MIN_USABLE_RATE * n_panels)
    required_draws = math.ceil(MIN_DRAW_RATE * n_draws)
    checks = [
        {
            "name": "usable_panels",
            "value": len(usable),
            "threshold": required_usable,
            "passed": len(usable) >= required_usable,
        },
        {
            "name": "minimum_successful_draws",
            "value": min((record["n_successful"] for record in usable), default=0),
            "threshold": required_draws,
            "passed": bool(usable)
            and min(record["n_successful"] for record in usable) >= required_draws,
        },
        {
            "name": "seeded_smoke_reproducibility",
            "value": smoke_reproducible,
            "threshold": True,
            "passed": smoke_reproducible,
        },
    ]
    oracle_targets = {name: design["truth"] for name, design in counterfactual_design().items()}
    checks.extend(
        {
            "name": f"{name}_oracle_policy_changes",
            "value": targets["mean_policy_tv"],
            "threshold": MIN_ORACLE_POLICY_TV,
            "passed": targets["mean_policy_tv"] >= MIN_ORACLE_POLICY_TV,
        }
        for name, targets in oracle_targets.items()
    )
    for name, summary in families.items():
        checks.extend(
            [
                {
                    "name": f"{name}_coverage",
                    "value": summary["coverage"],
                    "threshold": [MIN_COVERAGE, 1.0],
                    "passed": MIN_COVERAGE <= summary["coverage"] <= 1.0,
                },
                {
                    "name": f"{name}_lower_tail_miss",
                    "value": summary["lower_tail_miss"],
                    "threshold": MAX_TAIL_MISS,
                    "passed": summary["lower_tail_miss"] <= MAX_TAIL_MISS,
                },
                {
                    "name": f"{name}_upper_tail_miss",
                    "value": summary["upper_tail_miss"],
                    "threshold": MAX_TAIL_MISS,
                    "passed": summary["upper_tail_miss"] <= MAX_TAIL_MISS,
                },
                {
                    "name": f"{name}_positive_widths",
                    "value": summary["minimum_width"],
                    "threshold": 0.0,
                    "passed": summary["minimum_width"] is not None
                    and summary["minimum_width"] > 0.0,
                },
                {
                    "name": f"{name}_width_ratio",
                    "value": summary["p95_median_width_ratio"],
                    "threshold": MAX_WIDTH_RATIO,
                    "passed": summary["p95_median_width_ratio"] is not None
                    and summary["p95_median_width_ratio"] <= MAX_WIDTH_RATIO,
                },
            ]
        )
    return {
        "status": "ready" if all(check["passed"] for check in checks) else "not_ready",
        "paper_replication": False,
        "design": {
            "n_panels": n_panels,
            "bootstrap_draws_per_panel": n_draws,
            "bootstrap_unit": "individual_trajectory",
            "individuals_per_panel": 90 if n_draws == 5 else 300,
            "periods_per_individual": 1,
            "discount_factor": 0.5,
            "transition_design": "action-independent self transitions",
            "target_states": list(TARGET_STATES),
            "target_action": 1,
            "anchor_action": 0,
            "counterfactual_families": ["reward", "transitions", "action_mask"],
        },
        "frozen_thresholds": {
            "minimum_usable_panel_rate": MIN_USABLE_RATE,
            "minimum_successful_draw_rate": MIN_DRAW_RATE,
            "family_coverage": [MIN_COVERAGE, 1.0],
            "maximum_tail_miss": MAX_TAIL_MISS,
            "minimum_interval_width": 0.0,
            "maximum_p95_median_width_ratio": MAX_WIDTH_RATIO,
            "seeded_smoke_reproducibility": "exact",
            "minimum_oracle_policy_tv": MIN_ORACLE_POLICY_TV,
        },
        "summary": {
            "n_requested": len(records),
            "n_usable": len(usable),
            "usable_rate": len(usable) / len(records) if records else 0.0,
            "bootstrap_draws_requested": len(records) * n_draws,
            "bootstrap_draws_successful": sum(
                int(record.get("n_successful", 0)) for record in records
            ),
            "bootstrap_draw_success_rate": (
                sum(int(record.get("n_successful", 0)) for record in records)
                / (len(records) * n_draws)
                if records
                else 0.0
            ),
            "families": families,
            "errors": [record["error"] for record in records if record.get("error")],
            "oracle_counterfactual_targets": oracle_targets,
        },
        "checks": checks,
    }


def metadata() -> dict[str, Any]:
    """Return reproducibility metadata for the compact receipt."""
    packages = ("econirl", "jax", "jaxlib", "equinox", "optax", "numpy")
    return {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    n_panels = 2 if args.smoke else FULL_PANELS
    n_draws = 5 if args.smoke else FULL_DRAWS
    records = run_panels(
        n_panels,
        n_draws,
        smoke=args.smoke,
        checkpoint=args.checkpoint,
        quiet=args.quiet,
        jobs=args.jobs,
    )
    first_probe = run_panels(1, 2, smoke=True, checkpoint=None, quiet=True)
    second_probe = run_panels(1, 2, smoke=True, checkpoint=None, quiet=True)
    result = aggregate(
        records,
        n_panels=n_panels,
        n_draws=n_draws,
        smoke_reproducible=digest(first_probe) == digest(second_probe),
    )
    if args.smoke:
        smoke_passed = bool(
            result["checks"][2]["passed"]
            and all(record.get("error") is None for record in records)
            and all(record.get("n_successful") == n_draws for record in records)
        )
        result["status"] = "incomplete" if smoke_passed else "smoke_failed"
        result["full_gates_evaluated"] = False
    else:
        result["full_gates_evaluated"] = True
    result["mode"] = "smoke" if args.smoke else "full"
    result["metadata"] = metadata()
    result["record_digest"] = digest(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.quiet:
        try:
            display_path = args.output.relative_to(ROOT)
        except ValueError:
            display_path = args.output
        minimum_successful = result["checks"][1]["value"]
        print(f"wrote {display_path}")
        print(f"status: {result['status']}")
        print(f"usable panels: {result['summary']['n_usable']}/{result['summary']['n_requested']}")
        print(f"minimum successful draws: {minimum_successful}/{n_draws}")
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] not in ("ready", "incomplete"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
