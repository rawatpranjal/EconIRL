#!/usr/bin/env python3
"""Repeated-panel and training-seed stability checks for Neural MCE-IRL."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "deep_mce_irl_ready.json"
PRIMARY_CELL = "deep_mce_neural_reward"
FULL_PANELS = 100
FULL_TRAINING_SEEDS = 3

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from validation.known_truth import (  # noqa: E402
    build_known_truth_dgp,
    counterfactual_metrics,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    solve_counterfactual_oracle,
    solve_known_truth,
)


def public_counterfactual_metrics(dgp, run) -> dict[str, dict[str, float]]:
    """Evaluate all three public counterfactual modes against known truth."""
    adapter = run.fitted_estimator
    model = adapter.model_
    if model is None:
        raise RuntimeError("Neural MCE adapter did not retain its fitted model")
    true_reward = np.asarray(dgp.homogeneous_reward, dtype=np.float64)
    transitions = np.asarray(dgp.transitions, dtype=np.float64)
    baseline = solve_known_truth(dgp)

    def solve(reward: np.ndarray, changed_transitions: np.ndarray):
        result = value_iteration(
            SoftBellmanOperator(dgp.problem, changed_transitions),
            reward,
            tol=1e-8,
            max_iter=10_000,
        )
        if not result.converged:
            raise RuntimeError("counterfactual oracle did not converge")
        return result

    progress = np.asarray(dgp.state_features[:, 0], dtype=np.float64)
    reward_delta = np.zeros_like(true_reward)
    reward_delta[:, 1] = 0.25 * progress
    reward_truth = true_reward + reward_delta
    reward_oracle = solve(reward_truth, transitions)
    reward_result = model.counterfactual(reward_delta=reward_delta)

    transition_oracle = solve_counterfactual_oracle(dgp, "type_b")
    changed_transitions = np.asarray(transition_oracle.counterfactual.transitions)
    transition_result = model.counterfactual(transitions=changed_transitions)

    action_mask = np.ones_like(true_reward, dtype=bool)
    action_mask[:, 1] = False
    action_truth = np.where(action_mask, true_reward, -1e9)
    action_oracle = solve(action_truth, transitions)
    action_result = model.counterfactual(action_mask=action_mask)

    cases = {
        "type_a": (reward_oracle, reward_result, reward_truth, transitions),
        "type_b": (
            transition_oracle.counterfactual_solution,
            transition_result,
            true_reward,
            changed_transitions,
        ),
        "type_c": (action_oracle, action_result, action_truth, transitions),
    }
    metrics: dict[str, dict[str, float]] = {}
    for kind, (oracle, result, reward, changed_transitions) in cases.items():
        recovery = counterfactual_metrics(
            oracle_policy=oracle.policy,
            oracle_value=oracle.V,
            estimated_policy=result.counterfactual_policy,
            reward=reward,
            transitions=changed_transitions,
            discount_factor=dgp.problem.discount_factor,
            initial_distribution=dgp.initial_distribution,
            scale_parameter=dgp.problem.scale_parameter,
        )
        oracle_policy_tv = float(
            np.mean(
                0.5
                * np.abs(
                    np.asarray(oracle.policy, dtype=np.float64)
                    - np.asarray(baseline.policy, dtype=np.float64)
                ).sum(axis=1)
            )
        )
        metrics[kind] = {
            "regret": float(recovery.regret),
            "policy_tv": float(recovery.policy.tv),
            "oracle_policy_tv": oracle_policy_tv,
        }
    return metrics


def package_versions() -> dict[str, str]:
    names = ("econirl", "jax", "jaxlib", "equinox", "optax", "numpy")
    return {name: importlib.metadata.version(name) for name in names}


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def load_checkpoint(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[(int(record["panel_index"]), int(record["training_seed"]))] = record
    return records


def fit_once(panel_index: int, training_seed: int, *, smoke: bool) -> dict[str, Any]:
    cell = get_cell(PRIMARY_CELL)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation = replace(
        cell.simulation_config,
        n_individuals=80 if smoke else 600,
        n_periods=20 if smoke else 60,
        seed=71_000 + panel_index,
        show_progress=False,
    )
    panel = simulate_known_truth_panel(dgp, simulation)
    panel.metadata["neural_training_seed"] = 83_000 + training_seed
    started = time.perf_counter()
    try:
        run = run_estimator(
            "MCE-IRL Deep",
            dgp,
            panel,
            smoke=smoke,
            verbose=False,
            enforce_gates=False,
        )
        counterfactuals = public_counterfactual_metrics(dgp, run)
        return {
            "panel_index": panel_index,
            "panel_seed": simulation.seed,
            "training_seed": training_seed,
            "network_seed": panel.metadata["neural_training_seed"],
            "converged": bool(run.summary.converged),
            "occupancy_residual": float(run.summary.metadata["occupancy_moment_residual"]),
            "reward_nrmse": float(run.metrics["reward_normalized_rmse"]),
            "policy_tv": float(run.metrics["policy"].tv),
            "value_nrmse": float(run.metrics["value_normalized_rmse"]),
            "q_nrmse": float(run.metrics["q_normalized_rmse"]),
            "policy": np.asarray(run.summary.policy, dtype=float).tolist(),
            "counterfactual_regret": {
                kind: float(counterfactuals[kind]["regret"])
                for kind in ("type_a", "type_b", "type_c")
            },
            "counterfactual_policy_tv": {
                kind: float(counterfactuals[kind]["policy_tv"])
                for kind in ("type_a", "type_b", "type_c")
            },
            "oracle_counterfactual_policy_tv": {
                kind: float(counterfactuals[kind]["oracle_policy_tv"])
                for kind in ("type_a", "type_b", "type_c")
            },
            "runtime_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are evidence
        return {
            "panel_index": panel_index,
            "panel_seed": simulation.seed,
            "training_seed": training_seed,
            "network_seed": panel.metadata["neural_training_seed"],
            "converged": False,
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_records(
    checkpoint: Path,
    n_panels: int,
    n_training_seeds: int,
    *,
    smoke: bool,
    quiet: bool,
) -> list[dict[str, Any]]:
    completed = load_checkpoint(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = n_panels * n_training_seeds
    for panel_index in range(n_panels):
        for training_seed in range(n_training_seeds):
            key = (panel_index, training_seed)
            if key in completed:
                records.append(completed[key])
                continue
            record = fit_once(panel_index, training_seed, smoke=smoke)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
            if not quiet:
                print(
                    f"{len(records)}/{total}: panel={panel_index} "
                    f"training={training_seed} "
                    f"{record.get('error') or record['policy_tv']:.6}"
                )
    return records


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("error") is None]
    metrics: dict[str, Any] = {
        "n_requested": len(records),
        "n_successful": len(successful),
        "n_converged": sum(bool(record["converged"]) for record in successful),
        "convergence_rate": (
            sum(bool(record["converged"]) for record in successful) / len(records)
            if records
            else 0.0
        ),
    }
    for key in ("reward_nrmse", "policy_tv", "value_nrmse", "q_nrmse"):
        values = [float(record[key]) for record in successful]
        metrics[key] = {
            "median": float(np.median(values)) if values else None,
            "p95": percentile(values, 95) if values else None,
            "maximum": max(values) if values else None,
        }

    within_panel: list[float] = []
    grouped: dict[int, list[np.ndarray]] = {}
    for record in successful:
        grouped.setdefault(int(record["panel_index"]), []).append(
            np.asarray(record["policy"], dtype=float)
        )
    for policies in grouped.values():
        for left in range(len(policies)):
            for right in range(left + 1, len(policies)):
                within_panel.append(
                    float(np.mean(np.abs(policies[left] - policies[right]).sum(axis=1) / 2.0))
                )
    metrics["training_seed_policy_tv"] = {
        "median": float(np.median(within_panel)) if within_panel else 0.0,
        "p95": percentile(within_panel, 95) if within_panel else 0.0,
    }
    metrics["counterfactual_regret"] = {
        kind: {
            "median": float(
                np.median([record["counterfactual_regret"][kind] for record in successful])
            )
            if successful
            else None,
            "p95": percentile(
                [record["counterfactual_regret"][kind] for record in successful],
                95,
            )
            if successful
            else None,
        }
        for kind in ("type_a", "type_b", "type_c")
    }
    metrics["oracle_counterfactual_policy_tv"] = {
        kind: {
            "minimum": min(record["oracle_counterfactual_policy_tv"][kind] for record in successful)
            if successful
            else None,
            "median": float(
                np.median(
                    [record["oracle_counterfactual_policy_tv"][kind] for record in successful]
                )
            )
            if successful
            else None,
        }
        for kind in ("type_a", "type_b", "type_c")
    }
    return metrics


def readiness_checks(summary: dict[str, Any], *, final_run: bool) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "all_fits_successful",
            "passed": summary["n_successful"] == summary["n_requested"],
        },
        {
            "name": "all_fits_converged",
            "passed": summary["n_converged"] == summary["n_requested"],
        },
        {
            "name": "median_reward_nrmse",
            "passed": summary["reward_nrmse"]["median"] <= 0.15,
        },
        {
            "name": "p95_reward_nrmse",
            "passed": summary["reward_nrmse"]["p95"] <= 0.30,
        },
        {
            "name": "median_policy_tv",
            "passed": summary["policy_tv"]["median"] <= 0.05,
        },
        {
            "name": "p95_policy_tv",
            "passed": summary["policy_tv"]["p95"] <= 0.10,
        },
        {
            "name": "median_training_seed_policy_tv",
            "passed": summary["training_seed_policy_tv"]["median"] <= 0.02,
        },
        {
            "name": "p95_training_seed_policy_tv",
            "passed": summary["training_seed_policy_tv"]["p95"] <= 0.05,
        },
    ]
    for kind in ("type_a", "type_b", "type_c"):
        checks.extend(
            [
                {
                    "name": f"{kind}_median_regret",
                    "passed": summary["counterfactual_regret"][kind]["median"] <= 0.08,
                },
                {
                    "name": f"{kind}_p95_regret",
                    "passed": summary["counterfactual_regret"][kind]["p95"] <= 0.08,
                },
                {
                    "name": f"{kind}_oracle_policy_changes",
                    "passed": summary["oracle_counterfactual_policy_tv"][kind]["minimum"] >= 1e-4,
                },
            ]
        )
    checks.append({"name": "full_configuration", "passed": final_run})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--n-panels", type=int, default=FULL_PANELS)
    parser.add_argument("--n-training-seeds", type=int, default=FULL_TRAINING_SEEDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    n_panels = min(args.n_panels, 2) if args.smoke else args.n_panels
    n_training_seeds = min(args.n_training_seeds, 2) if args.smoke else args.n_training_seeds
    output = args.output or (
        Path("/tmp/deep_mce_irl_ready_smoke.json") if args.smoke else DEFAULT_OUTPUT
    )
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    records = run_records(
        checkpoint,
        n_panels,
        n_training_seeds,
        smoke=args.smoke,
        quiet=args.quiet,
    )
    summary = aggregate(records)
    final_run = (
        not args.smoke and n_panels >= FULL_PANELS and n_training_seeds >= FULL_TRAINING_SEEDS
    )
    checks = readiness_checks(summary, final_run=final_run)
    status = "ready" if all(check["passed"] for check in checks) else "incomplete"
    payload = {
        "estimator": "Neural MCE-IRL",
        "target": "anchored reward map and induced behavior",
        "inference": "not supported; repeated-panel and training-seed stability",
        "status": status,
        "configuration": {
            "cell": PRIMARY_CELL,
            "n_panels": n_panels,
            "n_training_seeds": n_training_seeds,
            "n_fits": n_panels * n_training_seeds,
            "execution": "CPU",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha": git_sha(),
            "packages": package_versions(),
        },
        "summary": summary,
        "checks": checks,
        "checkpoint": str(checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    display_output = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(f"wrote {display_output}")
    print(f"status: {status}")
    return 0 if (args.smoke or status == "ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
