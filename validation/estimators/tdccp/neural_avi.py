#!/usr/bin/env python3
"""Grade neural AVI and its locally robust correction over paired panels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from econirl import TDCCP  # noqa: E402
from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.environments import ArrayMDP  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402
from validation.estimators.nfxp.ready import _strict_json  # noqa: E402
from validation.estimators.tdccp.bootstrap_calibration import build_problem  # noqa: E402

DEFAULT_OUTPUT = ROOT / "validation" / "results" / "tdccp_neural_avi.json"
FINAL_REPLICATIONS = 30
N_INDIVIDUALS = 400
N_PERIODS = 40
BASE_SEED = 151_000


@lru_cache(maxsize=1)
def build_stationary_problem() -> tuple[ArrayMDP, np.ndarray, Any, np.ndarray, float]:
    """Use the stationary sampling law required by the backward recursion."""
    env, transitions, reward, truth = build_problem()
    solved = value_iteration(
        SoftBellmanOperator(env.problem_spec, transitions),
        env.true_reward_matrix,
        tol=1e-12,
        max_iter=20_000,
    )
    policy_kernel = np.einsum("sa,asj->sj", np.asarray(solved.policy), transitions)
    stationary = np.full(env.num_states, 1.0 / env.num_states, dtype=np.float64)
    for _ in range(20_000):
        updated = stationary @ policy_kernel
        if np.max(np.abs(updated - stationary)) <= 1e-14:
            stationary = updated
            break
        stationary = updated
    residual = float(np.max(np.abs(stationary @ policy_kernel - stationary)))
    stationary_env = ArrayMDP(
        transitions,
        np.asarray(env.feature_matrix),
        theta=truth,
        discount_factor=float(env.problem_spec.discount_factor),
        scale_parameter=float(env.problem_spec.scale_parameter),
        parameter_names=env.parameter_names,
        initial_distribution=stationary,
        seed=BASE_SEED,
    )
    return stationary_env, transitions, reward, truth, residual


def _fit(panel: Any, seed: int, *, locally_robust: bool) -> tuple[TDCCP, float]:
    env, transitions, reward, _truth, _residual = build_stationary_problem()
    started = time.perf_counter()
    model = TDCCP(
        n_states=env.num_states,
        n_actions=env.num_actions,
        discount=float(env.problem_spec.discount_factor),
        utility=reward,
        se_method="robust" if locally_robust else "asymptotic",
        seed=seed,
        method="neural",
        hidden_dim=32,
        num_hidden_layers=2,
        avi_iterations=10,
        avi_early_stop_tol=0.001,
        epochs_per_avi=20,
        learning_rate=0.003,
        batch_size=8192,
        ccp_method="logit",
        ccp_poly_degree=2,
        cross_fitting=locally_robust,
        robust_se=locally_robust,
        outer_max_iter=500,
        outer_tol=1e-7,
    )
    model.fit(panel, transitions=transitions)
    return model, time.perf_counter() - started


def fit_pair(rep: int) -> dict[str, Any]:
    """Fit plug-in and Algorithm 2 neural AVI on the same panel."""
    env, _transitions, _reward, truth, _residual = build_stationary_problem()
    seed = BASE_SEED + rep
    panel = simulate_panel(
        env,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=seed,
    )
    try:
        plugin, plugin_time = _fit(panel, seed, locally_robust=False)
        robust, robust_time = _fit(panel, seed, locally_robust=True)
        plugin_estimate = np.asarray(plugin.coef_, dtype=float)
        robust_estimate = np.asarray(robust.coef_, dtype=float)
        return {
            "rep": rep,
            "seed": seed,
            "truth": truth.tolist(),
            "plugin": {
                "parameters": plugin_estimate.tolist(),
                "relative_error": (np.abs(plugin_estimate - truth) / np.abs(truth)).tolist(),
                "converged": bool(plugin.converged_),
                "runtime_seconds": plugin_time,
            },
            "locally_robust": {
                "parameters": robust_estimate.tolist(),
                "relative_error": (np.abs(robust_estimate - truth) / np.abs(truth)).tolist(),
                "converged": bool(robust.converged_),
                "runtime_seconds": robust_time,
                "moment_norm_max": float(
                    robust._result.metadata["paper_inference"]["moment_norm_max"]
                ),
            },
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are validation evidence
        return {
            "rep": rep,
            "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def read_records(paths: list[Path]) -> dict[int, dict[str, Any]]:
    """Read and de-duplicate one or more resumable checkpoints."""
    records: dict[int, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[int(record["rep"])] = record
    return records


def summarize(records: list[dict[str, Any]], *, final_run: bool) -> dict[str, Any]:
    usable = [
        record
        for record in records
        if record.get("error") is None
        and record["plugin"]["converged"]
        and record["locally_robust"]["converged"]
        and np.all(np.isfinite(record["plugin"]["parameters"]))
        and np.all(np.isfinite(record["locally_robust"]["parameters"]))
    ]
    truth = np.asarray(build_stationary_problem()[3], dtype=float)
    plugin = np.asarray([record["plugin"]["parameters"] for record in usable])
    robust = np.asarray([record["locally_robust"]["parameters"] for record in usable])
    plugin_bias = np.abs(plugin.mean(axis=0) - truth) if len(usable) else np.full(2, np.inf)
    robust_bias = np.abs(robust.mean(axis=0) - truth) if len(usable) else np.full(2, np.inf)
    robust_relative_error = (
        np.abs(robust - truth[None, :]) / np.abs(truth[None, :])
        if len(usable)
        else np.full((1, 2), np.inf)
    )
    usable_rate = len(usable) / max(len(records), 1)
    mean_relative_error = robust_relative_error.mean(axis=0)
    plugin_aggregate_bias = float(np.linalg.norm(plugin_bias))
    robust_aggregate_bias = float(np.linalg.norm(robust_bias))
    gates = [
        {
            "name": "stationarity_residual_max_abs",
            "value": build_stationary_problem()[4],
            "operator": "<=",
            "threshold": 1e-10,
            "passed": build_stationary_problem()[4] <= 1e-10,
        },
        {
            "name": "paired_replications",
            "value": len(records),
            "operator": ">=",
            "threshold": FINAL_REPLICATIONS,
            "passed": len(records) >= FINAL_REPLICATIONS,
        },
        {
            "name": "usable_rate",
            "value": usable_rate,
            "operator": ">=",
            "threshold": 0.90,
            "passed": usable_rate >= 0.90,
        },
        {
            "name": "mean_parameter_relative_error_max",
            "value": float(np.max(mean_relative_error)),
            "operator": "<=",
            "threshold": 0.25,
            "passed": bool(np.max(mean_relative_error) <= 0.25),
        },
        {
            "name": "locally_robust_aggregate_bias_vs_plugin",
            "value": robust_aggregate_bias,
            "operator": "<=",
            "threshold": plugin_aggregate_bias,
            "passed": robust_aggregate_bias <= plugin_aggregate_bias,
        },
    ]
    passed = bool(final_run and all(gate["passed"] for gate in gates))
    return {
        "completed_replications": len(records),
        "usable_replications": len(usable),
        "usable_rate": usable_rate,
        "mean_locally_robust_relative_error": mean_relative_error.tolist(),
        "plugin_bias": plugin_bias.tolist(),
        "locally_robust_bias": robust_bias.tolist(),
        "plugin_aggregate_bias": plugin_aggregate_bias,
        "locally_robust_aggregate_bias": robust_aggregate_bias,
        "gates": gates,
        "passed": passed,
    }


def write_result(
    records: list[dict[str, Any]],
    *,
    output: Path,
    final_run: bool,
    checkpoints: list[Path],
) -> bool:
    summary = summarize(records, final_run=final_run)
    payload = {
        "estimator": "TD-CCP",
        "status": "ready" if summary["passed"] else "smoke_only" if not final_run else "not_ready",
        "design": {
            "method": "neural AVI",
            "comparison": "paired plug-in and Algorithm 2 locally robust estimates",
            "n_individuals": N_INDIVIDUALS,
            "n_periods": N_PERIODS,
            "base_seed": BASE_SEED,
            "initial_distribution": "stationary under the oracle policy",
            "stationarity_residual_max_abs": build_stationary_problem()[4],
        },
        "summary": summary,
        "records": records,
        "checkpoints": [str(path) for path in checkpoints],
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return bool(summary["passed"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=FINAL_REPLICATIONS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.merge:
        record_map = read_records(args.merge)
        records = [record_map[index] for index in sorted(record_map)]
        passed = write_result(
            records,
            output=args.output,
            final_run=len(records) >= FINAL_REPLICATIONS,
            checkpoints=args.merge,
        )
        return 0 if passed else 1
    n_reps = min(args.n_reps, 2) if args.smoke else args.n_reps
    output = Path("/tmp/econirl_tdccp_neural_avi_smoke.json") if args.smoke else args.output
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    done = read_records([checkpoint])
    for rep in range(args.start_rep, args.start_rep + n_reps):
        if rep in done:
            continue
        record = fit_pair(rep)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        done[rep] = record
        status = record.get("error") or "paired fits complete"
        print(f"AVI rep {rep}: {status}", flush=True)
    records = [done[index] for index in sorted(done)]
    complete_design = (
        args.start_rep == 0 and n_reps >= FINAL_REPLICATIONS and len(records) >= FINAL_REPLICATIONS
    )
    passed = write_result(
        records,
        output=output,
        final_run=complete_design,
        checkpoints=[checkpoint],
    )
    return 0 if not complete_design or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
