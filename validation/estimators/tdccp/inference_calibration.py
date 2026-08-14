#!/usr/bin/env python3
"""Calibrate TD-CCP Algorithm 2 standard errors on known truth."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.core.types import DDCProblem  # noqa: E402
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402
from validation.estimators.nfxp.ready import _strict_json  # noqa: E402
from validation.estimators.tdccp.bus_engine_mc import (  # noqa: E402
    THETA_TRUE,
    build_dgp,
    stationary_initial_distribution,
)

DEFAULT_OUTPUT = ROOT / "validation" / "results" / "tdccp_inference.json"
FINAL_REPLICATIONS = 1_000
N_INDIVIDUALS = 3_000
N_PERIODS = 30
BASE_SEED = 171_000


@lru_cache(maxsize=1)
def design() -> tuple[dict[str, Any], jnp.ndarray, jnp.ndarray]:
    """Build the paper-shaped bus design once per process."""
    dgp = build_dgp()
    states = np.arange(dgp["problem"].num_states)
    mileage = (states // 2).astype(np.float64) / 20.0
    kind = (states % 2).astype(np.float64)
    encoded = np.column_stack(
        [
            kind,
            mileage,
            mileage**2,
            kind * mileage,
            kind * mileage**2,
            mileage**3,
            kind * mileage**3,
        ]
    )

    def state_encoder(indices: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(encoded)[jnp.asarray(indices, dtype=jnp.int32)]

    dgp = dict(dgp)
    dgp["problem"] = DDCProblem(
        num_states=len(states),
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
        state_dim=encoded.shape[1],
        state_encoder=state_encoder,
    )
    reward = dgp["utility"].compute(jnp.asarray(THETA_TRUE))
    solved = value_iteration(
        SoftBellmanOperator(dgp["problem"], dgp["transitions"]),
        reward,
        tol=1e-12,
        max_iter=20_000,
    )
    initial = stationary_initial_distribution(dgp["problem"], dgp["transitions"], solved.policy)
    policy_kernel = np.einsum(
        "sa,asj->sj", np.asarray(solved.policy), np.asarray(dgp["transitions"])
    )
    dgp["stationarity_residual_max_abs"] = float(
        np.max(np.abs(np.asarray(initial) @ policy_kernel - np.asarray(initial)))
    )
    return dgp, solved.policy, initial


def fit_once(rep: int) -> dict[str, Any]:
    """Fit one panel with Algorithm 2 and retain its clustered standard errors."""
    dgp, oracle_policy, initial = design()
    truth = np.asarray(THETA_TRUE, dtype=float)
    seed = BASE_SEED + rep
    panel = simulate_panel_from_policy(
        dgp["problem"],
        dgp["transitions"],
        oracle_policy,
        initial,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=seed,
    )
    problem = dgp["problem"]
    utility = dgp["utility"]
    config = TDCCPConfig(
        method="semigradient",
        basis_type="encoded",
        basis_dim=1,
        basis_ridge=0.0,
        basis_action_coding="reference",
        ccp_method="logit",
        ccp_poly_degree=1,
        ccp_use_encoder=True,
        cross_fitting=True,
        cross_fit_ccp=False,
        robust_se=True,
        linear_robust_correction="sensitivity",
        n_policy_iterations=1,
        outer_max_iter=500,
        outer_tol=1e-7,
        compute_se=False,
        compute_policy=False,
    )
    started = time.perf_counter()
    try:
        result = TDCCPEstimator(config=config, seed=seed).estimate(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=dgp["transitions"],
        )
        inference = result.metadata.get("paper_inference") or {}
        standard_errors = np.asarray(inference.get("standard_errors"), dtype=float)
        return {
            "rep": rep,
            "seed": seed,
            "parameters": np.asarray(result.parameters, dtype=float).tolist(),
            "standard_errors": standard_errors.tolist(),
            "truth": truth.tolist(),
            "converged": bool(result.converged),
            "moment_norm_max": float(inference.get("moment_norm_max", np.nan)),
            "preliminary_optimizer_stationary": [
                bool(value) for value in inference.get("preliminary_optimizer_stationary", [])
            ],
            "robust_optimizer_stationary": [
                bool(value) for value in inference.get("robust_optimizer_stationary", [])
            ],
            "runtime_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are calibration evidence
        return {
            "rep": rep,
            "seed": seed,
            "converged": False,
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def read_records(paths: list[Path]) -> dict[int, dict[str, Any]]:
    """Read and de-duplicate checkpoint records."""
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
        and np.all(np.isfinite(record["parameters"]))
        and np.all(np.isfinite(record["standard_errors"]))
        and np.all(np.asarray(record["standard_errors"]) > 0.0)
        and all(record["preliminary_optimizer_stationary"])
        and all(record["robust_optimizer_stationary"])
    ]
    truth = np.asarray(THETA_TRUE, dtype=float)
    estimates = np.asarray([record["parameters"] for record in usable], dtype=float)
    standard_errors = np.asarray([record["standard_errors"] for record in usable], dtype=float)
    if len(usable) > 1:
        mean = estimates.mean(axis=0)
        empirical_sd = estimates.std(axis=0, ddof=1)
        mean_se = standard_errors.mean(axis=0)
        lower = estimates - 1.96 * standard_errors
        upper = estimates + 1.96 * standard_errors
        coverage = ((lower <= truth) & (truth <= upper)).mean(axis=0)
        lower_miss = (lower > truth).mean(axis=0)
        upper_miss = (upper < truth).mean(axis=0)
        standardized_bias = np.abs(mean - truth) / empirical_sd
        se_ratio = mean_se / empirical_sd
    else:
        mean = empirical_sd = mean_se = np.full(3, np.nan)
        coverage = lower_miss = upper_miss = np.full(3, np.nan)
        standardized_bias = se_ratio = np.full(3, np.nan)
    usable_rate = len(usable) / max(len(records), 1)
    gate_specs = [
        (
            "stationarity_residual_max_abs",
            design()[0]["stationarity_residual_max_abs"],
            "<=",
            1e-10,
            design()[0]["stationarity_residual_max_abs"] <= 1e-10,
        ),
        (
            "replications",
            len(records),
            ">=",
            FINAL_REPLICATIONS,
            len(records) >= FINAL_REPLICATIONS,
        ),
        ("usable_rate", usable_rate, ">=", 0.99, usable_rate >= 0.99),
        (
            "standardized_bias_max",
            float(np.nanmax(standardized_bias)),
            "<=",
            0.20,
            bool(np.all(standardized_bias <= 0.20)),
        ),
        (
            "mean_se_to_empirical_sd_min",
            float(np.nanmin(se_ratio)),
            ">=",
            0.80,
            bool(np.all(se_ratio >= 0.80)),
        ),
        (
            "mean_se_to_empirical_sd_max",
            float(np.nanmax(se_ratio)),
            "<=",
            1.20,
            bool(np.all(se_ratio <= 1.20)),
        ),
        (
            "coverage_min",
            float(np.nanmin(coverage)),
            ">=",
            0.91,
            bool(np.all(coverage >= 0.91)),
        ),
        (
            "coverage_max",
            float(np.nanmax(coverage)),
            "<=",
            0.99,
            bool(np.all(coverage <= 0.99)),
        ),
        (
            "tail_miss_min",
            float(np.nanmin(np.concatenate([lower_miss, upper_miss]))),
            ">=",
            0.01,
            bool(np.all(lower_miss >= 0.01) and np.all(upper_miss >= 0.01)),
        ),
        (
            "tail_miss_max",
            float(np.nanmax(np.concatenate([lower_miss, upper_miss]))),
            "<=",
            0.04,
            bool(np.all(lower_miss <= 0.04) and np.all(upper_miss <= 0.04)),
        ),
    ]
    gates = [
        {
            "name": name,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }
        for name, value, operator, threshold, passed in gate_specs
    ]
    passed = bool(final_run and all(gate["passed"] for gate in gates))
    return {
        "completed_replications": len(records),
        "usable_replications": len(usable),
        "usable_rate": usable_rate,
        "truth": truth.tolist(),
        "mean_estimate": mean.tolist(),
        "empirical_sd": empirical_sd.tolist(),
        "mean_standard_error": mean_se.tolist(),
        "standardized_bias": standardized_bias.tolist(),
        "mean_se_to_empirical_sd": se_ratio.tolist(),
        "coverage_95": coverage.tolist(),
        "lower_tail_miss_rate": lower_miss.tolist(),
        "upper_tail_miss_rate": upper_miss.tolist(),
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
            "problem": "paper-shaped bus replacement model",
            "n_individuals": N_INDIVIDUALS,
            "n_periods": N_PERIODS,
            "base_seed": BASE_SEED,
            "method": "semigradient with Algorithm 2 locally robust inference",
            "basis": "explicit third-order mileage and type polynomial",
            "initial_distribution": "stationary under the oracle policy",
            "stationarity_residual_max_abs": design()[0]["stationarity_residual_max_abs"],
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
    args = parser.parse_args()
    if args.merge:
        record_map = read_records(args.merge)
        records = [record_map[index] for index in sorted(record_map)]
        complete = len(records) >= FINAL_REPLICATIONS
        passed = write_result(
            records,
            output=args.output,
            final_run=complete,
            checkpoints=args.merge,
        )
        return 0 if passed else 1
    checkpoint = args.checkpoint or args.output.with_suffix(".jsonl")
    done = read_records([checkpoint])
    for rep in range(args.start_rep, args.start_rep + args.n_reps):
        if rep in done:
            continue
        record = fit_once(rep)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        done[rep] = record
        status = record.get("error") or f"{record['runtime_seconds']:.2f}s"
        print(f"inference rep {rep}: {status}", flush=True)
    records = [done[index] for index in sorted(done)]
    complete = (
        args.start_rep == 0
        and args.n_reps >= FINAL_REPLICATIONS
        and len(records) >= FINAL_REPLICATIONS
    )
    passed = write_result(
        records,
        output=args.output,
        final_run=complete,
        checkpoints=[checkpoint],
    )
    return 0 if not complete or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
