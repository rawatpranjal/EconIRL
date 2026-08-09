#!/usr/bin/env python3
"""Calibrate the public NFXP trajectory bootstrap on known truth."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "nfxp_bootstrap_calibration.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.reward_spec import RewardSpec  # noqa: E402
from econirl.estimators import NFXP  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402
from validation.estimators.nfxp.ready import (  # noqa: E402
    MC_CONFIG,
    _git_commit,
    _package_versions,
    _strict_json,
    build_problem,
)

CALIBRATION_REPS = 50
N_BOOTSTRAP = 99
PANEL_BASE_SEED = 41_000
BOOTSTRAP_BASE_SEED = 9_901


def fit_bootstrap(rep: int, n_bootstrap: int) -> dict[str, Any]:
    """Fit one independently simulated panel and retain public bootstrap output."""
    env = build_problem(MC_CONFIG)
    panel_seed = PANEL_BASE_SEED + rep
    bootstrap_seed = BOOTSTRAP_BASE_SEED + rep
    panel = simulate_panel(
        env,
        n_individuals=MC_CONFIG.n_individuals,
        n_periods=MC_CONFIG.n_periods,
        seed=panel_seed,
    )
    transitions = np.asarray(env.transition_matrices, dtype=np.float64)
    reward = RewardSpec(np.asarray(env.feature_matrix), names=env.parameter_names)
    truth = np.asarray(list(env.true_parameters.values()), dtype=np.float64)
    try:
        model = NFXP(
            n_states=env.num_states,
            n_actions=env.num_actions,
            discount=float(env.problem_spec.discount_factor),
            utility=reward,
            se_method="bootstrap",
            n_bootstrap=n_bootstrap,
            se_seed=bootstrap_seed,
        )
        model.fit(panel, transitions=transitions)
        result = model.bootstrap_
        if result is None:
            raise RuntimeError("public bootstrap result was not populated")
        widths = result.intervals[:, 1] - result.intervals[:, 0]
        covered = (result.intervals[:, 0] <= truth) & (truth <= result.intervals[:, 1])
        return {
            "rep": rep,
            "panel_seed": panel_seed,
            "bootstrap_seed": bootstrap_seed,
            "n_requested": result.n_requested,
            "n_successful": result.n_successful,
            "success_fraction": result.n_successful / result.n_requested,
            "estimate": np.asarray(model.coef_, dtype=float).tolist(),
            "standard_errors": result.standard_errors.tolist(),
            "intervals": result.intervals.tolist(),
            "interval_widths": widths.tolist(),
            "covered": covered.tolist(),
            "failure_count": len(result.failures),
            "failures": list(result.failures),
            "summary": model.summary() if rep == 0 else None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are calibration evidence
        return {
            "rep": rep,
            "panel_seed": panel_seed,
            "bootstrap_seed": bootstrap_seed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _checkpoint_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["rep"])] = record
    return records


def run_calibration(
    *,
    n_replications: int,
    n_bootstrap: int,
    checkpoint: Path,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Run or resume the frozen bootstrap calibration."""
    done = _checkpoint_records(checkpoint)
    records = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(n_replications):
        if rep in done:
            records.append(done[rep])
            continue
        record = fit_bootstrap(rep, n_bootstrap)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        if verbose:
            status = record["error"] or (
                f"successful={record['n_successful']}/{record['n_requested']}"
            )
            print(f"bootstrap {rep + 1}/{n_replications}: {status}", flush=True)
    return records


def summarize_bootstrap(
    records: list[dict[str, Any]],
    parameter_names: list[str],
) -> dict[str, Any]:
    """Aggregate success, width, and repeated-sample coverage."""
    usable = [record for record in records if record.get("error") is None]
    if not usable:
        return {
            "n_total": len(records),
            "n_usable": 0,
            "usable_rate": 0.0,
            "minimum_success_fraction": 0.0,
            "parameters": {},
        }
    covered = np.asarray([record["covered"] for record in usable], dtype=bool)
    widths = np.asarray([record["interval_widths"] for record in usable], dtype=float)
    return {
        "n_total": len(records),
        "n_usable": len(usable),
        "usable_rate": len(usable) / len(records),
        "minimum_success_fraction": float(min(record["success_fraction"] for record in usable)),
        "total_failed_draws": int(sum(record["failure_count"] for record in usable)),
        "parameters": {
            name: {
                "coverage_95": float(covered[:, index].mean()),
                "mean_interval_width": float(widths[:, index].mean()),
                "median_interval_width": float(np.median(widths[:, index])),
            }
            for index, name in enumerate(parameter_names)
        },
    }


def calibration_gates(summary: dict[str, Any], *, final_run: bool) -> list[dict[str, Any]]:
    """Evaluate the thresholds frozen before this run."""
    gates = []

    def add(name: str, value: float, operator: str, threshold: float) -> None:
        passed = value >= threshold if operator == ">=" else value <= threshold
        gates.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
                "enforced": final_run,
            }
        )

    add("usable_rate", summary["usable_rate"], ">=", 0.95)
    add("minimum_success_fraction", summary["minimum_success_fraction"], ">=", 0.95)
    for name, values in summary.get("parameters", {}).items():
        add(f"{name}_coverage_low", values["coverage_95"], ">=", 0.86)
        add(f"{name}_coverage_high", values["coverage_95"], "<=", 1.00)
        add(f"{name}_mean_width_positive", values["mean_interval_width"], ">=", 1e-12)
    return gates


def reproducibility_check(n_bootstrap: int) -> dict[str, Any]:
    """Run the same bootstrap program twice and require exact public parity."""
    first = fit_bootstrap(10_000, n_bootstrap)
    second = fit_bootstrap(10_000, n_bootstrap)
    if first.get("error") is not None or second.get("error") is not None:
        return {
            "passed": False,
            "first_error": first.get("error"),
            "second_error": second.get("error"),
        }
    fields = ("estimate", "standard_errors", "intervals", "covered", "failures")
    passed = all(first[field] == second[field] for field in fields)
    return {
        "passed": passed,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": first["bootstrap_seed"],
        "fields_compared": list(fields),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--records-parquet", type=Path)
    parser.add_argument("--n-reps", type=int, default=CALIBRATION_REPS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    n_replications = min(args.n_reps, 2) if args.smoke else args.n_reps
    n_bootstrap = min(args.n_bootstrap, 9) if args.smoke else args.n_bootstrap
    final_run = n_replications >= CALIBRATION_REPS and n_bootstrap >= N_BOOTSTRAP
    output = args.output or (
        Path("/tmp/econirl_nfxp_bootstrap_smoke.json") if args.smoke else DEFAULT_OUTPUT
    )
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    records = run_calibration(
        n_replications=n_replications,
        n_bootstrap=n_bootstrap,
        checkpoint=checkpoint,
        verbose=not args.quiet,
    )
    env = build_problem(MC_CONFIG)
    summary = summarize_bootstrap(records, list(env.parameter_names))
    gates = calibration_gates(summary, final_run=final_run)
    program_check = reproducibility_check(9 if args.smoke else 19)
    gates.append(
        {
            "name": "program_reproducibility",
            "value": program_check["passed"],
            "operator": "is",
            "threshold": True,
            "passed": bool(program_check["passed"]),
            "enforced": final_run,
        }
    )
    payload = {
        "estimator": "NFXP",
        "status": (
            "ready"
            if final_run and all(gate["passed"] for gate in gates)
            else "smoke_only"
            if not final_run
            else "not_ready"
        ),
        "design": {
            "resampling_method": "pairs cluster bootstrap",
            "resampling_unit": "whole individual trajectory",
            "failure_policy": "record failure and do not retry",
            "problem": asdict(MC_CONFIG),
            "n_calibration_panels": n_replications,
            "n_bootstrap": n_bootstrap,
            "panel_base_seed": PANEL_BASE_SEED,
            "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        },
        "summary": summary,
        "program_check": program_check,
        "gates": gates,
        "provenance": {
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
        },
        "checkpoint": str(checkpoint),
        "records_parquet": (
            str(args.records_parquet) if args.records_parquet is not None else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    failed = [gate["name"] for gate in gates if gate["enforced"] and not gate["passed"]]
    if failed:
        print("failed gates:")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
