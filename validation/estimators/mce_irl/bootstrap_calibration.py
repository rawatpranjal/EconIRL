#!/usr/bin/env python3
"""Calibrate the public MCE-IRL trajectory bootstrap on known truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "mce_irl_bootstrap_calibration.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validation.estimators.mce_irl.ready import (  # noqa: E402
    TRUE_THETA,
    make_panel,
    model_spec,
)
from validation.estimators.nfxp.ready import (  # noqa: E402
    _git_commit,
    _package_versions,
    _strict_json,
)

CALIBRATION_REPS = 50
N_BOOTSTRAP = 99
PANEL_BASE_SEED = 91_000
BOOTSTRAP_BASE_SEED = 92_000
N_OBSERVATIONS = 400


def fit_bootstrap(replication: int, n_bootstrap: int) -> dict[str, Any]:
    """Fit one independent panel and retain the public bootstrap record."""
    panel_seed = PANEL_BASE_SEED + replication
    bootstrap_seed = BOOTSTRAP_BASE_SEED + replication
    panel = make_panel(seed=panel_seed, n_observations=N_OBSERVATIONS)
    try:
        model, transitions, tasks = model_spec(
            se_method="bootstrap",
            n_bootstrap=n_bootstrap,
            se_seed=bootstrap_seed,
        )
        model.fit(panel, transitions=transitions, tasks=tasks)
        result = model.bootstrap_
        if result is None:
            raise RuntimeError("public bootstrap result was not populated")
        estimate = float(model.params_["action_one"])
        lower, upper = (float(value) for value in result.intervals[0])
        width = upper - lower
        return {
            "replication": replication,
            "panel_seed": panel_seed,
            "bootstrap_seed": bootstrap_seed,
            "n_requested": result.n_requested,
            "n_successful": result.n_successful,
            "success_fraction": result.n_successful / result.n_requested,
            "estimate": estimate,
            "standard_error": float(result.standard_errors[0]),
            "interval": [lower, upper],
            "interval_width": width,
            "covered": lower <= TRUE_THETA <= upper,
            "lower_miss": TRUE_THETA < lower,
            "upper_miss": TRUE_THETA > upper,
            "failure_count": len(result.failures),
            "failures": list(result.failures),
            "summary": model.summary() if replication == 0 else None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are calibration evidence
        return {
            "replication": replication,
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
            records[int(record["replication"])] = record
    return records


def run_calibration(
    *,
    n_replications: int,
    n_bootstrap: int,
    checkpoint: Path,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Run or resume the frozen MCE-IRL bootstrap calibration."""
    completed = _checkpoint_records(checkpoint)
    records = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for replication in range(n_replications):
        if replication in completed:
            records.append(completed[replication])
            continue
        record = fit_bootstrap(replication, n_bootstrap)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        if verbose:
            status = record["error"] or (
                f"successful={record['n_successful']}/{record['n_requested']}"
            )
            print(f"bootstrap {replication + 1}/{n_replications}: {status}", flush=True)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate success, coverage, tail misses, and interval widths."""
    usable = [record for record in records if record.get("error") is None]
    if not usable:
        return {
            "n_total": len(records),
            "n_usable": 0,
            "usable_rate": 0.0,
            "minimum_success_fraction": 0.0,
        }
    widths = np.asarray([record["interval_width"] for record in usable], dtype=float)
    return {
        "n_total": len(records),
        "n_usable": len(usable),
        "usable_rate": len(usable) / len(records),
        "minimum_success_fraction": float(min(record["success_fraction"] for record in usable)),
        "total_successful_draws": int(sum(record["n_successful"] for record in usable)),
        "total_requested_draws": int(sum(record["n_requested"] for record in usable)),
        "total_failed_draws": int(sum(record["failure_count"] for record in usable)),
        "coverage_95": float(np.mean([record["covered"] for record in usable])),
        "lower_tail_miss_rate": float(np.mean([record["lower_miss"] for record in usable])),
        "upper_tail_miss_rate": float(np.mean([record["upper_miss"] for record in usable])),
        "mean_interval_width": float(widths.mean()),
        "median_interval_width": float(np.median(widths)),
        "minimum_interval_width": float(widths.min()),
        "maximum_interval_width": float(widths.max()),
    }


def gates(summary: dict[str, Any], *, final_run: bool) -> list[dict[str, Any]]:
    """Evaluate thresholds frozen before the final calibration."""
    specifications = (
        ("usable_rate", summary["usable_rate"], ">=", 0.95),
        (
            "minimum_success_fraction",
            summary["minimum_success_fraction"],
            ">=",
            0.95,
        ),
        ("coverage_low", summary.get("coverage_95", 0.0), ">=", 0.86),
        ("coverage_high", summary.get("coverage_95", 0.0), "<=", 1.0),
        (
            "mean_interval_width_positive",
            summary.get("mean_interval_width", 0.0),
            ">=",
            1e-12,
        ),
    )
    output = []
    for name, value, operator, threshold in specifications:
        passed = value >= threshold if operator == ">=" else value <= threshold
        output.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
                "enforced": final_run,
            }
        )
    return output


def reproducibility_check(n_bootstrap: int) -> dict[str, Any]:
    """Run the same bootstrap program twice and require exact parity."""
    first = fit_bootstrap(10_000, n_bootstrap)
    second = fit_bootstrap(10_000, n_bootstrap)
    if first.get("error") is not None or second.get("error") is not None:
        return {
            "passed": False,
            "first_error": first.get("error"),
            "second_error": second.get("error"),
        }
    fields = ("estimate", "standard_error", "interval", "covered", "failures")
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
    parser.add_argument("--n-reps", type=int, default=CALIBRATION_REPS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    n_replications = min(args.n_reps, 2) if args.smoke else args.n_reps
    n_bootstrap = min(args.n_bootstrap, 9) if args.smoke else args.n_bootstrap
    final_run = n_replications >= CALIBRATION_REPS and n_bootstrap >= N_BOOTSTRAP
    output = args.output or (
        Path("/tmp/econirl_mce_irl_bootstrap_smoke.json") if args.smoke else DEFAULT_OUTPUT
    )
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    records = run_calibration(
        n_replications=n_replications,
        n_bootstrap=n_bootstrap,
        checkpoint=checkpoint,
        verbose=not args.quiet,
    )
    summary = summarize(records)
    result_gates = gates(summary, final_run=final_run)
    program_check = reproducibility_check(9 if args.smoke else 19)
    result_gates.append(
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
        "estimator": "MCE-IRL",
        "status": (
            "ready"
            if final_run and all(gate["passed"] for gate in result_gates)
            else "smoke_only"
            if not final_run
            else "not_ready"
        ),
        "design": {
            "resampling_method": "pairs cluster bootstrap",
            "resampling_unit": "whole individual trajectory",
            "transition_policy": "supplied deterministic transitions held fixed",
            "failure_policy": "record failure and do not retry",
            "truth": TRUE_THETA,
            "n_observations_per_panel": N_OBSERVATIONS,
            "n_calibration_panels": n_replications,
            "n_bootstrap": n_bootstrap,
            "panel_base_seed": PANEL_BASE_SEED,
            "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        },
        "summary": summary,
        "program_check": program_check,
        "gates": result_gates,
        "provenance": {
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
        },
        "checkpoint": str(checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    failed = [gate["name"] for gate in result_gates if gate["enforced"] and not gate["passed"]]
    if failed:
        print("failed gates:")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
