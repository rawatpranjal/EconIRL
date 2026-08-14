#!/usr/bin/env python3
"""Calibrate the public TD-CCP pairs-cluster bootstrap on known truth."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from econirl import TDCCP  # noqa: E402
from econirl.core.reward_spec import RewardSpec  # noqa: E402
from econirl.environments import ArrayMDP  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402
from validation.estimators.nfxp.bootstrap_calibration import (  # noqa: E402
    calibration_gates,
    summarize_bootstrap,
)
from validation.estimators.nfxp.ready import _strict_json  # noqa: E402

DEFAULT_OUTPUT = ROOT / "validation" / "results" / "tdccp_bootstrap_calibration.json"
CALIBRATION_REPS = 50
N_BOOTSTRAP = 99
N_INDIVIDUALS = 250
N_PERIODS = 40
PANEL_BASE_SEED = 121_000
BOOTSTRAP_BASE_SEED = 131_000


def build_problem() -> tuple[ArrayMDP, np.ndarray, RewardSpec, np.ndarray]:
    """Build a compact identified replacement problem."""
    n_states = 12
    transitions = np.zeros((2, n_states, n_states), dtype=np.float64)
    for state in range(n_states):
        transitions[0, state, state] = 0.65
        transitions[0, state, min(state + 1, n_states - 1)] += 0.35
    transitions[1, :, 0] = 1.0
    condition = np.linspace(0.0, 1.0, n_states)
    features = np.zeros((n_states, 2, 2), dtype=np.float64)
    features[:, 0, 0] = -condition
    features[:, 1, 1] = -1.0
    truth = np.array([1.5, 2.2], dtype=np.float64)
    names = ["condition_cost", "replacement_cost"]
    reward = RewardSpec(features, names=names)
    env = ArrayMDP(
        transitions,
        features,
        theta=truth,
        discount_factor=0.95,
        scale_parameter=1.0,
        parameter_names=names,
        seed=PANEL_BASE_SEED,
    )
    return env, transitions, reward, truth


def fit_bootstrap(rep: int, n_bootstrap: int) -> dict[str, Any]:
    """Fit one independent panel and retain the public bootstrap result."""
    env, transitions, reward, truth = build_problem()
    panel_seed = PANEL_BASE_SEED + rep
    bootstrap_seed = BOOTSTRAP_BASE_SEED + rep
    panel = simulate_panel(
        env,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=panel_seed,
    )
    try:
        model = TDCCP(
            n_states=env.num_states,
            n_actions=env.num_actions,
            discount=float(env.problem_spec.discount_factor),
            utility=reward,
            se_method="bootstrap",
            n_bootstrap=n_bootstrap,
            se_seed=bootstrap_seed,
            seed=panel_seed,
            method="semigradient",
            basis_type="polynomial",
            basis_dim=4,
            basis_ridge=1e-7,
            ccp_method="logit",
            ccp_poly_degree=2,
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=500,
            outer_tol=1e-7,
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


def _merge_checkpoint_records(paths: list[Path]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in paths:
        for rep, record in _checkpoint_records(path).items():
            if rep in records and records[rep] != record:
                raise ValueError(f"conflicting bootstrap record for rep={rep}")
            records[rep] = record
    return records


def run_calibration(
    *,
    start_rep: int,
    n_replications: int,
    n_bootstrap: int,
    checkpoint: Path,
) -> list[dict[str, Any]]:
    """Run or resume a contiguous bootstrap shard."""
    done = _checkpoint_records(checkpoint)
    records = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(start_rep, start_rep + n_replications):
        record = done.get(rep)
        if record is None:
            record = fit_bootstrap(rep, n_bootstrap)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        status = record.get("error") or (
            f"successful={record['n_successful']}/{record['n_requested']}"
        )
        print(f"bootstrap rep {rep}: {status}", flush=True)
    return records


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
    fields = ("estimate", "standard_errors", "intervals", "covered", "failures")
    return {
        "passed": all(first[field] == second[field] for field in fields),
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": first["bootstrap_seed"],
        "fields_compared": list(fields),
    }


def _write_payload(
    records: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    final_run: bool,
    output: Path,
    checkpoint: Path,
    program_check: dict[str, Any] | None,
) -> bool:
    _env, _transitions, reward, _truth = build_problem()
    summary = summarize_bootstrap(records, reward.parameter_names)
    gates = calibration_gates(summary, final_run=final_run)
    if program_check is not None:
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
    passed = bool(final_run and all(gate["passed"] for gate in gates))
    payload = {
        "estimator": "TD-CCP",
        "status": "ready" if passed else "smoke_only" if not final_run else "not_ready",
        "design": {
            "resampling_method": "pairs cluster bootstrap",
            "resampling_unit": "whole individual trajectory",
            "transition_policy": "supplied transition tensor held fixed",
            "failure_policy": "record failure and do not retry",
            "n_individuals": N_INDIVIDUALS,
            "n_periods": N_PERIODS,
            "n_calibration_panels": len(records),
            "n_bootstrap": n_bootstrap,
            "panel_base_seed": PANEL_BASE_SEED,
            "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        },
        "summary": summary,
        "program_check": program_check,
        "gates": gates,
        "records": records,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        },
        "checkpoint": str(checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=CALIBRATION_REPS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-program-check", action="store_true")
    args = parser.parse_args()
    if args.merge:
        record_map = _merge_checkpoint_records(args.merge)
        records = [record_map[index] for index in sorted(record_map)]
        complete_design = len(records) >= CALIBRATION_REPS and args.n_bootstrap >= N_BOOTSTRAP
        program_check = None
        if not args.skip_program_check:
            program_check = reproducibility_check(19)
        passed = _write_payload(
            records,
            n_bootstrap=args.n_bootstrap,
            final_run=complete_design,
            output=args.output,
            checkpoint=args.merge[0],
            program_check=program_check,
        )
        return 0 if passed else 1
    n_reps = min(args.n_reps, 2) if args.smoke else args.n_reps
    n_bootstrap = min(args.n_bootstrap, 9) if args.smoke else args.n_bootstrap
    complete_design = (
        args.start_rep == 0 and n_reps >= CALIBRATION_REPS and n_bootstrap >= N_BOOTSTRAP
    )
    output = Path("/tmp/econirl_tdccp_bootstrap_smoke.json") if args.smoke else args.output
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    records = run_calibration(
        start_rep=args.start_rep,
        n_replications=n_reps,
        n_bootstrap=n_bootstrap,
        checkpoint=checkpoint,
    )
    program_check = None
    if not args.skip_program_check:
        program_check = reproducibility_check(9 if args.smoke else 19)
    passed = _write_payload(
        records,
        n_bootstrap=n_bootstrap,
        final_run=complete_design,
        output=output,
        checkpoint=checkpoint,
        program_check=program_check,
    )
    return 0 if not complete_design or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
