#!/usr/bin/env python3
"""Generate UFXP results from the known-truth DGP harness.

Runs the optimally weighted UFXP (OUFXP form, the package default) on the
same canonical synthetic cell as the other structural estimators, with known
reward, value, policy, and Type A/B/C counterfactual oracle objects.

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python validation/estimators/ufxp/run.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "ufxp.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_ufxp_known_truth")
CELL_ID = "canonical_low_action"
ESTIMATOR = "UFXP"
PAPER_TARGET = (
    "Unnested fixed point estimation from Bellman first-order conditions "
    "(Bray; Oguz and Bray 2026), optimally weighted"
)
RELEASE_STATUS = "Certified"

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validation.known_truth import (  # noqa: E402
    build_known_truth_dgp,
    get_cell,
    run_cell_estimator,
    to_jsonable,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", default=CELL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("UFXP: running known-truth validation")
    print(f"  cell: {args.cell_id}")

    run_dir = run_cell_estimator(
        args.cell_id,
        ESTIMATOR,
        args.output_dir,
        smoke=False,
        show_progress=args.show_progress,
        verbose=args.verbose,
    )
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    cell = get_cell(args.cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    JSON_OUT.write_text(
        json.dumps(compact_payload(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(render_markdown(result, dgp))
    print(f"  result: {result_path}")
    print(f"  wrote: {JSON_OUT}")


def compact_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return the repo-local machine-readable artifact."""

    return {
        "estimator": ESTIMATOR,
        "paper_target": PAPER_TARGET,
        "release_status": RELEASE_STATUS,
        "primary_cell_id": result.get("cell", {}).get("cell_id", CELL_ID),
        "result": _finite_jsonable(result),
    }


def _finite_jsonable(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, dict):
        return {key: _finite_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def render_markdown(result: dict[str, Any], dgp: Any) -> str:
    """Markdown tables for the public validation page (copied verbatim there)."""
    summary = result["summary"]
    diagnostics = result["diagnostics"]
    metrics = result["metrics"]
    gates = result["gates"]
    simulation = result["simulation"]

    names = list(summary["parameter_names"])
    estimates = np.asarray(summary["parameters"], dtype=float)
    ses = np.asarray(summary["standard_errors"], dtype=float)
    truth = np.asarray(dgp.homogeneous_parameters, dtype=float)

    L: list[str] = []
    L.append("\n## Fit Summary\n")
    L.append("| Quantity | Value |")
    L.append("| --- | --- |")
    L.append(f"| Converged | {summary['converged']} |")
    L.append(f"| Log-likelihood | {summary['log_likelihood']:.4f} |")
    L.append(f"| Estimation time | {summary['estimation_time']:.2f} seconds |")
    L.append(f"| Individuals | {int(simulation['n_individuals']):,} |")
    L.append(f"| Periods | {int(simulation['n_periods']):,} |")
    L.append(f"| Observations | {int(summary['num_observations']):,} |")
    L.append("\n## Parameter Recovery\n")
    L.append("| Parameter | Truth | Estimate | Std. error | Error |")
    L.append("| --- | --- | --- | --- | --- |")
    for k, name in enumerate(names):
        L.append(f"| {name} | {truth[k]:.4f} | {estimates[k]:.4f} | "
                 f"{ses[k]:.4f} | {estimates[k] - truth[k]:+.4f} |")
    L.append("\n## Recovery Metrics\n")
    L.append("| Metric | Value |")
    L.append("| --- | --- |")
    p = metrics["parameters"]
    L.append(f"| Parameter RMSE | {p['rmse']:.4f} |")
    L.append(f"| Parameter cosine similarity | {p['cosine_similarity']:.4f} |")
    L.append(f"| Reward RMSE | {metrics['reward_rmse']:.4f} |")
    L.append(f"| Value RMSE | {metrics['value_rmse']:.4f} |")
    L.append(f"| Q RMSE | {metrics['q_rmse']:.4f} |")
    L.append(f"| Policy TV | {metrics['policy']['tv']:.4f} |")
    L.append(f"| Policy KL | {metrics['policy']['kl']:.6f} |")
    L.append("\n## Numerical Checks\n")
    L.append("| Check | Value | Threshold | Status |")
    L.append("| --- | --- | --- | --- |")
    for g in gates:
        status = "pass" if g["passed"] else "FAIL"
        L.append(f"| {g['name']} | {g['value']:.4g} | {g['operator']} "
                 f"{g['threshold']:.4g} | {status} |")
    L.append("\n## Pre-Estimation Checks\n")
    L.append("| Check | Value |")
    L.append("| --- | --- |")
    L.append(f"| Feature rank | {diagnostics['feature_rank']} / "
             f"{diagnostics['num_features']} |")
    L.append(f"| Condition number | {diagnostics['condition_number']:.3g} |")
    L.append(f"| Observed states | {diagnostics['observed_states']} / "
             f"{diagnostics['num_states']} |")
    L.append(f"| State-action coverage | {diagnostics['state_action_coverage']:.3f} |")
    L.append(f"| Minimum action share | {diagnostics['min_action_share']:.3f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
