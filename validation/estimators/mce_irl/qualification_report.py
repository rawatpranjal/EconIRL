#!/usr/bin/env python3
"""Print the scientific MCE-IRL qualification results and fail closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "validation" / "results"
DEFAULT_INPUTS = {
    "known_truth": RESULTS / "mce_irl.json",
    "repeated_inference": RESULTS / "mce_irl_ready.json",
    "bootstrap": RESULTS / "mce_irl_bootstrap_calibration.json",
    "road": RESULTS / "mce_irl_ziebart_synthetic.json",
    "serialization": RESULTS / "mce_irl_serialization.json",
}


def load_receipts(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Load every required receipt, refusing to omit a missing result."""
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qualification receipt(s): " + ", ".join(missing))
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}


def failures(receipts: dict[str, dict[str, Any]]) -> list[str]:
    """Return every failed scientific or operational requirement."""
    failed: list[str] = []
    known_truth = receipts["known_truth"]
    for result in known_truth.get("results", []):
        for gate in result.get("gates", []):
            if not gate.get("passed", False):
                failed.append(f"known_truth:{result.get('cell_id')}:{gate.get('name')}")
    if not known_truth.get("results"):
        failed.append("known_truth:no_results")

    repeated = receipts["repeated_inference"]
    if repeated.get("status") != "ready":
        failed.append(f"repeated_inference:status={repeated.get('status')}")
    failed.extend(
        f"repeated_inference:{gate.get('name')}"
        for gate in repeated.get("gates", [])
        if not gate.get("passed", False)
    )

    bootstrap = receipts["bootstrap"]
    if bootstrap.get("status") != "ready":
        failed.append(f"bootstrap:status={bootstrap.get('status')}")
    failed.extend(
        f"bootstrap:{gate.get('name')}"
        for gate in bootstrap.get("gates", [])
        if gate.get("enforced", False) and not gate.get("passed", False)
    )

    if not receipts["road"].get("passed", False):
        failed.append("road:passed=false")
    if receipts["serialization"].get("status") != "ready":
        failed.append(f"serialization:status={receipts['serialization'].get('status')}")
    return failed


def render(receipts: dict[str, dict[str, Any]]) -> str:
    """Render the quantities a reviewer needs to evaluate the evidence."""
    repeated = receipts["repeated_inference"]["repeated_run_inference"]
    se_check = receipts["repeated_inference"]["standard_error_check"]
    bootstrap = receipts["bootstrap"]["summary"]
    road = receipts["road"]
    primary = receipts["known_truth"]["result"]
    simulation_gates = sum(len(result["gates"]) for result in receipts["known_truth"]["results"])
    lines = [
        "MCE-IRL qualification",
        "=====================",
        f"Known-truth gates: {simulation_gates}/{simulation_gates} passed",
        (
            "Known-truth primary: "
            f"policy TV={primary['metrics']['policy']['tv']:.6f}, "
            f"normalized reward RMSE={primary['metrics']['reward_normalized_rmse']:.6f}"
        ),
        (
            "Repeated inference: "
            f"{repeated['n_successful']}/"
            f"{receipts['repeated_inference']['n_replications']} fits, "
            f"coverage={repeated['coverage']:.3f}, "
            f"bias={repeated['bias']:.6f}, RMSE={repeated['rmse']:.6f}"
        ),
        (
            "Standard errors: "
            f"mean asymptotic/Monte Carlo SD={repeated['mean_se_to_empirical_sd']:.3f}, "
            f"single-panel asymptotic/bootstrap ratio={se_check['ratio']:.3f}"
        ),
        (
            "Trajectory bootstrap: "
            f"{bootstrap['n_usable']}/{bootstrap['n_total']} panels, "
            f"{bootstrap['total_successful_draws']}/"
            f"{bootstrap['total_requested_draws']} draws, "
            f"coverage={bootstrap['coverage_95']:.3f}, "
            f"mean width={bootstrap['mean_interval_width']:.6f}"
        ),
        (
            "Road study: "
            f"distance match={road['synthetic_metrics']['distance_match_percent']:.2f}%, "
            f"routes >=90%={road['synthetic_metrics']['routes_at_least_90_percent']:.2f}%, "
            f"average log probability={road['synthetic_metrics']['average_log_probability']:.2f}"
        ),
        (
            "Counterfactual: max policy change="
            f"{receipts['repeated_inference']['intervention_check']['max_policy_change']:.6f}; "
            "welfare levels withheld"
        ),
        "Serialization: fresh wheel process with exact summary and prediction parity",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="directory containing the five required JSON receipts",
    )
    args = parser.parse_args()
    paths = {name: args.results_dir / path.name for name, path in DEFAULT_INPUTS.items()}
    receipts = load_receipts(paths)
    failed = failures(receipts)
    if failed:
        print("MCE-IRL qualification failed")
        for failure in failed:
            print(f"- {failure}")
        return 1
    print(render(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
