"""Print the complete TD-CCP qualification evidence and fail on any bad gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "validation" / "results"


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing required result: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def value_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def print_gate(gate: dict[str, Any], failures: list[str], section: str) -> None:
    status = "PASS" if gate["passed"] else "FAIL"
    print(
        f"  {status:<4} {gate['name']}: {value_text(gate['value'])} "
        f"{gate['operator']} {value_text(gate['threshold'])}"
    )
    if not gate["passed"]:
        failures.append(f"{section}: {gate['name']}")


def require(condition: bool, label: str, failures: list[str]) -> None:
    print(f"  {'PASS' if condition else 'FAIL':<4} {label}")
    if not condition:
        failures.append(label)


def paper_report(failures: list[str]) -> None:
    payload = load("tdccp_table_e1.json")
    print("\nPAPER COMPARISON | Adusumilli and Eckardt (2025), Table E.1")
    print("  result: validation/results/tdccp_table_e1.json")
    print(f"  source commit: {payload['git_sha']}")
    print(f"  panels: {payload['completed_replications']} regenerated from official code")
    for mode in ("nonrobust", "robust"):
        result = payload["modes"][mode]
        mean_gap = max(result["mean_absolute_error"])
        sd_gap = max(result["empirical_sd_absolute_error"])
        print(
            f"  {mode:<9} n={result['n_replications']}; "
            f"max mean gap={mean_gap:.3g}; max empirical-SD gap={sd_gap:.3g}; "
            f"4+ significant figures={result['matches_four_significant_figures']}"
        )
    require(payload["exact_replication_passed"], "all 12 published quantities match", failures)


def inference_report(failures: list[str]) -> None:
    payload = load("tdccp_inference.json")
    summary = payload["summary"]
    print("\nINFERENCE | 1,000 independent stationary panels")
    print("  result: validation/results/tdccp_inference.json")
    print(f"  source commit: {payload['provenance']['git_commit']}")
    print(
        f"  usable: {summary['usable_replications']} / "
        f"{summary['completed_replications']} ({summary['usable_rate']:.3f})"
    )
    print("  parameter      std bias    mean SE / empirical SD    coverage    lower    upper")
    for index, name in enumerate(("theta_0", "theta_1", "theta_2")):
        print(
            f"  {name:<10} {summary['standardized_bias'][index]:>9.4f}"
            f" {summary['mean_se_to_empirical_sd'][index]:>25.4f}"
            f" {summary['coverage_95'][index]:>11.3f}"
            f" {summary['lower_tail_miss_rate'][index]:>8.3f}"
            f" {summary['upper_tail_miss_rate'][index]:>8.3f}"
        )
    for gate in summary["gates"]:
        print_gate(gate, failures, "inference")


def bootstrap_report(failures: list[str]) -> None:
    payload = load("tdccp_bootstrap_calibration.json")
    summary = payload["summary"]
    print("\nBOOTSTRAP | 50 panels, 99 whole-trajectory draws per panel")
    print("  result: validation/results/tdccp_bootstrap_calibration.json")
    print(f"  source commit: {payload['provenance']['git_commit']}")
    print(
        f"  usable: {summary['n_usable']} / {summary['n_total']}; "
        f"failed draws: {summary['total_failed_draws']}"
    )
    for name, result in summary["parameters"].items():
        print(
            f"  {name:<18} coverage={result['coverage_95']:.3f}; "
            f"mean width={result['mean_interval_width']:.4f}; "
            f"median width={result['median_interval_width']:.4f}"
        )
    for gate in payload["gates"]:
        if gate.get("enforced", True):
            print_gate(gate, failures, "bootstrap")


def ready_report(failures: list[str]) -> None:
    payload = load("tdccp_ready.json")
    summary = payload["summary"]
    print("\nHELD-OUT AND COUNTERFACTUALS | 20 independent hard-problem panels")
    print("  result: validation/results/tdccp_ready.json")
    print(f"  source commit: {payload['provenance']['git_commit']}")
    print(
        f"  usable={summary['usable_replications']}/{summary['completed_replications']}; "
        f"median relative error={summary['median_relative_parameter_error']:.4f}; "
        f"p90={summary['p90_relative_parameter_error']:.4f}; "
        f"policy TV={summary['mean_policy_tv']:.5f}"
    )
    print(
        f"  held out: excess NLL={summary['mean_excess_negative_log_likelihood']:.6g}; "
        f"excess Brier={summary['mean_excess_brier_score']:.6g}; "
        f"max runtime={summary['max_runtime_seconds']:.2f}s"
    )
    for name, result in summary["counterfactuals"].items():
        print(
            f"  {name:<10} oracle effect={result['oracle_effect_policy_tv']:.5f}; "
            f"policy TV={result['policy_tv']:.5f}; regret={result['regret']:.5f}"
        )
    for gate in summary["gates"]:
        print_gate(gate, failures, "held-out and counterfactual")


def neural_report(failures: list[str]) -> None:
    payload = load("tdccp_neural_avi.json")
    summary = payload["summary"]
    print("\nNEURAL AVI | 30 paired stationary panels")
    print("  result: validation/results/tdccp_neural_avi.json")
    print(f"  source commit: {payload['provenance']['git_commit']}")
    print(
        f"  usable={summary['usable_replications']}/{summary['completed_replications']}; "
        f"max mean relative error={max(summary['mean_locally_robust_relative_error']):.4f}; "
        f"robust bias={summary['locally_robust_aggregate_bias']:.4f}; "
        f"plug-in bias={summary['plugin_aggregate_bias']:.4f}"
    )
    for gate in summary["gates"]:
        print_gate(gate, failures, "neural AVI")


def highdim_report(failures: list[str]) -> None:
    payload = load("tdccp_highdim.json")
    print("\nENCODED STATE | 30 paired seeds and successor-shuffle negative control")
    print("  result: validation/results/tdccp_highdim.json")
    print(f"  source commit: {payload['provenance']['git_commit']}")
    print(f"  K=20 / K=0 mean parameter RMSE: {payload['nuisance_error_ratio']:.4f}")
    print(
        "  shuffled / correct theta_1 mean absolute error: "
        f"{payload['negative_control']['error_ratio']:.4f}"
    )
    for gate in payload["gates"]:
        print_gate(gate, failures, "encoded state")


def serialization_report(failures: list[str]) -> None:
    payload = load("tdccp_serialization.json")
    print("\nWHEEL SERIALIZATION | fresh Python process")
    print("  result: validation/results/tdccp_serialization.json")
    print(f"  source commit: {payload['git_commit']}")
    print(f"  interpreter: {payload['python_executable']}")
    print(f"  module: {payload['econirl_module']}")
    for name, gap in payload["maximum_absolute_gaps"].items():
        print(f"  {name:<26} max absolute gap={gap:.3g}")
    require(payload["status"] == "ready", "serialization status is ready", failures)
    require(payload["fresh_process"], "serialization used a fresh process", failures)
    require(payload["module_outside_checkout"], "module resolved outside checkout", failures)
    require(payload["summary_equal"], "summary round trip is exact", failures)
    require(
        all(gap <= payload["threshold"] for gap in payload["maximum_absolute_gaps"].values()),
        "all serialized outputs meet tolerance",
        failures,
    )


def main() -> int:
    failures: list[str] = []
    print("TD-CCP 0.1.0 QUALIFICATION REPORT")
    paper_report(failures)
    inference_report(failures)
    bootstrap_report(failures)
    ready_report(failures)
    neural_report(failures)
    highdim_report(failures)
    serialization_report(failures)
    print("\nFINAL VERDICT")
    if failures:
        print("  NOT READY")
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1
    print("  READY: every scientific and serialization gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
