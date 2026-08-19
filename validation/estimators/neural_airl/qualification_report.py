#!/usr/bin/env python3
"""Print the complete NeuralAIRL qualification evidence and fail closed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "validation" / "results"
NOTEBOOK = ROOT / "examples" / "neural-airl" / "neural_airl_applied_workflow.ipynb"
INPUTS = {
    "nonlinear": RESULTS / "neural_airl_nonlinear.json",
    "bootstrap": RESULTS / "neural_airl_bootstrap.json",
    "serialization": RESULTS / "neural_airl_serialization.json",
}


def load() -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in (*INPUTS.values(), NOTEBOOK) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qualification receipt(s): " + ", ".join(missing))
    receipts = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in INPUTS.items()}
    receipts["notebook"] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return receipts


def notebook_outputs(notebook: dict[str, Any]) -> str:
    return "\n".join(
        "".join(output.get("text", []))
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )


def failures(receipts: dict[str, dict[str, Any]]) -> list[str]:
    failed: list[str] = []
    for name in ("nonlinear", "bootstrap"):
        receipt = receipts[name]
        if receipt.get("status") != "ready":
            failed.append(f"{name}:status={receipt.get('status')}")
        failed.extend(
            f"{name}:{check.get('name')}"
            for check in receipt.get("checks", [])
            if not check.get("passed", False)
        )
        if receipt.get("paper_replication") is not False:
            failed.append(f"{name}:paper_replication_boundary")

    nonlinear = receipts["nonlinear"]
    if nonlinear.get("configuration", {}).get("n_replications") != 10:
        failed.append("nonlinear:n_replications")

    bootstrap = receipts["bootstrap"]
    if bootstrap.get("mode") != "full":
        failed.append(f"bootstrap:mode={bootstrap.get('mode')}")
    if bootstrap.get("design", {}).get("n_panels") != 10:
        failed.append("bootstrap:n_panels")
    if bootstrap.get("design", {}).get("bootstrap_draws_per_panel") != 9:
        failed.append("bootstrap:draws_per_panel")
    if bootstrap.get("design", {}).get("dynamics") != "deterministic_decomposable":
        failed.append("bootstrap:dynamics")
    if bootstrap.get("design", {}).get("interval_method") != ("normal_bootstrap_standard_error"):
        failed.append("bootstrap:interval_method")

    serialization = receipts["serialization"]
    if serialization.get("status") != "passed":
        failed.append(f"serialization:status={serialization.get('status')}")
    for key in (
        "fresh_process",
        "module_outside_checkout",
        "summary_equal",
        "confidence_intervals_equal",
    ):
        if not serialization.get(key, False):
            failed.append(f"serialization:{key}=false")
    failed.extend(
        f"serialization:{name}={gap}"
        for name, gap in serialization.get("maximum_absolute_gaps", {}).items()
        if float(gap) > 1e-12
    )

    notebook = receipts["notebook"]
    if any(
        cell.get("execution_count") is None
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ):
        failed.append("notebook:unexecuted_code_cell")
    if any(
        output.get("output_type") == "error"
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    ):
        failed.append("notebook:error_output")
    outputs = notebook_outputs(notebook)
    if "Installed package import: True" not in outputs:
        failed.append("notebook:not_wheel_owned")
    if "Standalone estimator: True" not in outputs:
        failed.append("notebook:not_standalone")
    return failed


def render(receipts: dict[str, dict[str, Any]]) -> str:
    nonlinear = receipts["nonlinear"]["summary"]
    bootstrap = receipts["bootstrap"]["summary"]
    return "\n".join(
        [
            "NeuralAIRL qualification",
            "========================",
            (
                "Nonlinear recovery: "
                f"{nonlinear['n_converged']}/{nonlinear['n_requested']} converged, "
                f"{nonlinear['n_improved']}/{nonlinear['n_requested']} beat linear AIRL"
            ),
            (
                "Behavioral recovery: "
                f"policy TV p95={nonlinear['neural_policy_tv']['p95']:.6f}, "
                f"mean TV ratio={nonlinear['mean_tv_ratio_to_linear_airl']:.6f}"
            ),
            (
                "Reward and transfer: "
                f"reward NRMSE median={nonlinear['affine_reward_nrmse']['median']:.6f}, "
                f"transfer policy TV p95={nonlinear['transfer_policy_tv']['p95']:.6f}"
            ),
            (
                "Trajectory bootstrap: "
                f"{bootstrap['n_converged']}/{bootstrap['n_requested']} panels, "
                f"{bootstrap['bootstrap_draws_successful']}/"
                f"{bootstrap['bootstrap_draws_requested']} draws"
            ),
            (
                "Bootstrap coverage: "
                f"reward={bootstrap['reward_coverage']:.3f}, "
                f"policy={bootstrap['policy_coverage']:.3f}"
            ),
            "Serialization: fresh wheel process with exact supported-output parity",
            "Notebook: all cells executed from the installed wheel with no errors",
            (
                "Paper boundary: generated finite-state nonlinear recovery evidence, "
                "not a replication of Fu et al. continuous-control results"
            ),
        ]
    )


def main() -> int:
    receipts = load()
    failed = failures(receipts)
    if failed:
        print("NeuralAIRL qualification failed")
        for failure in failed:
            print(f"- {failure}")
        return 1
    print(render(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
