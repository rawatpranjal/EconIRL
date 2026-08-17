#!/usr/bin/env python3
"""Print the complete AIRL qualification evidence and fail closed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "validation" / "results"
NOTEBOOK = ROOT / "examples" / "airl" / "airl_applied_workflow.ipynb"
INPUTS = {
    "controlled": RESULTS / "airl_controlled_recovery.json",
    "bootstrap": RESULTS / "airl_bootstrap_calibration.json",
    "taxi": RESULTS / "airl_taxi_transfer.json",
    "serialization": RESULTS / "airl_serialization.json",
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
    for name in ("controlled", "bootstrap", "taxi"):
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
    bootstrap = receipts["bootstrap"]
    if bootstrap.get("mode") != "full":
        failed.append(f"bootstrap:mode={bootstrap.get('mode')}")
    if bootstrap.get("design", {}).get("n_panels") != 20:
        failed.append("bootstrap:n_panels")
    if bootstrap.get("design", {}).get("bootstrap_draws_per_panel") != 19:
        failed.append("bootstrap:draws_per_panel")
    if receipts["controlled"].get("configuration", {}).get("n_replications") != 3:
        failed.append("controlled:n_replications")
    if receipts["taxi"].get("configuration", {}).get("n_replications") != 3:
        failed.append("taxi:n_replications")
    serialization = receipts["serialization"]
    if serialization.get("status") != "passed":
        failed.append(f"serialization:status={serialization.get('status')}")
    if not serialization.get("fresh_process", False):
        failed.append("serialization:fresh_process=false")
    if not serialization.get("module_outside_checkout", False):
        failed.append("serialization:module_outside_checkout=false")
    if not serialization.get("summary_equal", False):
        failed.append("serialization:summary_equal=false")
    if not serialization.get("confidence_intervals_equal", False):
        failed.append("serialization:confidence_intervals_equal=false")
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
    if "Installed package import: True" not in notebook_outputs(notebook):
        failed.append("notebook:not_wheel_owned")
    return failed


def render(receipts: dict[str, dict[str, Any]]) -> str:
    controlled = receipts["controlled"]["summary"]
    bootstrap = receipts["bootstrap"]["summary"]
    taxi = receipts["taxi"]["summary"]
    return "\n".join(
        [
            "AIRL qualification",
            "==================",
            (
                "Controlled recovery: "
                f"{controlled['n_converged']}/{controlled['n_requested']} converged, "
                f"reward NRMSE median={controlled['reward_nrmse']['median']:.6f}, "
                f"policy TV p95={controlled['policy_tv']['p95']:.6f}"
            ),
            (
                "Controlled transfer: "
                f"policy TV p95={controlled['transfer_policy_tv']['p95']:.6f}, "
                f"regret p95={controlled['transfer_regret']['p95']:.6f}"
            ),
            (
                "Trajectory bootstrap: "
                f"{bootstrap['n_usable']}/{bootstrap['n_requested']} usable panels, "
                f"{bootstrap['bootstrap_draws_successful']}/"
                f"{bootstrap['bootstrap_draws_requested']} successful draws"
            ),
            (
                "Bootstrap coverage: "
                f"reward={bootstrap['families']['reward']['coverage']:.3f}, "
                f"policy={bootstrap['families']['policy']['coverage']:.3f}"
            ),
            (
                "Taxi transfer: "
                f"oracle change TV={taxi['oracle_policy_tv']['median']:.6f}, "
                f"fitted transfer TV p95={taxi['transfer_policy_tv']['p95']:.6f}, "
                f"flow regret p95={taxi['transfer_flow_regret']['p95']:.6f}"
            ),
            "Serialization: fresh wheel process with exact supported-output parity",
            "Notebook: all cells executed from the installed wheel with no errors",
            (
                "Paper boundary: generated adversarial recovery and transfer studies, "
                "not an exact replication of Fu et al. Section 7.1"
            ),
        ]
    )


def main() -> int:
    receipts = load()
    failed = failures(receipts)
    if failed:
        print("AIRL qualification failed")
        for failure in failed:
            print(f"- {failure}")
        return 1
    print(render(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
