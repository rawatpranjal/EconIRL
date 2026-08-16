#!/usr/bin/env python3
"""Print the Neural MCE-IRL qualification results and fail closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "validation" / "results"
NOTEBOOK = ROOT / "examples" / "neural-mce-irl" / "neural_mce_irl_applied_workflow.ipynb"
DEFAULT_INPUTS = {
    "known_truth": RESULTS / "deep_mce_irl.json",
    "stability": RESULTS / "deep_mce_irl_ready.json",
    "wulfmeier": RESULTS / "deep_mce_irl_wulfmeier.json",
    "bootstrap": RESULTS / "deep_mce_irl_bootstrap_calibration.json",
    "serialization": RESULTS / "deep_mce_irl_serialization.json",
}


def load_receipts(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Load every required receipt, refusing to omit a missing result."""
    missing = [str(path) for path in (*paths.values(), NOTEBOOK) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qualification receipt(s): " + ", ".join(missing))
    receipts = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    receipts["notebook"] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return receipts


def _notebook_outputs(notebook: dict[str, Any]) -> str:
    return "\n".join(
        "".join(output.get("text", []))
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )


def failures(receipts: dict[str, dict[str, Any]]) -> list[str]:
    """Return every failed scientific or operational requirement."""
    failed: list[str] = []
    known_truth = receipts["known_truth"]
    records = known_truth.get("records", [])
    if not records:
        failed.append("known_truth:no_records")
    failed.extend(
        f"known_truth:{record.get('cell', {}).get('cell_id')}:{gate.get('name')}"
        for record in records
        for gate in record.get("gates", [])
        if not gate.get("passed", False)
    )
    for name in ("stability", "wulfmeier", "bootstrap"):
        receipt = receipts[name]
        if receipt.get("status") != "ready":
            failed.append(f"{name}:status={receipt.get('status')}")
        failed.extend(
            f"{name}:{check.get('name')}"
            for check in receipt.get("checks", [])
            if not check.get("passed", False)
        )
    bootstrap = receipts["bootstrap"]
    bootstrap_design = bootstrap.get("design", {})
    bootstrap_summary = bootstrap.get("summary", {})
    required_bootstrap_fields = {
        "mode": (bootstrap.get("mode"), "full"),
        "full_gates_evaluated": (bootstrap.get("full_gates_evaluated"), True),
        "paper_replication": (bootstrap.get("paper_replication"), False),
        "n_panels": (bootstrap_design.get("n_panels"), 50),
        "draws_per_panel": (bootstrap_design.get("bootstrap_draws_per_panel"), 99),
        "individuals_per_panel": (bootstrap_design.get("individuals_per_panel"), 300),
        "periods_per_individual": (bootstrap_design.get("periods_per_individual"), 1),
        "bootstrap_unit": (
            bootstrap_design.get("bootstrap_unit"),
            "individual_trajectory",
        ),
        "target_states": (bootstrap_design.get("target_states"), [0, 1, 2]),
        "requested_panels": (bootstrap_summary.get("n_requested"), 50),
        "requested_draws": (bootstrap_summary.get("bootstrap_draws_requested"), 4_950),
    }
    failed.extend(
        f"bootstrap:{name}={actual}"
        for name, (actual, expected) in required_bootstrap_fields.items()
        if actual != expected
    )
    serialization = receipts["serialization"]
    if serialization.get("status") != "passed":
        failed.append(f"serialization:status={serialization.get('status')}")
    if not serialization.get("fresh_process", False):
        failed.append("serialization:fresh_process=false")
    if not serialization.get("module_outside_checkout", False):
        failed.append("serialization:module_outside_checkout=false")
    if serialization.get("econirl_version") != "0.0.10":
        failed.append(f"serialization:version={serialization.get('econirl_version')}")
    for field in (
        "summary_equal",
        "confidence_intervals_equal",
        "counterfactual_intervals_equal",
    ):
        if not serialization.get(field, False):
            failed.append(f"serialization:{field}=false")
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
    if "Installed package import: True" not in _notebook_outputs(notebook):
        failed.append("notebook:not_wheel_owned")
    return failed


def render(receipts: dict[str, dict[str, Any]]) -> str:
    """Render the quantities a reviewer needs to evaluate the evidence."""
    known_truth = receipts["known_truth"]
    primary = next(
        record
        for record in known_truth["records"]
        if record["cell"]["cell_id"] == known_truth["primary_cell"]
    )
    known_truth_gates = sum(len(record["gates"]) for record in known_truth["records"])
    stability = receipts["stability"]["summary"]
    wulfmeier = receipts["wulfmeier"]["summary"]
    bootstrap = receipts["bootstrap"]["summary"]
    family_coverages = bootstrap["families"]
    lines = [
        "Neural MCE-IRL qualification",
        "============================",
        f"Controlled recovery gates: {known_truth_gates}/{known_truth_gates} passed",
        (
            "Controlled primary: "
            f"policy TV={primary['metrics']['policy']['tv']:.6f}, "
            f"normalized reward RMSE={primary['metrics']['reward_normalized_rmse']:.6f}"
        ),
        (
            "Repeated stability: "
            f"{stability['n_successful']}/{stability['n_requested']} fits, "
            f"reward RMSE median={stability['reward_nrmse']['median']:.6f}, "
            f"policy TV median={stability['policy_tv']['median']:.6f}"
        ),
        (
            "Wulfmeier-shaped study: "
            f"{wulfmeier['n_successful']}/{wulfmeier['n_requested']} finite fits, "
            f"{wulfmeier['n_converged']} met strict stopping rules"
        ),
        (
            "Trajectory bootstrap: "
            f"{bootstrap['n_usable']}/{bootstrap['n_requested']} usable panels, "
            f"{bootstrap['bootstrap_draws_successful']}/"
            f"{bootstrap['bootstrap_draws_requested']} successful draws"
        ),
        (
            "Bootstrap coverage: "
            + ", ".join(
                f"{name}={summary['coverage']:.3f}" for name, summary in family_coverages.items()
            )
        ),
        "Counterfactual uncertainty: reward, transition, and action-availability families",
        "Serialization: fresh wheel process with exact summary and supported-output parity",
        "Notebook: all cells executed from the installed wheel with no errors",
        "Paper boundary: generated comparison, not a paper-number replication",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    paths = {name: args.results_dir / path.name for name, path in DEFAULT_INPUTS.items()}
    receipts = load_receipts(paths)
    failed = failures(receipts)
    if failed:
        print("Neural MCE-IRL qualification failed")
        for failure in failed:
            print(f"- {failure}")
        return 1
    print(render(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
