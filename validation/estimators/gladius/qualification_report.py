#!/usr/bin/env python3
"""Print complete GLADIUS qualification evidence and fail closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "validation" / "results"
NOTEBOOK = ROOT / "examples" / "gladius" / "gladius_applied_workflow.ipynb"
DEFAULT_INPUTS = {
    "known_truth": RESULTS / "gladius.json",
    "bootstrap": RESULTS / "gladius_bootstrap_calibration.json",
    "paper": RESULTS / "gladius_paper_table2.json",
    "serialization": RESULTS / "gladius_serialization.json",
}


def load_receipts(
    paths: dict[str, Path], notebook_path: Path = NOTEBOOK
) -> dict[str, dict[str, Any]]:
    """Load every required receipt, refusing to omit a missing result."""
    missing = [str(path) for path in (*paths.values(), notebook_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qualification receipt(s): " + ", ".join(missing))
    receipts = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    receipts["notebook"] = json.loads(notebook_path.read_text(encoding="utf-8"))
    return receipts


def notebook_outputs(notebook: dict[str, Any]) -> str:
    return "\n".join(
        "".join(output.get("text", []))
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )


def failures(receipts: dict[str, dict[str, Any]]) -> list[str]:
    """Return every failed scientific or release requirement."""
    failed: list[str] = []
    known_truth = receipts["known_truth"]
    if known_truth.get("status") != "strict_structural_counterfactual_pass":
        failed.append(f"known_truth:status={known_truth.get('status')}")
    known_gates = known_truth.get("gates", [])
    if len(known_gates) != 12:
        failed.append(f"known_truth:gate_count={len(known_gates)}")
    failed.extend(
        f"known_truth:{gate.get('name')}" for gate in known_gates if not gate.get("passed", False)
    )

    bootstrap = receipts["bootstrap"]
    if bootstrap.get("design", {}).get("panels") != 20:
        failed.append("bootstrap:panels")
    if bootstrap.get("design", {}).get("draws_per_panel") != 19:
        failed.append("bootstrap:draws_per_panel")
    failed.extend(
        f"bootstrap:{name}" for name, passed in bootstrap.get("gates", {}).items() if not passed
    )
    if not bootstrap.get("all_passed", False):
        failed.append("bootstrap:all_passed=false")

    paper = receipts["paper"]
    if len(paper.get("cells", [])) != 6:
        failed.append("paper:cell_count")
    failed.extend(f"paper:{name}" for name, passed in paper.get("gates", {}).items() if not passed)
    if not paper.get("all_passed", False):
        failed.append("paper:all_passed=false")
    boundary = str(paper.get("selection_boundary", ""))
    if "simulation-only" not in boundary or "not used by public fit" not in boundary:
        failed.append("paper:oracle_selection_boundary")

    serialization = receipts["serialization"]
    if serialization.get("status") != "ready":
        failed.append(f"serialization:status={serialization.get('status')}")
    for key in (
        "fresh_process",
        "module_outside_checkout",
        "wheel_origin_required",
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
    if "Installed package import: True" not in notebook_outputs(notebook):
        failed.append("notebook:not_wheel_owned")
    return failed


def render(receipts: dict[str, dict[str, Any]]) -> str:
    known = receipts["known_truth"]
    bootstrap = receipts["bootstrap"]
    paper = receipts["paper"]
    metrics = known["metrics"]
    cells = ", ".join(f"N={cell['n_traj']}: {cell['mean_mape']:.3f}%" for cell in paper["cells"])
    return "\n".join(
        [
            "GLADIUS qualification",
            "=====================",
            "Known truth: 12/12 structural and counterfactual gates passed",
            (
                "Known-truth recovery: "
                f"raw reward NRMSE={metrics['raw_bellman_reward_normalized_rmse']:.6f}, "
                f"Q NRMSE={metrics['q_normalized_rmse']:.6f}, "
                f"value NRMSE={metrics['value_normalized_rmse']:.6f}"
            ),
            (
                "Trajectory bootstrap: "
                f"{bootstrap['usable_panels']}/{bootstrap['design']['panels']} usable panels, "
                f"reward coverage={bootstrap['reward']['coverage']:.3f}, "
                f"policy coverage={bootstrap['policy']['coverage']:.3f}"
            ),
            f"Paper Table 2 (20 reps each): {cells}",
            "Serialization: fresh wheel process with exact supported-output parity",
            "Notebook: all cells executed from the installed wheel with no errors",
            (
                "Oracle boundary: the paper replication matches the authors' simulation-only "
                "true-held-out-MAPE epoch rule; the public estimator never uses it"
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    args = parser.parse_args()
    paths = {name: args.results_dir / path.name for name, path in DEFAULT_INPUTS.items()}
    receipts = load_receipts(paths, args.notebook)
    failed = failures(receipts)
    if failed:
        print("GLADIUS qualification failed")
        for failure in failed:
            print(f"- {failure}")
        return 1
    print(render(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
