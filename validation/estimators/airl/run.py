#!/usr/bin/env python3
"""Generate AIRL known-truth validation artifacts.

AIRL is evaluated on a no-anchor state-only deterministic surface that matches
the paper's positive-identification side conditions as closely as the package
harness allows, plus an anchored action-dependent diagnostic. The original
condition cell is the paper-side validation target; the anchored cell records
the out-of-scope failure.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "airl.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_airl_primer_known_truth")
PAPER_CELL_ID = "airl_paper_identification"
ANCHORED_CELL_ID = "airl_anchor_action_dependent"
PRIMARY_CELL_ID = PAPER_CELL_ID
DEFAULT_CELL_IDS = (PAPER_CELL_ID, ANCHORED_CELL_ID)
ESTIMATOR = "AIRL"
CELL_ROLES = {
    PAPER_CELL_ID: "original-conditions validation",
    ANCHORED_CELL_ID: "anchored diagnostic",
}
CELL_LABELS = {
    PAPER_CELL_ID: "Original AIRL conditions",
    ANCHORED_CELL_ID: "Anchored action-dependent reward",
}

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validation_display import (  # noqa: E402
    validation_context,
    validation_display_name,
    validation_role,
)
from validation.known_truth import (  # noqa: E402
    KnownTruthCell,
    KnownTruthDGPConfig,
    RecoveryGateFailure,
    SimulationConfig,
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    stable_hash,
    write_json,
)

AIRL_ANCHORED_CELL = KnownTruthCell(
    cell_id=ANCHORED_CELL_ID,
    dgp_config=KnownTruthDGPConfig(
        state_mode="low_dim",
        reward_mode="action_dependent",
        reward_dim="low",
        heterogeneity="none",
        num_regular_states=20,
        transition_noise=0.0,
        seed=744,
    ),
    simulation_config=SimulationConfig(
        n_individuals=300,
        n_periods=80,
        seed=744,
        show_progress=False,
    ),
    description=(
        "AIRL anchored known-truth cell: action-dependent reward with the "
        "exit action pinned to zero and the absorbing-state reward row pinned "
        "to zero, matching the AIRL-Het gauge."
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help=(
            "Known-truth cell to run. May be repeated. Defaults to the AIRL "
            "state-only and anchored action-dependent diagnostic cells."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=False)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()

    cell_ids = args.cell_id if args.cell_id is not None else list(DEFAULT_CELL_IDS)

    print("AIRL primer: running known-truth validation")
    print(f"  cells: {', '.join(cell_ids)}")
    print(f"  estimator: {ESTIMATOR}")
    print(f"  output_dir: {args.output_dir}")

    records = [run_validation_cell(cell_id, args) for cell_id in cell_ids]

    write_json(JSON_OUT, compact_payload(records))

    failed_records: list[tuple[str, list[Any]]] = []
    total_gates = 0
    failed_gates = 0
    for record in records:
        payload = record["payload"]
        failed = [gate for gate in payload["gates"] if not gate.passed]
        total_gates += len(payload["gates"])
        failed_gates += len(failed)
        if failed:
            failed_records.append((payload["cell"].cell_id, failed))
        print(f"  result ({payload['cell'].cell_id}): {record['run_dir'] / 'result.json'}")
        print(
            f"  hard gates ({payload['cell'].cell_id}): "
            f"{len(payload['gates']) - len(failed)} pass, {len(failed)} fail"
        )

    print(f"  wrote: {JSON_OUT}")
    print(f"  hard gates total: {total_gates - failed_gates} pass, {failed_gates} fail")

    if args.enforce_gates and failed_records:
        details = []
        for cell_id, failed in failed_records:
            details.append(
                f"{cell_id}: "
                + ", ".join(
                    f"{gate.name}={gate.value} {gate.operator} {gate.threshold}"
                    for gate in failed
                )
            )
        raise RecoveryGateFailure("; ".join(details))


def run_validation_cell(cell_id: str, args: argparse.Namespace) -> dict[str, Any]:
    cell = _get_airl_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = replace(cell.simulation_config, show_progress=args.show_progress)
    panel = simulate_known_truth_panel(dgp, simulation_config)
    main_run = run_estimator(
        ESTIMATOR,
        dgp,
        panel,
        smoke=False,
        verbose=args.verbose,
        enforce_gates=False,
    )

    payload = {
        "cell": cell,
        "simulation": simulation_config,
        "estimator": ESTIMATOR,
        "diagnostics": main_run.diagnostics,
        "compatibility": main_run.compatibility,
        "summary": main_run.summary,
        "metrics": main_run.metrics,
        "gates": main_run.gates,
    }
    run_hash = stable_hash(
        {
            "cell": cell.dgp_config.to_dict(),
            "simulation": simulation_config,
            "estimator": ESTIMATOR,
            "primer": "airl",
            "enforce_gates": bool(args.enforce_gates),
        }
    )
    run_dir = args.output_dir / f"{cell.cell_id}_airl_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "dgp": dgp, "run_dir": run_dir}


def _get_airl_cell(cell_id: str) -> KnownTruthCell:
    if cell_id == AIRL_ANCHORED_CELL.cell_id:
        return AIRL_ANCHORED_CELL
    return get_cell(cell_id)


def render_results_tex(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    add = lines.append

    add("% Auto-generated by airl_run.py. Do not edit by hand.")
    cell_list = ", ".join(record["payload"]["cell"].cell_id for record in records)
    add("% Known-truth DGP cells: " + cell_list)
    add("")
    add(r"\subsection{Known-Truth Validation Results}")
    add(
        "The current AIRL path is evaluated on two surfaces: a no-anchor "
        "state-only deterministic-transition DGP that matches the original "
        "AIRL identification conditions, and an action-dependent DGP with the "
        "same exit-action/absorbing-state anchor used by AIRL-Het. The "
        "original-conditions DGP passes all hard gates after learning the AIRL "
        "shaping potential inside the discriminator. The anchored "
        "action-dependent DGP remains a diagnostic failure."
    )
    add("")
    add(r"\subsection{Validation Design Context}")
    add(validation_context([record["payload"]["cell"].cell_id for record in records]))
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{AIRL known-truth validation cells.}")
    add(r"\begin{tabular}{llrrrr}")
    add(r"\toprule")
    add(r"Cell & Role & States & Actions & Individuals & Periods \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        dgp = record["dgp"]
        cell_id = payload["cell"].cell_id
        sim = payload["simulation"]
        add(
            f"{tex_text(validation_display_name(cell_id))} & "
            f"{tex_text(validation_role(cell_id, CELL_ROLES.get(cell_id, 'diagnostic')))} & "
            f"{dgp.problem.num_states} & {dgp.problem.num_actions} & "
            f"{sim.n_individuals:,} & {sim.n_periods} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{AIRL recovery metrics. Lower is better.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{lrrrrrr}")
    add(r"\toprule")
    add(r"Cell & Gates & Reward nRMSE & Policy TV & Value nRMSE & Q nRMSE & Type A/B/C regret \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        cell_id = payload["cell"].cell_id
        gates = payload["gates"]
        passed = sum(gate.passed for gate in gates)
        regrets = "/".join(
            fmt(counterfactual_value(record, kind), 4)
            for kind in ("type_a", "type_b", "type_c")
        )
        add(
            f"{tex_text(validation_display_name(cell_id))} & "
            f"{passed}/{len(gates)} & "
            f"{fmt(metric_value(record, 'reward_normalized_rmse'), 4)} & "
            f"{fmt(metric_value(record, 'policy_tv'), 4)} & "
            f"{fmt(metric_value(record, 'value_normalized_rmse'), 4)} & "
            f"{fmt(metric_value(record, 'q_normalized_rmse'), 4)} & "
            f"{regrets} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"}")
    add(r"\end{table}")
    add("")

    for record in records:
        payload = record["payload"]
        failed = [gate for gate in payload["gates"] if not gate.passed]
        if failed:
            add(
                f"The {tex_text(validation_display_name(payload['cell'].cell_id))} "
                f"fails {len(failed)}/{len(payload['gates'])} hard gates."
            )
        else:
            add(
                f"The {tex_text(validation_display_name(payload['cell'].cell_id))} "
                f"passes {len(payload['gates'])}/{len(payload['gates'])} hard gates."
            )
    add("")
    return "\n".join(lines)


def metric_value(record: dict[str, Any], key: str) -> float | None:
    metrics = record["payload"]["metrics"]
    if key == "policy_tv":
        return metrics["policy"].tv
    return metrics.get(key)


def counterfactual_value(record: dict[str, Any], kind: str) -> float | None:
    return record["payload"]["metrics"]["counterfactuals"][kind].regret


def compact_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "estimator": ESTIMATOR,
        "primary_cell_id": PRIMARY_CELL_ID,
        "status": "partial",
        "interpretation": (
            "AIRL is validated on the no-anchor deterministic state-only cell "
            "matching the original identification conditions. The anchored "
            "action-dependent cell remains a failed diagnostic."
        ),
        "results": [compact_cell_payload(record["payload"]) for record in records],
    }


def compact_cell_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    metadata = summary.metadata
    return {
        "cell_id": payload["cell"].cell_id,
        "estimator": payload["estimator"],
        "simulation": payload["simulation"],
        "compatibility": payload["compatibility"],
        "diagnostics": payload["diagnostics"],
        "summary": {
            "log_likelihood": summary.log_likelihood,
            "converged": summary.converged,
            "num_iterations": summary.num_iterations,
            "num_observations": summary.num_observations,
            "estimation_time": summary.estimation_time,
            "convergence_message": summary.convergence_message,
            "metadata": {
                "reward_type": metadata.get("reward_type"),
                "reward_arg": metadata.get("reward_arg"),
                "anchor_action": metadata.get("anchor_action"),
                "absorbing_state": metadata.get("absorbing_state"),
                "use_shaping": metadata.get("use_shaping"),
                "learned_shaping": metadata.get("learned_shaping"),
                "generator_reward": metadata.get("generator_reward"),
                "min_rounds": metadata.get("min_rounds"),
                "final_disc_loss": metadata.get("final_disc_loss"),
            },
        },
        "metrics": payload["metrics"],
        "gates": payload["gates"],
    }


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "---"
    number = float(value)
    if not math.isfinite(number):
        return "---"
    return f"{number:.{digits}f}"


def tex_text(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


if __name__ == "__main__":
    main()
