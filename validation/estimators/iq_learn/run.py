#!/usr/bin/env python3
"""Generate IQ-Learn strict known-truth validation artifacts.

IQ-Learn is treated as counterfactual-valid only when the Bellman-implied
reward table, its projection into the true feature basis, value function,
Q-function, policy, support coverage, and Type A/B/C counterfactual regrets all
pass hard structural gates.
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
JSON_OUT = ROOT / "validation" / "results" / "iq_learn.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_iq_learn_primer_known_truth")
PRIMARY_CELL_ID = "canonical_low_action"
HIGH_ACTION_CELL_ID = "canonical_high_action"
STATE_ONLY_CELL_ID = "canonical_low_state_only"
DEFAULT_CELL_IDS = (PRIMARY_CELL_ID, HIGH_ACTION_CELL_ID, STATE_ONLY_CELL_ID)
ESTIMATOR = "IQ-Learn"
CELL_ROLES = {
    PRIMARY_CELL_ID: "baseline structural",
    HIGH_ACTION_CELL_ID: "high-dimensional stress",
    STATE_ONLY_CELL_ID: "negative control",
}
CELL_LABELS = {
    PRIMARY_CELL_ID: "Canonical low-dimensional",
    HIGH_ACTION_CELL_ID: "Canonical high-dimensional",
    STATE_ONLY_CELL_ID: "Canonical state-only",
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
    RecoveryGateFailure,
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    stable_hash,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help="Known-truth cell to run. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=False)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()

    cell_ids = args.cell_id if args.cell_id is not None else list(DEFAULT_CELL_IDS)

    print("IQ-Learn primer: running known-truth validation")
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
            f"  strict structural gates ({payload['cell'].cell_id}): "
            f"{len(payload['gates']) - len(failed)} pass, {len(failed)} fail"
        )

    print(f"  wrote: {JSON_OUT}")
    print(
        "  strict structural gates total: "
        f"{total_gates - failed_gates} pass, {failed_gates} fail"
    )

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
    cell = get_cell(cell_id)
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
            "primer": "iq_learn",
            "enforce_gates": bool(args.enforce_gates),
        }
    )
    run_dir = args.output_dir / f"{cell.cell_id}_iq_learn_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "dgp": dgp, "run_dir": run_dir}


def render_results_tex(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    add = lines.append

    add("% Auto-generated by iq_learn_run.py. Do not edit by hand.")
    cell_list = ", ".join(record["payload"]["cell"].cell_id for record in records)
    add("% Known-truth DGP cells: " + cell_list)
    add("")
    add(r"\subsection{Known-Truth Validation Results}")
    add(
        "IQ-Learn is treated as Type A/B/C counterfactual-valid only if every "
        "structural gate passes: optimizer convergence, expert support "
        "coverage, policy distance, raw Bellman reward recovery, projected "
        "reward recovery in the true feature basis, value recovery, Q recovery, "
        "and counterfactual regret."
    )
    add("")
    add(r"\subsection{Validation Design Context}")
    add(validation_context([record["payload"]["cell"].cell_id for record in records]))
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{IQ-Learn known-truth validation design.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{llrrrrr}")
    add(r"\toprule")
    add(r"Cell & Role & States & Actions & Reward features & Individuals & Periods \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        dgp = record["dgp"]
        cell_id = payload["cell"].cell_id
        sim = payload["simulation"]
        add(
            f"{tex_text(validation_display_name(cell_id))} & "
            f"{tex_text(validation_role(cell_id, CELL_ROLES.get(cell_id, 'validation')))} & "
            f"{dgp.problem.num_states} & {dgp.problem.num_actions} & "
            f"{dgp.feature_matrix.shape[-1]} & {sim.n_individuals:,} & "
            f"{sim.n_periods} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{IQ-Learn validation design and behavioral diagnostics. Lower is better.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{llrrrr}")
    add(r"\toprule")
    add(r"DGP & Q class & Gates passed & State coverage & State-action coverage & Policy TV \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        cell_id = payload["cell"].cell_id
        gates = payload["gates"]
        passed = sum(gate.passed for gate in gates)
        add(
            f"{tex_text(validation_display_name(cell_id))} & "
            f"{tex_text(metadata_value(record, 'q_type') or '---')} & "
            f"{passed}/{len(gates)} & "
            f"{fmt(metadata_value(record, 'expert_state_coverage'), 4)} & "
            f"{fmt(metadata_value(record, 'expert_state_action_coverage'), 4)} & "
            f"{fmt(metric_value(record, 'policy_tv'), 4)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{IQ-Learn structural recovery and counterfactual diagnostics. Lower is better.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{lrrrrrrr}")
    add(r"\toprule")
    add(
        r"DGP & Raw reward nRMSE & Projected reward nRMSE & Value nRMSE & "
        r"Q nRMSE & Type A regret & Type B regret & Type C regret \\"
    )
    add(r"\midrule")
    for record in records:
        cell_id = record["payload"]["cell"].cell_id
        add(
            f"{tex_text(validation_display_name(cell_id))} & "
            f"{fmt(metric_value(record, 'raw_bellman_reward_normalized_rmse'), 4)} & "
            f"{fmt(metric_value(record, 'projected_reward_normalized_rmse'), 4)} & "
            f"{fmt(metric_value(record, 'value_normalized_rmse'), 4)} & "
            f"{fmt(metric_value(record, 'q_normalized_rmse'), 4)} & "
            f"{fmt(counterfactual_value(record, 'type_a'), 4)} & "
            f"{fmt(counterfactual_value(record, 'type_b'), 4)} & "
            f"{fmt(counterfactual_value(record, 'type_c'), 4)} \\\\"
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
                "fails "
                f"{len(failed)}/{len(payload['gates'])} strict structural "
                "gates, so it is not treated as counterfactual-valid on "
                "this DGP."
            )
        else:
            add(
                f"The {tex_text(validation_display_name(payload['cell'].cell_id))} "
                "passes "
                f"{len(payload['gates'])}/{len(payload['gates'])} strict "
                "structural gates and is treated as counterfactual-valid "
                "for the tested Type A/B/C interventions."
            )
    add("")
    return "\n".join(lines)


def metric_value(record: dict[str, Any], key: str) -> float | None:
    metrics = record["payload"]["metrics"]
    if key == "policy_tv":
        return metrics["policy"].tv
    return metrics.get(key)


def metadata_value(record: dict[str, Any], key: str) -> float | None:
    return record["payload"]["summary"].metadata.get(key)


def counterfactual_value(record: dict[str, Any], kind: str) -> float | None:
    return record["payload"]["metrics"]["counterfactuals"][kind].regret


def compact_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_pass = all(
        all(gate.passed for gate in record["payload"]["gates"])
        for record in records
    )
    return {
        "estimator": ESTIMATOR,
        "primary_cell_id": PRIMARY_CELL_ID,
        "status": (
            "strict_structural_counterfactual_pass"
            if all_pass
            else "strict_structural_counterfactual_fail"
        ),
        "counterfactual_valid_certified": all_pass,
        "interpretation": (
            "IQ-Learn is treated as counterfactual-valid only on cells where "
            "all structural gates pass. Low policy distance or low "
            "counterfactual regret alone is insufficient."
        ),
        "results": [compact_cell_payload(record["payload"]) for record in records],
    }


def compact_cell_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    metadata = summary.metadata
    gates = payload["gates"]
    certified = all(gate.passed for gate in gates)
    metrics = payload["metrics"]
    return {
        "cell_id": payload["cell"].cell_id,
        "estimator": payload["estimator"],
        "counterfactual_valid_certified": certified,
        "failed_gates": [gate.name for gate in gates if not gate.passed],
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
                "q_type": metadata.get("q_type"),
                "divergence": metadata.get("divergence"),
                "alpha": metadata.get("alpha"),
                "final_objective": metadata.get("final_objective"),
                "expert_state_coverage": metadata.get("expert_state_coverage"),
                "expert_state_action_coverage": metadata.get(
                    "expert_state_action_coverage"
                ),
            },
        },
        "metric_groups": {
            "imitation_policy": {
                "policy_tv": metric_value_from_metrics(metrics, "policy_tv"),
                "policy_kl": (
                    metrics["policy"].kl
                    if metrics.get("policy") is not None
                    else None
                ),
            },
            "reward_table_recovery": {
                "raw_bellman_reward_rmse": metrics.get("raw_bellman_reward_rmse"),
                "raw_bellman_reward_normalized_rmse": metrics.get(
                    "raw_bellman_reward_normalized_rmse"
                ),
            },
            "projected_structural_reward_recovery": {
                "projected_reward_rmse": metrics.get("projected_reward_rmse"),
                "projected_reward_normalized_rmse": metrics.get(
                    "projected_reward_normalized_rmse"
                ),
            },
            "value_q_recovery": {
                "value_normalized_rmse": metrics.get("value_normalized_rmse"),
                "q_normalized_rmse": metrics.get("q_normalized_rmse"),
            },
            "counterfactual_recovery": {
                kind: cf.regret
                for kind, cf in metrics["counterfactuals"].items()
            },
        },
        "metrics": payload["metrics"],
        "gates": gates,
    }


def metric_value_from_metrics(metrics: dict[str, Any], key: str) -> float | None:
    if key == "policy_tv":
        policy = metrics.get("policy")
        return policy.tv if policy is not None else None
    return metrics.get(key)


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
