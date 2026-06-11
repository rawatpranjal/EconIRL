#!/usr/bin/env python3
"""Generate AIRL-Het known-truth validation artifacts.

AIRL-Het is evaluated on an anchored two-segment serialized-content known-truth
cell with repeated books per user. The artifact reports segment separation and
structural recovery gates directly.
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
JSON_OUT = ROOT / "validation" / "results" / "aairl.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_airl_het_primer_known_truth")
PRIMARY_CELL_ID = "airl_het_paper_identification"
DEFAULT_CELL_IDS = (PRIMARY_CELL_ID,)
ESTIMATOR = "AIRL-Het"
CELL_LABELS = {
    PRIMARY_CELL_ID: "Serialized content",
    "canonical_latent_segments": "Canonical latent segments",
}
CELL_ROLES = {
    PRIMARY_CELL_ID: "paper-style validation",
    "canonical_latent_segments": "large diagnostic",
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
        help=(
            "Known-truth cell to run. May be repeated. Defaults to the "
            "anchored AIRL-Het paper-style validation cell."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=False)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()

    cell_ids = args.cell_id if args.cell_id is not None else list(DEFAULT_CELL_IDS)

    print("AIRL-Het primer: running known-truth validation")
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
            "primer": "airl_het",
            "enforce_gates": bool(args.enforce_gates),
        }
    )
    run_dir = args.output_dir / f"{cell.cell_id}_airl_het_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "dgp": dgp, "run_dir": run_dir}


def render_results_tex(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    add = lines.append

    cell_list = ", ".join(record["payload"]["cell"].cell_id for record in records)
    add("% Auto-generated by aairl_run.py. Do not edit by hand.")
    add("% Known-truth DGP cells: " + cell_list)
    add("")
    add(r"\subsection{Known-Truth Validation Results}")
    add(
        "The AIRL-Het validation target is a serialized-content DGP with two "
        "latent user segments, "
        "multiple books per user, pay/wait/exit actions, an exit-action reward "
        "anchor, an absorbing terminal state, an 18-dimensional state encoding, "
        "20 known finite reward features, and observed solver truth. The "
        "estimator uses a behavioral-anchor warm start with ridge projection "
        "and then runs the EM-AIRL refinement."
    )
    add("")
    add(r"\subsection{Validation Design Context}")
    add(validation_context([record["payload"]["cell"].cell_id for record in records]))
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{AIRL-Het known-truth validation cells.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{llrrrrrrr}")
    add(r"\toprule")
    add(
        r"Cell & Role & States & Actions & State dim & Reward dim & "
        r"Segments & Individuals & Periods \\"
    )
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
            f"{int(dgp.state_features.shape[1])} & "
            f"{int(dgp.feature_matrix.shape[-1])} & "
            f"{dgp.num_segments} & {sim.n_individuals:,} & {sim.n_periods} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{AIRL-Het segment recovery metrics. Lower is better except assignment.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{lrrrrrrrr}")
    add(r"\toprule")
    add(
        r"Cell & Gates & Assignment & Prior L1 & Reward nRMSE & Policy TV & "
        r"Value nRMSE & Q nRMSE & Type A/B/C regret \\"
    )
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        cell_id = payload["cell"].cell_id
        gates = payload["gates"]
        passed = sum(gate.passed for gate in gates)
        metrics = payload["metrics"]
        regrets = "/".join(
            fmt(metrics["max_segment_counterfactual_regret"].get(kind), 4)
            for kind in ("type_a", "type_b", "type_c")
        )
        add(
            f"{tex_text(validation_display_name(cell_id))} & "
            f"{passed}/{len(gates)} & "
            f"{fmt(metrics.get('segment_assignment_accuracy'), 4)} & "
            f"{fmt(metrics.get('segment_prior_l1'), 4)} & "
            f"{fmt(metrics.get('max_segment_reward_normalized_rmse'), 4)} & "
            f"{fmt(metrics.get('max_segment_policy_tv'), 4)} & "
            f"{fmt(metrics.get('max_segment_value_normalized_rmse'), 4)} & "
            f"{fmt(metrics.get('max_segment_q_normalized_rmse'), 4)} & "
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


def compact_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = next(
        (
            record
            for record in records
            if record["payload"]["cell"].cell_id == PRIMARY_CELL_ID
        ),
        records[0],
    )
    primary_failed = [gate for gate in primary["payload"]["gates"] if not gate.passed]
    status = "pass" if not primary_failed else "fail"
    return {
        "estimator": ESTIMATOR,
        "primary_cell_id": PRIMARY_CELL_ID,
        "status": status,
        "interpretation": (
            "AIRL-Het is a diagnostic failure on the anchored serialized-content "
            "latent-segment known-truth cell."
            if primary_failed
            else (
                "AIRL-Het passes the anchored serialized-content latent-segment "
                "known-truth cell."
            )
        ),
        "results": [compact_cell_payload(record["payload"]) for record in records],
    }


def compact_cell_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    metadata = summary.metadata
    metrics = payload["metrics"]
    return {
        "cell_id": payload["cell"].cell_id,
        "dgp_config": payload["cell"].dgp_config.to_dict(),
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
                "num_segments": metadata.get("num_segments"),
                "segment_priors": metadata.get("segment_priors"),
                "learned_shaping": metadata.get("learned_shaping"),
                "initialization": metadata.get("initialization"),
                "initialization_smoothing": metadata.get("initialization_smoothing"),
                "initialization_l2_penalty": metadata.get(
                    "initialization_l2_penalty"
                ),
                "generator_reward": metadata.get("generator_reward"),
                "min_airl_rounds": metadata.get("min_airl_rounds"),
                "em_log_likelihoods": metadata.get("em_log_likelihoods"),
            },
        },
        "metrics": {
            "segment_permutation": metrics.get("segment_permutation"),
            "segment_assignment_accuracy": metrics.get("segment_assignment_accuracy"),
            "segment_prior_l1": metrics.get("segment_prior_l1"),
            "segment_prior_max_abs_error": metrics.get(
                "segment_prior_max_abs_error"
            ),
            "segment_reward_normalized_rmse": metrics.get(
                "segment_reward_normalized_rmse"
            ),
            "max_segment_reward_normalized_rmse": metrics.get(
                "max_segment_reward_normalized_rmse"
            ),
            "max_segment_policy_tv": metrics.get("max_segment_policy_tv"),
            "segment_value_normalized_rmse": metrics.get(
                "segment_value_normalized_rmse"
            ),
            "max_segment_value_normalized_rmse": metrics.get(
                "max_segment_value_normalized_rmse"
            ),
            "segment_q_normalized_rmse": metrics.get("segment_q_normalized_rmse"),
            "max_segment_q_normalized_rmse": metrics.get(
                "max_segment_q_normalized_rmse"
            ),
            "max_segment_counterfactual_regret": metrics.get(
                "max_segment_counterfactual_regret"
            ),
        },
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
