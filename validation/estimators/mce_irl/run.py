#!/usr/bin/env python3
"""Generate MCE-IRL primer simulation results from known-truth DGPs.

The MCE-IRL primer runs the low-level tabular estimator directly with known
transitions and known action-dependent reward features. The primary path uses
root feature matching, not the likelihood optimizer, because this is the
paper-faithful Ziebart-style MCE-IRL stationarity condition.

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python validation/estimators/mce_irl/run.py --enforce-gates
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "mce_irl.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_mce_irl_primer_known_truth")
DEFAULT_CELL_IDS = ("canonical_low_action", "mce_low_high_reward")
PRIMARY_CELL_ID = "mce_low_high_reward"
ESTIMATOR = "MCE-IRL"
SIMULATION_TARGET = (
    "Tabular maximum causal entropy feature matching with known transitions "
    "and known action-dependent reward features."
)
CELL_ROLES = {
    "canonical_low_action": "sanity check",
    "mce_low_high_reward": "main simulation",
}
CELL_LABELS = {
    "canonical_low_action": "Small sanity design",
    "mce_low_high_reward": "Feature-rich primary design",
}

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validation.known_truth import (  # noqa: E402
    KnownTruthCell,
    KnownTruthDGPConfig,
    RecoveryGateFailure,
    SimulationConfig,
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    stable_hash,
    write_json,
)


MCE_HARD_CELL = KnownTruthCell(
    cell_id="mce_low_high_reward",
    dgp_config=KnownTruthDGPConfig(
        state_mode="low_dim",
        reward_mode="action_dependent",
        reward_dim="high",
        heterogeneity="none",
        num_regular_states=24,
        high_reward_features=8,
        transition_noise=0.02,
        seed=742,
    ),
    simulation_config=SimulationConfig(
        n_individuals=3_000,
        n_periods=100,
        seed=742,
    ),
    description=(
        "MCE-friendly hard case: low-dimensional states, high-dimensional "
        "action-dependent reward features, known transitions, and strong "
        "state-action coverage."
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help=(
            "Known-truth cell to run. May be repeated. Defaults to "
            "canonical_low_action and mce_low_high_reward."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()

    cell_ids = args.cell_id if args.cell_id is not None else list(DEFAULT_CELL_IDS)

    print("MCE-IRL primer: running known-truth simulation")
    print(f"  cells: {', '.join(cell_ids)}")
    print(f"  estimator: {ESTIMATOR}")
    print(f"  output_dir: {args.output_dir}")

    records = [run_simulation_cell(cell_id, args) for cell_id in cell_ids]

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


def run_simulation_cell(cell_id: str, args: argparse.Namespace) -> dict[str, Any]:
    cell = _get_mce_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = replace(
        cell.simulation_config,
        show_progress=args.show_progress,
    )
    panel = simulate_panel(dgp, simulation_config)
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
            "primer": "mce_irl",
            "enforce_gates": bool(args.enforce_gates),
        }
    )
    run_dir = args.output_dir / f"{cell.cell_id}_mce_irl_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "dgp": dgp, "run_dir": run_dir}


def _get_mce_cell(cell_id: str) -> KnownTruthCell:
    if cell_id == MCE_HARD_CELL.cell_id:
        return MCE_HARD_CELL
    return get_cell(cell_id)


def simulate_panel(dgp: Any, simulation_config: SimulationConfig) -> Any:
    from validation.known_truth import simulate_known_truth_panel

    return simulate_known_truth_panel(dgp, simulation_config)


def render_results_tex(records: list[dict[str, Any]]) -> str:
    records_by_id = {record["payload"]["cell"].cell_id: record for record in records}
    low_record = records_by_id.get("canonical_low_action")
    hard_record = records_by_id.get(PRIMARY_CELL_ID)

    lines: list[str] = []
    add = lines.append

    add("% Auto-generated by mce_irl_run.py. Do not edit by hand.")
    add("% Known-truth DGP cells: " + ", ".join(records_by_id))
    add("")
    add(r"\subsection{Simulation Study}")
    add(
        "We run a simulation to ask a simple question: if the data really come "
        "from a maximum-causal-entropy model, can MCE-IRL recover the reward "
        "and produce the right counterfactual behavior? This cannot be checked "
        "from real choice data alone, because the true reward and the true "
        "counterfactual policies are unobserved."
    )
    add("")
    add(
        "The data-generating process is finite-state. We choose transition "
        "matrices, action-dependent reward features, and true reward weights. "
        "Together these objects define the true reward, policy, value function, "
        "Q function, occupancy measure, and Type A/B/C counterfactual oracles. "
        "We then simulate panel data from the true policy and estimate MCE-IRL "
        "using the simulated choices, the known transitions, and the same "
        "feature basis. The estimator sees the features and transitions, but "
        "not the true reward weights."
    )
    add("")
    add(
        "The first design is a small sanity check. The second design is the "
        "main simulation: it keeps the state space compact, but uses a richer "
        "action-dependent reward basis. The tables below show both designs."
    )
    add("")
    add(r"\subsection{Simulation Design}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{MCE-IRL simulation designs.}")
    add(r"\begin{tabular}{llrrrrr}")
    add(r"\toprule")
    add(r"Design & Purpose & States & Actions & Reward features & Individuals & Periods \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        dgp = record["dgp"]
        cell_id = payload["cell"].cell_id
        sim = payload["simulation"]
        add(
            f"{tex_text(CELL_LABELS.get(cell_id, cell_id))} & "
            f"{tex_text(CELL_ROLES.get(cell_id, 'simulation'))} & "
            f"{dgp.problem.num_states} & {dgp.problem.num_actions} & "
            f"{dgp.feature_matrix.shape[-1]} & {sim.n_individuals:,} & "
            f"{sim.n_periods} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    if hard_record is not None:
        summary = hard_record["payload"]["summary"]
        metadata = summary.metadata
        add(r"\begin{table}[H]")
        add(r"\centering\small")
        add(r"\caption{MCE-IRL primary fit summary.}")
        add(r"\begin{tabular}{lr}")
        add(r"\toprule")
        add(r"Quantity & Value \\")
        add(r"\midrule")
        add(f"Converged & {tex_text(str(summary.converged).lower())} \\\\")
        add(f"Outer iterations & {summary.num_iterations} \\\\")
        add(f"Log likelihood & {fmt(summary.log_likelihood, 4)} \\\\")
        add(f"Estimation time & {fmt(summary.estimation_time, 2)} seconds \\\\")
        add(f"Optimizer & {tt(str(metadata.get('optimizer')))} \\\\")
        add(
            "Feature residual & "
            f"{fmt(metadata.get('feature_difference'), 6)} \\\\"
        )
        add(
            "Occupancy moment residual & "
            f"{fmt(metadata.get('occupancy_moment_residual'), 6)} \\\\"
        )
        add(r"\bottomrule")
        add(r"\end{tabular}")
        add(r"\end{table}")
        add("")

    add(
        "After estimation, we compare the recovered objects with the truth. "
        "Reward, value, and Q recovery are evaluated after the usual IRL "
        "location-and-scale normalization. Policy distance and counterfactual "
        "regret are not normalized; they ask whether the recovered reward "
        "induces the right behavior in the original and changed environments."
    )
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{MCE-IRL recovery and counterfactual results.}")
    add(r"\begin{tabular}{lrr}")
    add(r"\toprule")
    add(r"Metric & Sanity cell & Primary cell \\")
    add(r"\midrule")
    metric_rows = (
        ("Feature residual", "feature_residual", 6),
        ("Occupancy moment residual", "occupancy_moment_residual", 6),
        ("Reward normalized RMSE", "reward_normalized_rmse", 6),
        ("Policy TV", "policy_tv", 6),
        ("Value normalized RMSE", "value_normalized_rmse", 6),
        ("Q normalized RMSE", "q_normalized_rmse", 6),
    )
    for label, key, digits in metric_rows:
        add(
            f"{tex_text(label)} & "
            f"{fmt(metric_value(low_record, key), digits)} & "
            f"{fmt(metric_value(hard_record, key), digits)} \\\\"
        )
    for kind, label in (
        ("type_a", "Type A regret"),
        ("type_b", "Type B regret"),
        ("type_c", "Type C regret"),
    ):
        add(
            f"{label} & "
            f"{fmt(counterfactual_value(low_record, kind), 6)} & "
            f"{fmt(counterfactual_value(hard_record, kind), 6)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    return "\n".join(lines)


def metric_value(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None:
        return None
    payload = record["payload"]
    metrics = payload["metrics"]
    if key == "feature_residual":
        return payload["summary"].metadata.get("feature_difference")
    if key == "occupancy_moment_residual":
        return payload["summary"].metadata.get("occupancy_moment_residual")
    if key == "policy_tv":
        return metrics["policy"].tv
    return metrics.get(key)


def counterfactual_value(record: dict[str, Any] | None, kind: str) -> float | None:
    if record is None:
        return None
    return record["payload"]["metrics"]["counterfactuals"][kind].regret


def compact_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    results = [compact_cell_payload(record["payload"]) for record in records]
    primary_result = next(
        (result for result in results if result["cell_id"] == PRIMARY_CELL_ID),
        results[0] if results else {},
    )
    return {
        "estimator": ESTIMATOR,
        "simulation_target": SIMULATION_TARGET,
        "primary_cell_id": PRIMARY_CELL_ID,
        "artifact_type": "simulation_study",
        "result": primary_result,
        "results": results,
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
            "parameter_names": summary.parameter_names,
            "parameters": summary.parameters,
            "standard_errors": finite_list(summary.standard_errors),
            "log_likelihood": summary.log_likelihood,
            "converged": summary.converged,
            "num_iterations": summary.num_iterations,
            "num_observations": summary.num_observations,
            "estimation_time": summary.estimation_time,
            "convergence_message": summary.convergence_message,
            "metadata": {
                "optimizer": metadata.get("optimizer"),
                "feature_difference": metadata.get("feature_difference"),
                "occupancy_moment_residual": metadata.get(
                    "occupancy_moment_residual"
                ),
                "empirical_features": metadata.get("empirical_features"),
                "final_expected_features": metadata.get("final_expected_features"),
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


def finite_list(values: Any) -> list[float | None]:
    array = np.asarray(values, dtype=float)
    return [float(value) if math.isfinite(float(value)) else None for value in array]


def tt(value: str) -> str:
    return r"\texttt{" + tex_text(value) + "}"


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
