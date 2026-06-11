"""Rust bus random-start recovery benchmark for SEES variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from econirl.environments import RustBusEnvironment
from econirl.estimation import NFXPEstimator
from econirl.estimation.sees import SEESEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


MODES = ("value", "q", "ev", "policy", "collocation")
PANEL_SEEDS = (2, 17, 29)
RANDOM_STARTS = (
    (0.09936, 5.04395),
    (-0.09200, 4.08347),
    (0.00596, 0.95535),
)
THRESHOLDS = {
    "param_rmse": 0.01,
    "policy_tv": 0.001,
    "value_rmse": 0.05,
    "q_rmse": 0.05,
    "bellman_violation": 0.0015,
}


def rmse(estimated: jnp.ndarray, reference: jnp.ndarray) -> float:
    """Root mean squared error."""
    return float(jnp.sqrt(jnp.mean((estimated - reference) ** 2)))


def q_values(
    params: jnp.ndarray,
    value: jnp.ndarray,
    utility: LinearUtility,
    problem,
    transitions: jnp.ndarray,
) -> jnp.ndarray:
    """Choice-specific values implied by structural parameters and V."""
    reward = utility.compute(params)
    continuation = problem.discount_factor * jnp.einsum(
        "ast,t->as",
        transitions,
        value,
    ).T
    return reward + continuation


def metric_record(
    *,
    seed: int,
    solution: str,
    start: jnp.ndarray,
    result,
    reference,
    reference_q: jnp.ndarray,
    utility: LinearUtility,
    problem,
    transitions: jnp.ndarray,
) -> dict[str, Any]:
    """Compute SEES-vs-NFXP recovery metrics for one fit."""
    policy_tv = float(
        0.5 * jnp.mean(jnp.sum(jnp.abs(result.policy - reference.policy), axis=1))
    )
    value_rmse = rmse(result.value_function, reference.value_function)
    estimated_q = q_values(
        result.parameters,
        result.value_function,
        utility,
        problem,
        transitions,
    )
    q_rmse = rmse(estimated_q, reference_q)
    bellman_violation = float(result.metadata["bellman_violation"])
    selected_gradient_norm = float(
        result.metadata.get("selected_gradient_norm", float("nan"))
    )
    metrics = {
        "param_rmse": rmse(result.parameters, reference.parameters),
        "policy_tv": policy_tv,
        "value_rmse": value_rmse,
        "q_rmse": q_rmse,
        "bellman_violation": bellman_violation,
    }
    passed = all(metrics[name] < threshold for name, threshold in THRESHOLDS.items())
    return {
        "seed": seed,
        "solution": solution,
        "initial_params": [float(x) for x in np.asarray(start)],
        "estimated_params": [float(x) for x in np.asarray(result.parameters)],
        "reference_params": [float(x) for x in np.asarray(reference.parameters)],
        "selected_theta_start": result.metadata.get("selected_theta_start"),
        "num_theta_starts": int(result.metadata.get("num_theta_starts", 1)),
        "optimizer_success": bool(result.metadata.get("optimizer_success", result.converged)),
        "summary_converged": bool(result.converged),
        "selected_gradient_norm": selected_gradient_norm,
        "selected_objective": float(result.metadata.get("selected_objective", float("nan"))),
        "num_iterations": int(result.num_iterations),
        "metrics": metrics,
        "passed": bool(passed),
    }


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Worst-case summary by SEES solution mode."""
    rows: list[dict[str, Any]] = []
    for solution in MODES:
        solution_records = [record for record in records if record["solution"] == solution]
        if not solution_records:
            continue
        rows.append(
            {
                "solution": solution,
                "passes": sum(record["passed"] for record in solution_records),
                "runs": len(solution_records),
                "max_param_rmse": max(
                    record["metrics"]["param_rmse"] for record in solution_records
                ),
                "max_policy_tv": max(
                    record["metrics"]["policy_tv"] for record in solution_records
                ),
                "max_value_rmse": max(
                    record["metrics"]["value_rmse"] for record in solution_records
                ),
                "max_q_rmse": max(
                    record["metrics"]["q_rmse"] for record in solution_records
                ),
                "max_bellman_violation": max(
                    record["metrics"]["bellman_violation"] for record in solution_records
                ),
                "max_gradient_norm": max(
                    record["selected_gradient_norm"] for record in solution_records
                ),
                "optimizer_successes": sum(
                    record["optimizer_success"] for record in solution_records
                ),
            }
        )
    return rows


def run_benchmark(
    *,
    panel_seeds: tuple[int, ...] = PANEL_SEEDS,
    random_starts: tuple[tuple[float, float], ...] = RANDOM_STARTS,
    n_individuals: int = 300,
    n_periods: int = 50,
    num_theta_starts: int = 4,
) -> dict[str, Any]:
    """Run the SEES random-start Rust bus benchmark."""
    records: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for seed in panel_seeds:
        env = RustBusEnvironment(
            operating_cost=0.01,
            replacement_cost=3.0,
            num_mileage_bins=12,
            discount_factor=0.95,
        )
        panel = simulate_panel(
            env,
            n_individuals=n_individuals,
            n_periods=n_periods,
            seed=seed,
        )
        utility = LinearUtility.from_environment(env)
        problem = env.problem_spec
        transitions = env.transition_matrices
        reference = NFXPEstimator(
            inner_tol=1e-10,
            inner_max_iter=20_000,
            compute_hessian=False,
            verbose=False,
        ).estimate(panel, utility, problem, transitions)
        reference_q = q_values(
            reference.parameters,
            reference.value_function,
            utility,
            problem,
            transitions,
        )
        references.append(
            {
                "seed": seed,
                "nfxp_params": [float(x) for x in np.asarray(reference.parameters)],
                "nfxp_converged": bool(reference.converged),
                "nfxp_iterations": int(reference.num_iterations),
            }
        )

        for solution in MODES:
            for start_tuple in random_starts:
                start = jnp.asarray(start_tuple, dtype=jnp.float64)
                estimator = SEESEstimator(
                    solution=solution,
                    basis_type="bspline",
                    basis_dim=env.num_states,
                    penalty_weight=30.0,
                    num_theta_starts=num_theta_starts,
                    max_iter=1000,
                    tol=1e-7,
                    compute_se=False,
                    verbose=False,
                )
                result = estimator.estimate(
                    panel,
                    utility,
                    problem,
                    transitions,
                    initial_params=start,
                )
                records.append(
                    metric_record(
                        seed=seed,
                        solution=solution,
                        start=start,
                        result=result,
                        reference=reference,
                        reference_q=reference_q,
                        utility=utility,
                        problem=problem,
                        transitions=transitions,
                    )
                )

    return {
        "design": {
            "environment": "RustBusEnvironment",
            "operating_cost": 0.01,
            "replacement_cost": 3.0,
            "num_mileage_bins": 12,
            "discount_factor": 0.95,
            "n_individuals": n_individuals,
            "n_periods": n_periods,
            "panel_seeds": list(panel_seeds),
            "random_starts": [list(start) for start in random_starts],
            "num_theta_starts": num_theta_starts,
            "reference": "NFXP same-sample estimate",
        },
        "thresholds": dict(THRESHOLDS),
        "references": references,
        "records": records,
        "summary": summarize(records),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact markdown table from a benchmark payload."""
    lines = [
        "# SEES Rust Random-Start Benchmark",
        "",
        "Finite-sample recovery is measured against NFXP on the same simulated panel.",
        "",
        "| Mode | Passes | Max param RMSE | Max policy TV | Max value RMSE | Max Q RMSE | Max Bellman | Max grad norm | Optimizer flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {solution} | {passes}/{runs} | {max_param_rmse:.6f} | "
            "{max_policy_tv:.6f} | {max_value_rmse:.6f} | {max_q_rmse:.6f} | "
            "{max_bellman_violation:.3e} | {max_gradient_norm:.3e} | "
            "{optimizer_successes}/{runs} |".format(**row)
        )
    lines.append("")
    lines.append("The optimizer flag is the strict JAXopt gradient flag; recovery gates use the metrics above.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    payload = run_benchmark()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.md_out is not None:
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
