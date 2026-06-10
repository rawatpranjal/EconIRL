"""Linear-reward, value-spline benchmark for SEES.

The benchmark keeps the structural reward finite-dimensional and linear in
state progress. Only the SEES Bellman object is represented with a B-spline
value basis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from econirl.estimation.sees import SEESEstimator
from experiments.known_truth import (
    KnownTruthDGP,
    KnownTruthDGPConfig,
    SimulationConfig,
    build_known_truth_dgp,
    known_truth_initial_params,
    simulate_known_truth_panel,
    solve_known_truth,
)

VALUE_BASIS_DIM = 20
PANEL_SEEDS = (43, 65, 91)
START_KINDS = ("near_truth", "zero", "random")
THRESHOLDS = {
    "param_rmse": 0.08,
    "policy_tv": 0.02,
    "value_rmse": 0.08,
    "q_rmse": 0.08,
    "bellman_violation": 0.01,
}


def build_linear_reward_dgp() -> KnownTruthDGP:
    """Build a low-dimensional DDC DGP with a strictly linear reward."""

    config = KnownTruthDGPConfig(
        state_mode="low_dim",
        reward_mode="action_dependent",
        reward_dim="low",
        heterogeneity="none",
        num_regular_states=30,
        num_actions=3,
        discount_factor=0.80,
        transition_noise=0.0,
        seed=42,
    )
    base = build_known_truth_dgp(config)
    progress = np.linspace(0.0, 1.0, config.num_regular_states)
    full_progress = np.concatenate([progress, np.array([0.0])])
    features = np.zeros(
        (base.problem.num_states, base.problem.num_actions, 4),
        dtype=np.float64,
    )
    features[:, 0, 0] = 1.0
    features[:, 0, 1] = full_progress
    features[:, 1, 2] = 1.0
    features[:, 1, 3] = full_progress
    features[config.absorbing_state, :, :] = 0.0
    features[:, config.exit_action, :] = 0.0

    parameter_names = [
        "action_0_intercept",
        "action_0_progress",
        "action_1_intercept",
        "action_1_progress",
    ]
    true_parameters = jnp.asarray([0.10, 0.50, 0.00, -0.20], dtype=jnp.float32)
    feature_matrix = jnp.asarray(features, dtype=jnp.float32)
    reward_matrix = jnp.einsum("sak,k->sa", feature_matrix, true_parameters)
    return KnownTruthDGP(
        config=config,
        problem=base.problem,
        transitions=base.transitions,
        feature_matrix=feature_matrix,
        state_features=base.state_features,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        reward_matrix=reward_matrix,
        initial_distribution=base.initial_distribution,
        segment_probabilities=None,
    )


def rmse(
    estimated: jnp.ndarray,
    reference: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> float:
    """Root mean squared error with an optional Boolean mask."""

    estimated = jnp.asarray(estimated)
    reference = jnp.asarray(reference)
    if mask is not None:
        estimated = estimated[mask]
        reference = reference[mask]
    return float(jnp.sqrt(jnp.mean((estimated - reference) ** 2)))


def policy_tv(estimated: jnp.ndarray, reference: jnp.ndarray) -> float:
    """Mean total-variation distance between policies."""

    return float(0.5 * jnp.mean(jnp.sum(jnp.abs(estimated - reference), axis=1)))


def q_values(
    reward: jnp.ndarray,
    value: jnp.ndarray,
    transitions: jnp.ndarray,
    discount_factor: float,
) -> jnp.ndarray:
    """Choice-specific values implied by reward and continuation value."""

    continuation = discount_factor * jnp.einsum("ast,t->as", transitions, value).T
    return reward + continuation


def initial_params_for(
    dgp: KnownTruthDGP,
    *,
    panel_seed: int,
    start_kind: str,
) -> jnp.ndarray:
    """Deterministic initial parameter choices for robustness checks."""

    if start_kind == "near_truth":
        return known_truth_initial_params(dgp, perturbation_scale=0.03)
    if start_kind == "zero":
        return jnp.zeros_like(dgp.homogeneous_parameters)
    if start_kind == "random":
        rng = np.random.default_rng(panel_seed + 2_026)
        return jnp.asarray(
            rng.normal(0.0, 0.5, size=dgp.homogeneous_parameters.shape[0]),
            dtype=jnp.float64,
        )
    raise ValueError(f"unknown start_kind {start_kind!r}")


def fit_sees(
    *,
    panel,
    dgp: KnownTruthDGP,
    initial_params: jnp.ndarray,
    value_basis_dim: int = VALUE_BASIS_DIM,
) -> Any:
    """Fit V-SEES with a B-spline value basis."""

    estimator = SEESEstimator(
        solution="value",
        basis_type="bspline",
        basis_dim=value_basis_dim,
        penalty_weight=100.0,
        num_theta_starts=4,
        max_iter=1_000,
        tol=1e-5,
        compute_se=False,
        verbose=False,
    )
    return estimator.estimate(
        panel,
        dgp.utility(),
        dgp.problem,
        dgp.transitions,
        initial_params=initial_params,
    )


def metric_record(
    *,
    panel_seed: int,
    start_kind: str,
    result,
    dgp: KnownTruthDGP,
    truth,
    value_basis_dim: int,
) -> dict[str, Any]:
    """Compute known-truth metrics for one fit."""

    estimated_reward = dgp.utility().compute(result.parameters)
    estimated_q = q_values(
        estimated_reward,
        result.value_function,
        dgp.transitions,
        dgp.problem.discount_factor,
    )
    metrics = {
        "param_rmse": rmse(result.parameters, dgp.homogeneous_parameters),
        "policy_tv": policy_tv(result.policy, truth.policy),
        "value_rmse": rmse(result.value_function, truth.V),
        "q_rmse": rmse(estimated_q, truth.Q),
        "bellman_violation": float(result.metadata["bellman_violation"]),
    }
    passed = all(metrics[name] <= threshold for name, threshold in THRESHOLDS.items())
    return {
        "panel_seed": panel_seed,
        "start_kind": start_kind,
        "value_basis_dim": value_basis_dim,
        "num_states": dgp.problem.num_states,
        "estimated_params": [float(x) for x in np.asarray(result.parameters)],
        "true_params": [float(x) for x in np.asarray(dgp.homogeneous_parameters)],
        "selected_theta_start": result.metadata.get("selected_theta_start"),
        "num_theta_starts": int(result.metadata.get("num_theta_starts", 1)),
        "optimizer_success": bool(result.metadata.get("optimizer_success", result.converged)),
        "summary_converged": bool(result.converged),
        "selected_gradient_norm": float(
            result.metadata.get("selected_gradient_norm", float("nan"))
        ),
        "selected_objective": float(result.metadata.get("selected_objective", float("nan"))),
        "num_iterations": int(result.num_iterations),
        "metrics": metrics,
        "passed": bool(passed),
    }


def run_benchmark(
    *,
    panel_seeds: tuple[int, ...] = PANEL_SEEDS,
    start_kinds: tuple[str, ...] = START_KINDS,
    n_individuals: int = 4_000,
    n_periods: int = 80,
    value_basis_dim: int = VALUE_BASIS_DIM,
) -> dict[str, Any]:
    """Run the linear-reward, value-spline SEES benchmark."""

    dgp = build_linear_reward_dgp()
    truth = solve_known_truth(dgp)
    records: list[dict[str, Any]] = []
    for panel_seed in panel_seeds:
        panel = simulate_known_truth_panel(
            dgp,
            SimulationConfig(
                n_individuals=n_individuals,
                n_periods=n_periods,
                seed=panel_seed,
            ),
        )
        for start_kind in start_kinds:
            result = fit_sees(
                panel=panel,
                dgp=dgp,
                initial_params=initial_params_for(
                    dgp,
                    panel_seed=panel_seed,
                    start_kind=start_kind,
                ),
                value_basis_dim=value_basis_dim,
            )
            records.append(
                metric_record(
                    panel_seed=panel_seed,
                    start_kind=start_kind,
                    result=result,
                    dgp=dgp,
                    truth=truth,
                    value_basis_dim=value_basis_dim,
                )
            )

    return {
        "design": {
            "environment": "linear low-dimensional known-truth DDC",
            "reward": "action-specific intercept and progress slope",
            "state_mode": dgp.config.state_mode,
            "reward_mode": dgp.config.reward_mode,
            "reward_dim": "strict_linear",
            "num_states": dgp.problem.num_states,
            "num_regular_states": dgp.config.num_regular_states,
            "num_actions": dgp.problem.num_actions,
            "discount_factor": dgp.problem.discount_factor,
            "transition_noise": dgp.config.transition_noise,
            "value_basis_type": "bspline",
            "value_basis_dim": value_basis_dim,
            "value_basis_compression": f"{value_basis_dim}/{dgp.problem.num_states}",
            "n_individuals": n_individuals,
            "n_periods": n_periods,
            "panel_seeds": list(panel_seeds),
            "start_kinds": list(start_kinds),
        },
        "thresholds": dict(THRESHOLDS),
        "records": records,
        "summary": summarize(records),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Worst-case benchmark summary."""

    return {
        "runs": len(records),
        "passes": sum(record["passed"] for record in records),
        "max_param_rmse": max(record["metrics"]["param_rmse"] for record in records),
        "max_policy_tv": max(record["metrics"]["policy_tv"] for record in records),
        "max_value_rmse": max(record["metrics"]["value_rmse"] for record in records),
        "max_q_rmse": max(record["metrics"]["q_rmse"] for record in records),
        "max_bellman_violation": max(
            record["metrics"]["bellman_violation"] for record in records
        ),
        "max_gradient_norm": max(record["selected_gradient_norm"] for record in records),
        "optimizer_successes": sum(record["optimizer_success"] for record in records),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact markdown summary."""

    lines = [
        "# SEES Linear Reward, Value-Spline Benchmark",
        "",
        "The reward is action-specific linear in state progress. SEES approximates "
        "only the value function with a cubic B-spline basis.",
        "",
        "| Seed | Start | Param RMSE | Policy TV | Value RMSE | Q RMSE | "
        "Bellman | Grad | Opt flag | Pass |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for record in payload["records"]:
        metrics = record["metrics"]
        lines.append(
            "| {panel_seed} | {start_kind} | {param_rmse:.6f} | {policy_tv:.6f} | "
            "{value_rmse:.6f} | {q_rmse:.6f} | {bellman:.3e} | {grad:.3e} | "
            "{opt} | {passed} |".format(
                panel_seed=record["panel_seed"],
                start_kind=record["start_kind"],
                param_rmse=metrics["param_rmse"],
                policy_tv=metrics["policy_tv"],
                value_rmse=metrics["value_rmse"],
                q_rmse=metrics["q_rmse"],
                bellman=metrics["bellman_violation"],
                grad=record["selected_gradient_norm"],
                opt="yes" if record["optimizer_success"] else "no",
                passed="yes" if record["passed"] else "no",
            )
        )
    summary = payload["summary"]
    design = payload["design"]
    lines.extend(
        [
            "",
            "Summary: {passes}/{runs} runs passed; worst parameter RMSE "
            "{max_param_rmse:.6f}; worst policy TV {max_policy_tv:.6f}; "
            "worst value RMSE {max_value_rmse:.6f}; worst Q RMSE "
            "{max_q_rmse:.6f}.".format(**summary),
            "",
            "Value basis compression: {value_basis_compression}. The optimizer "
            "flag is the strict solver gradient flag and is reported separately "
            "from recovery gates.".format(**design),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--n-individuals", type=int, default=4_000)
    parser.add_argument("--n-periods", type=int, default=80)
    parser.add_argument("--panel-seed", action="append", type=int, default=None)
    parser.add_argument("--start-kind", action="append", default=None)
    parser.add_argument("--value-basis-dim", type=int, default=VALUE_BASIS_DIM)
    args = parser.parse_args()

    payload = run_benchmark(
        panel_seeds=tuple(args.panel_seed) if args.panel_seed else PANEL_SEEDS,
        start_kinds=tuple(args.start_kind) if args.start_kind else START_KINDS,
        n_individuals=args.n_individuals,
        n_periods=args.n_periods,
        value_basis_dim=args.value_basis_dim,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.md_out is not None:
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
