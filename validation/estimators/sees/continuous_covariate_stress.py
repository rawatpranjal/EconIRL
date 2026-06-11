"""SEES vs NFXP on encoded continuous-covariate product grids.

This is a stress probe for the claim that sieve value approximations can be
useful when continuous state covariates make a full NFXP grid unattractive.

Important scope note: the current econirl SEES implementation still consumes a
dense finite-state transition tensor. This script therefore tests the package's
encoded-state SEES path and reports the state-grid explosion explicitly. It is
not a full sparse/collocation continuous-state SEES implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.environments.shapeshifter import (
    ShapeshifterConfig,
    ShapeshifterEnvironment,
)
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.sees import SEESEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

DEFAULT_OUTPUT = Path("validation/results/sees_continuous_covariate_stress")


@dataclass(frozen=True)
class StressConfig:
    bins_per_dim: int
    state_dim: int
    num_actions: int
    num_features: int
    n_individuals: int
    n_periods: int
    seed: int
    sees_basis_dim: int
    sees_penalty_weight: float

    @property
    def num_states(self) -> int:
        return self.bins_per_dim ** self.state_dim


def _rmse(estimated: jnp.ndarray, truth: jnp.ndarray) -> float:
    estimated = jnp.asarray(estimated)
    truth = jnp.asarray(truth)
    return float(jnp.sqrt(jnp.mean((estimated - truth) ** 2)))


def _policy_tv(estimated: jnp.ndarray, truth: jnp.ndarray) -> float:
    estimated = jnp.asarray(estimated)
    truth = jnp.asarray(truth)
    return float(0.5 * jnp.mean(jnp.sum(jnp.abs(estimated - truth), axis=1)))


def _transition_gib(num_states: int, num_actions: int, *, dtype_bytes: int = 8) -> float:
    return num_actions * num_states * num_states * dtype_bytes / (1024 ** 3)


def _build_env(config: StressConfig) -> ShapeshifterEnvironment:
    env_config = ShapeshifterConfig(
        num_states=config.bins_per_dim,
        num_actions=config.num_actions,
        num_features=config.num_features,
        discount_factor=0.92,
        scale_parameter=1.0,
        reward_type="linear",
        feature_type="linear",
        action_dependent=True,
        stochastic_rewards=False,
        stochastic_transitions=True,
        transition_branching=min(4, config.bins_per_dim),
        state_dim=config.state_dim,
        seed=config.seed,
        max_total_states=max(4096, config.num_states),
        reward_scale=0.7,
    )
    return ShapeshifterEnvironment(env_config)


def _solve_truth(env: ShapeshifterEnvironment):
    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    return value_iteration(
        operator,
        env.compute_utility_matrix(),
        tol=1e-10,
        max_iter=20_000,
    )


def _q_from_value(
    reward: jnp.ndarray,
    value: jnp.ndarray,
    transitions: jnp.ndarray,
    discount_factor: float,
) -> jnp.ndarray:
    continuation = discount_factor * jnp.einsum("ast,t->as", transitions, value).T
    return reward + continuation


def _result_metrics(
    *,
    result,
    utility: LinearUtility,
    transitions: jnp.ndarray,
    discount_factor: float,
    true_params: jnp.ndarray,
    truth,
) -> dict[str, Any]:
    estimated_reward = utility.compute(result.parameters)
    estimated_q = _q_from_value(
        estimated_reward,
        result.value_function,
        transitions,
        discount_factor,
    )
    return {
        "converged": bool(result.converged),
        "iterations": int(result.num_iterations),
        "time_seconds": float(result.estimation_time),
        "log_likelihood": float(result.log_likelihood),
        "estimated_params": [float(x) for x in np.asarray(result.parameters)],
        "param_rmse": _rmse(result.parameters, true_params),
        "policy_tv": _policy_tv(result.policy, truth.policy),
        "value_rmse": _rmse(result.value_function, truth.V),
        "q_rmse": _rmse(estimated_q, truth.Q),
        "bellman_violation": (
            None
            if "bellman_violation" not in result.metadata
            else float(result.metadata["bellman_violation"])
        ),
        "basis_source": result.metadata.get("basis_source"),
        "basis_family": result.metadata.get("basis_family"),
        "state_basis_dim": result.metadata.get("state_basis_dim"),
        "configured_basis_dim": result.metadata.get("configured_basis_dim"),
        "gradient_norm": result.metadata.get("selected_gradient_norm"),
        "message": result.convergence_message,
    }


def _fit_nfxp(panel, utility, problem, transitions, *, initial_params: jnp.ndarray):
    estimator = NFXPEstimator(
        optimizer="BHHH",
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=30_000,
        outer_tol=1e-5,
        outer_max_iter=40,
        compute_hessian=False,
        verbose=False,
    )
    return estimator.estimate(
        panel,
        utility,
        problem,
        transitions,
        initial_params=initial_params,
    )


def _fit_sees(
    panel,
    utility,
    problem,
    transitions,
    *,
    initial_params: jnp.ndarray,
    basis_dim: int,
    penalty_weight: float,
):
    estimator = SEESEstimator(
        solution="value",
        basis_type="bspline",
        basis_dim=basis_dim,
        penalty_weight=penalty_weight,
        state_basis_mode="encoded",
        num_theta_starts=3,
        warm_start_value=False,
        max_iter=600,
        tol=1e-5,
        compute_se=False,
        verbose=False,
    )
    return estimator.estimate(
        panel,
        utility,
        problem,
        transitions,
        initial_params=initial_params,
    )


def run_one(config: StressConfig, *, max_nfxp_states: int) -> dict[str, Any]:
    env = _build_env(config)
    truth = _solve_truth(env)
    panel = simulate_panel(
        env,
        n_individuals=config.n_individuals,
        n_periods=config.n_periods,
        seed=config.seed + 1000,
        policy=truth.policy,
    )
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()
    initial_params = jnp.zeros_like(true_params)
    basis_dim = min(config.sees_basis_dim, config.num_states)

    record: dict[str, Any] = {
        "config": asdict(config),
        "num_states": config.num_states,
        "observations": int(panel.num_observations),
        "transition_tensor_gib_float64": _transition_gib(
            config.num_states,
            config.num_actions,
            dtype_bytes=8,
        ),
        "true_params": [float(x) for x in np.asarray(true_params)],
        "truth_solver": {
            "converged": bool(truth.converged),
            "iterations": int(truth.num_iterations),
            "final_error": float(truth.final_error),
        },
        "sees": None,
        "nfxp": None,
    }

    t0 = time.time()
    try:
        sees_result = _fit_sees(
            panel,
            utility,
            env.problem_spec,
            env.transition_matrices,
            initial_params=initial_params,
            basis_dim=basis_dim,
            penalty_weight=config.sees_penalty_weight,
        )
        record["sees"] = _result_metrics(
            result=sees_result,
            utility=utility,
            transitions=env.transition_matrices,
            discount_factor=env.problem_spec.discount_factor,
            true_params=true_params,
            truth=truth,
        )
        record["sees"]["wall_seconds"] = time.time() - t0
    except Exception as exc:  # pragma: no cover - diagnostic script
        record["sees"] = {
            "error": type(exc).__name__,
            "message": str(exc),
            "wall_seconds": time.time() - t0,
        }

    if config.num_states > max_nfxp_states:
        record["nfxp"] = {
            "skipped": True,
            "reason": (
                f"num_states={config.num_states} exceeds "
                f"max_nfxp_states={max_nfxp_states}"
            ),
        }
        return record

    t0 = time.time()
    try:
        nfxp_result = _fit_nfxp(
            panel,
            utility,
            env.problem_spec,
            env.transition_matrices,
            initial_params=initial_params,
        )
        record["nfxp"] = _result_metrics(
            result=nfxp_result,
            utility=utility,
            transitions=env.transition_matrices,
            discount_factor=env.problem_spec.discount_factor,
            true_params=true_params,
            truth=truth,
        )
        record["nfxp"]["wall_seconds"] = time.time() - t0
    except Exception as exc:  # pragma: no cover - diagnostic script
        record["nfxp"] = {
            "error": type(exc).__name__,
            "message": str(exc),
            "wall_seconds": time.time() - t0,
        }
    return record


def explosion_table(
    *,
    bins_per_dim: int,
    num_actions: int,
    max_dim: int,
    max_nfxp_states: int,
    sees_basis_dim: int,
) -> list[dict[str, Any]]:
    rows = []
    for state_dim in range(1, max_dim + 1):
        states = bins_per_dim ** state_dim
        rows.append(
            {
                "bins_per_dim": bins_per_dim,
                "state_dim": state_dim,
                "num_states": states,
                "nfxp_value_unknowns": states,
                "sees_value_unknowns": min(sees_basis_dim, states),
                "transition_tensor_gib_float64": _transition_gib(
                    states,
                    num_actions,
                    dtype_bytes=8,
                ),
                "nfxp_status": "run" if states <= max_nfxp_states else "skip",
            }
        )
    return rows


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# SEES Continuous-Covariate Stress Probe",
        "",
        "This probe uses product-grid states encoded as continuous covariates. "
        "It tests the current package's encoded-state SEES path and reports the "
        "NFXP state-grid explosion.",
        "",
        "Scope: this is not yet a true sparse/collocation continuous-state SEES "
        "implementation. The current package still materializes dense transition "
        "tensors, so very large grids are skipped rather than attempted.",
        "",
        "## Grid Explosion",
        "",
        "| State dim | States | NFXP value unknowns | SEES value unknowns | "
        "Dense transition GiB | NFXP status |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["grid_explosion"]:
        lines.append(
            "| {state_dim} | {num_states} | {nfxp_value_unknowns} | "
            "{sees_value_unknowns} | {transition_tensor_gib_float64:.3f} | "
            "{nfxp_status} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Actual Runs",
            "",
            "| Dim | States | Method | Time sec | Converged | Param RMSE | "
            "Policy TV | Value RMSE | Bellman | Notes |",
            "| ---: | ---: | --- | ---: | :---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in payload["runs"]:
        for method in ("sees", "nfxp"):
            result = record[method]
            if result is None:
                continue
            if result.get("skipped"):
                lines.append(
                    f"| {record['config']['state_dim']} | {record['num_states']} | "
                    f"{method.upper()} |  |  |  |  |  |  | {result['reason']} |"
                )
                continue
            if "error" in result:
                lines.append(
                    f"| {record['config']['state_dim']} | {record['num_states']} | "
                    f"{method.upper()} | {result['wall_seconds']:.3f} | no |  |  |  |  | "
                    f"{result['error']}: {result['message']} |"
                )
                continue
            bellman = result["bellman_violation"]
            bellman_text = "" if bellman is None else f"{bellman:.3e}"
            lines.append(
                f"| {record['config']['state_dim']} | {record['num_states']} | "
                f"{method.upper()} | {result['wall_seconds']:.3f} | "
                f"{'yes' if result['converged'] else 'no'} | "
                f"{result['param_rmse']:.6f} | {result['policy_tv']:.6f} | "
                f"{result['value_rmse']:.6f} | {bellman_text} | "
                f"{result.get('basis_source') or ''} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- NFXP's exact value object has one unknown per grid state and "
            "must solve a full Bellman problem at each theta.",
            "- SEES uses a lower-dimensional encoded-state value basis in the "
            "optimization, but the current implementation still pays "
            "dense-transition costs.",
            "- The paper-level continuous-state advantage requires a sparse or "
            "simulation/collocation SEES path that evaluates expectations "
            "without a full S by S transition tensor.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins-per-dim", type=int, default=5)
    parser.add_argument("--run-dims", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--explosion-max-dim", type=int, default=8)
    parser.add_argument("--num-actions", type=int, default=3)
    parser.add_argument("--num-features", type=int, default=8)
    parser.add_argument("--n-individuals", type=int, default=300)
    parser.add_argument("--n-periods", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sees-basis-dim", type=int, default=40)
    parser.add_argument("--sees-penalty-weight", type=float, default=100.0)
    parser.add_argument("--max-nfxp-states", type=int, default=250)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    for offset, state_dim in enumerate(args.run_dims):
        config = StressConfig(
            bins_per_dim=args.bins_per_dim,
            state_dim=state_dim,
            num_actions=args.num_actions,
            num_features=args.num_features,
            n_individuals=args.n_individuals,
            n_periods=args.n_periods,
            seed=args.seed + offset,
            sees_basis_dim=args.sees_basis_dim,
            sees_penalty_weight=args.sees_penalty_weight,
        )
        print(
            f"running dim={state_dim}, states={config.num_states}, "
            f"obs={config.n_individuals * config.n_periods}"
        )
        runs.append(run_one(config, max_nfxp_states=args.max_nfxp_states))

    payload = {
        "grid_explosion": explosion_table(
            bins_per_dim=args.bins_per_dim,
            num_actions=args.num_actions,
            max_dim=args.explosion_max_dim,
            max_nfxp_states=args.max_nfxp_states,
            sees_basis_dim=args.sees_basis_dim,
        ),
        "runs": runs,
        "notes": {
            "scope": (
                "Encoded finite product-grid stress. Current SEES still uses "
                "dense finite-state transitions; true continuous-state advantage "
                "requires a sparse/collocation path."
            ),
            "nfxp_skip_rule": (
                f"NFXP skipped above {args.max_nfxp_states} states to avoid "
                "long exact Bellman estimation in an interactive run."
            ),
        },
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
