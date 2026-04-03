"""Sample-size sweep for neural estimators on simulated linear Rust bus data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from econirl import NFXP
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimators.neural_gladius import NeuralGLADIUS
from econirl.estimators.nnes import NNES
from econirl.estimators.tdccp import TDCCP
from econirl.simulation.synthetic import simulate_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable neural-estimator sweep on linear Rust bus data."
    )
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[250, 2000])
    parser.add_argument("--n-periods", type=int, default=80)
    parser.add_argument("--num-states", type=int, default=90)
    parser.add_argument("--discount", type=float, default=0.9999)
    parser.add_argument("--operating-cost", type=float, default=0.001)
    parser.add_argument("--replacement-cost", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    sample_label = "-".join(str(n) for n in args.sample_sizes)
    filename = (
        f"neural_linear_rust_sample_sweep_"
        f"n{sample_label}_t{args.n_periods}_s{args.seed}.json"
    )
    return Path("experiments/identification/results") / filename


def make_state_encoder(num_states: int):
    max_state = max(num_states - 1, 1)
    return lambda s, _max_state=max_state: (s.float() / _max_state).unsqueeze(-1)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def implied_policy(
    feature_matrix: np.ndarray,
    coefficients: np.ndarray,
    operator: SoftBellmanOperator,
) -> np.ndarray:
    reward_matrix = jnp.einsum(
        "sak,k->sa",
        jnp.array(feature_matrix),
        jnp.array(coefficients),
    )
    return np.asarray(value_iteration(operator, reward_matrix).policy[:, 1])


def load_existing_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    payload = json.loads(output_path.read_text())
    return payload.get("results", [])


def save_results(
    output_path: Path,
    args: argparse.Namespace,
    results: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "sample_sizes": args.sample_sizes,
            "n_periods": args.n_periods,
            "num_states": args.num_states,
            "discount": args.discount,
            "operating_cost": args.operating_cost,
            "replacement_cost": args.replacement_cost,
            "seed": args.seed,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def neural_estimators(num_states: int, discount: float, parameter_names: list[str]) -> list[tuple[str, object]]:
    return [
        (
            "NNES",
            NNES(
                n_states=num_states,
                discount=discount,
                bellman="npl",
                hidden_dim=64,
                v_epochs=500,
                n_outer_iterations=3,
                verbose=False,
            ),
        ),
        (
            "TD-CCP",
            TDCCP(
                n_states=num_states,
                discount=discount,
                hidden_dim=64,
                avi_iterations=20,
                epochs_per_avi=30,
                batch_size=4096,
                n_policy_iterations=3,
                verbose=False,
            ),
        ),
        (
            "NeuralGLADIUS",
            NeuralGLADIUS(
                n_actions=2,
                discount=discount,
                state_encoder=make_state_encoder(num_states),
                state_dim=1,
                feature_names=parameter_names,
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-4,
                max_epochs=1000,
                batch_size=512,
                patience=200,
                verbose=False,
            ),
        ),
    ]


def fit_gladius_full_projection(
    model: NeuralGLADIUS,
    df,
    feature_matrix: np.ndarray,
    num_states: int,
):
    model.fit(
        df,
        state="mileage_bin",
        action="replaced",
        id="id",
        features=feature_matrix,
    )
    model._n_states = num_states
    model._project_onto_features(feature_matrix, None, None, None)
    return model


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args)
    resume = not args.no_resume

    env = RustBusEnvironment(
        operating_cost=args.operating_cost,
        replacement_cost=args.replacement_cost,
        num_mileage_bins=args.num_states,
        discount_factor=args.discount,
        seed=args.seed,
    )
    problem = DDCProblem(
        num_states=args.num_states,
        num_actions=2,
        discount_factor=args.discount,
        scale_parameter=1.0,
    )
    true_params = np.array([args.operating_cost, args.replacement_cost])
    feature_matrix = np.asarray(env.feature_matrix)
    operator = SoftBellmanOperator(problem, jnp.array(env.transition_matrices))
    true_policy = implied_policy(feature_matrix, true_params, operator)

    existing_results = load_existing_results(output_path) if resume else []
    results_by_key = {
        (row["sample_size"], row["estimator"]): row for row in existing_results
    }

    print(f"Output: {output_path}", flush=True)
    print(
        f"{'N':>6s}  {'Estimator':<14s}  {'Policy MAE':>10s}  {'Corr':>7s}  "
        f"{'Param MAE':>10s}  {'Time':>7s}",
        flush=True,
    )
    print("-" * 72, flush=True)

    ordered_results: list[dict] = []
    for n_individuals in args.sample_sizes:
        panel = simulate_panel(
            env,
            n_individuals=n_individuals,
            n_periods=args.n_periods,
            seed=args.seed,
        )
        df = panel.to_dataframe().rename(
            columns={"state": "mileage_bin", "action": "replaced"}
        )

        baseline_key = (n_individuals, "NFXP")
        if baseline_key not in results_by_key:
            start = time.time()
            nfxp = NFXP(n_states=args.num_states, discount=args.discount, verbose=False)
            nfxp.fit(df, state="mileage_bin", action="replaced", id="id")
            baseline_policy = implied_policy(feature_matrix, np.asarray(nfxp.coef_), operator)
            baseline_row = {
                "sample_size": n_individuals,
                "estimator": "NFXP",
                "status": "ok",
                "policy_mae": float(np.abs(baseline_policy - true_policy).mean()),
                "policy_corr": safe_corr(baseline_policy, true_policy),
                "param_mae": float(np.abs(np.asarray(nfxp.coef_) - true_params).mean()),
                "runtime_seconds": time.time() - start,
                "params": dict(nfxp.params_),
            }
            results_by_key[baseline_key] = baseline_row
            save_results(output_path, args, list(results_by_key.values()))

        for estimator_name, estimator in neural_estimators(
            args.num_states,
            args.discount,
            env.parameter_names,
        ):
            key = (n_individuals, estimator_name)
            if key in results_by_key and results_by_key[key].get("status") == "ok":
                row = results_by_key[key]
                print(
                    f"{n_individuals:6d}  {estimator_name:<14s}  {row['policy_mae']:10.4f}  "
                    f"{row['policy_corr']:7.4f}  {row['param_mae']:10.4f}  "
                    f"{row['runtime_seconds']:7.1f}",
                    flush=True,
                )
                continue

            start = time.time()
            try:
                if estimator_name == "NeuralGLADIUS":
                    model = fit_gladius_full_projection(
                        estimator,
                        df,
                        feature_matrix,
                        args.num_states,
                    )
                else:
                    model = estimator.fit(
                        df,
                        state="mileage_bin",
                        action="replaced",
                        id="id",
                    )

                coefficients = np.asarray(model.coef_)
                policy = implied_policy(feature_matrix, coefficients, operator)
                row = {
                    "sample_size": n_individuals,
                    "estimator": estimator_name,
                    "status": "ok",
                    "policy_mae": float(np.abs(policy - true_policy).mean()),
                    "policy_corr": safe_corr(policy, true_policy),
                    "param_mae": float(np.abs(coefficients - true_params).mean()),
                    "runtime_seconds": time.time() - start,
                    "params": {
                        name: float(value)
                        for name, value in zip(env.parameter_names, coefficients)
                    },
                }
                print(
                    f"{n_individuals:6d}  {estimator_name:<14s}  {row['policy_mae']:10.4f}  "
                    f"{row['policy_corr']:7.4f}  {row['param_mae']:10.4f}  "
                    f"{row['runtime_seconds']:7.1f}",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "sample_size": n_individuals,
                    "estimator": estimator_name,
                    "status": "error",
                    "runtime_seconds": time.time() - start,
                    "error": repr(exc),
                }
                print(
                    f"{n_individuals:6d}  {estimator_name:<14s}  {'ERROR':>10s}  {repr(exc)[:38]}",
                    flush=True,
                )

            results_by_key[key] = row
            save_results(output_path, args, list(results_by_key.values()))

    ordered_results = sorted(
        results_by_key.values(),
        key=lambda row: (row["sample_size"], row["estimator"]),
    )
    save_results(output_path, args, ordered_results)
    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
