"""Hyperparameter sweep for NeuralGLADIUS on simulated Rust bus data."""

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
from econirl.simulation.synthetic import simulate_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable NeuralGLADIUS hyperparameter sweep."
    )
    parser.add_argument("--n-individuals", type=int, default=250)
    parser.add_argument("--n-periods", type=int, default=80)
    parser.add_argument("--num-states", type=int, default=90)
    parser.add_argument("--discount", type=float, default=0.9999)
    parser.add_argument("--operating-cost", type=float, default=0.001)
    parser.add_argument("--replacement-cost", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    filename = (
        f"gladius_hyperparam_sweep_"
        f"n{args.n_individuals}_t{args.n_periods}_s{args.seed}.json"
    )
    return Path("experiments/identification/results") / filename


def build_configs() -> list[tuple[str, dict]]:
    return [
        (
            "baseline (64x2, lr=1e-3, bs=512)",
            dict(
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-3,
                max_epochs=500,
                batch_size=512,
                patience=100,
            ),
        ),
        (
            "bigger net (128x3)",
            dict(
                q_hidden_dim=128,
                q_num_layers=3,
                ev_hidden_dim=128,
                ev_num_layers=3,
                lr=1e-3,
                max_epochs=500,
                batch_size=512,
                patience=100,
            ),
        ),
        (
            "smaller lr (1e-4)",
            dict(
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-4,
                max_epochs=1000,
                batch_size=512,
                patience=200,
            ),
        ),
        (
            "bigger batch (2048)",
            dict(
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-3,
                max_epochs=500,
                batch_size=2048,
                patience=100,
            ),
        ),
        (
            "longer training (2000 epochs)",
            dict(
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-3,
                max_epochs=2000,
                batch_size=512,
                patience=300,
            ),
        ),
        (
            "tikhonov annealing",
            dict(
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-3,
                max_epochs=500,
                batch_size=512,
                patience=100,
                tikhonov_annealing=True,
                tikhonov_initial_weight=100.0,
            ),
        ),
        (
            "no alternating (joint)",
            dict(
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-3,
                max_epochs=500,
                batch_size=512,
                patience=100,
                alternating_updates=False,
            ),
        ),
        (
            "high bellman weight (10)",
            dict(
                q_hidden_dim=64,
                q_num_layers=2,
                ev_hidden_dim=64,
                ev_num_layers=2,
                lr=1e-3,
                max_epochs=500,
                batch_size=512,
                patience=100,
                bellman_weight=10.0,
            ),
        ),
    ]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def make_state_encoder(num_states: int):
    max_state = max(num_states - 1, 1)
    return lambda s, _max_state=max_state: (s.float() / _max_state).unsqueeze(-1)


def load_existing_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    payload = json.loads(output_path.read_text())
    return payload.get("results", [])


def save_results(
    output_path: Path,
    args: argparse.Namespace,
    nfxp_mae: float | None,
    results: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "n_individuals": args.n_individuals,
            "n_periods": args.n_periods,
            "num_states": args.num_states,
            "discount": args.discount,
            "operating_cost": args.operating_cost,
            "replacement_cost": args.replacement_cost,
            "seed": args.seed,
        },
        "nfxp_policy_mae": nfxp_mae,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args)
    resume = not args.no_resume

    configs = build_configs()
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    print(f"Output: {output_path}", flush=True)
    print(
        f"Simulating panel with {args.n_individuals} individuals and "
        f"{args.n_periods} periods.",
        flush=True,
    )

    env = RustBusEnvironment(
        operating_cost=args.operating_cost,
        replacement_cost=args.replacement_cost,
        num_mileage_bins=args.num_states,
        discount_factor=args.discount,
        seed=args.seed,
    )
    panel = simulate_panel(
        env,
        n_individuals=args.n_individuals,
        n_periods=args.n_periods,
        seed=args.seed,
    )
    df = panel.to_dataframe().rename(
        columns={"state": "mileage_bin", "action": "replaced"}
    )
    print(f"Data: {len(df):,} observations", flush=True)

    problem = DDCProblem(
        num_states=args.num_states,
        num_actions=2,
        discount_factor=args.discount,
        scale_parameter=1.0,
    )
    true_params = np.array([args.operating_cost, args.replacement_cost])
    true_reward = jnp.einsum(
        "sak,k->sa",
        jnp.array(env.feature_matrix),
        jnp.array(true_params),
    )
    true_operator = SoftBellmanOperator(problem, jnp.array(env.transition_matrices))
    true_policy = np.asarray(value_iteration(true_operator, true_reward).policy[:, 1])

    nfxp = NFXP(n_states=args.num_states, discount=args.discount, verbose=False)
    nfxp.fit(df, state="mileage_bin", action="replaced", id="id")
    nfxp_mae = float(np.abs(nfxp.policy_[:, 1] - true_policy).mean())
    print(f"NFXP policy MAE: {nfxp_mae:.4f}", flush=True)

    keep_transition = np.asarray(nfxp.transitions_)
    transitions = np.zeros((2, args.num_states, args.num_states))
    transitions[0] = keep_transition
    transitions[1] = keep_transition[0]
    estimated_operator = SoftBellmanOperator(problem, jnp.array(transitions))

    existing_results = load_existing_results(output_path) if resume else []
    results_by_name = {row["config_name"]: row for row in existing_results}

    print(
        f"{'Config':<35s}  {'Epochs':>7s}  {'MAE':>7s}  {'Corr':>7s}  "
        f"{'θ MAE':>7s}  {'Time':>7s}",
        flush=True,
    )
    print("-" * 81, flush=True)

    ordered_results: list[dict] = []
    for name, kwargs in configs:
        if name in results_by_name and results_by_name[name].get("status") == "ok":
            row = results_by_name[name]
            ordered_results.append(row)
            print(
                f"{name:<35s}  {row['epochs']:>7d}  "
                f"{row['policy_mae']:7.4f}  {row['policy_corr']:7.4f}  "
                f"{row['param_mae']:7.4f}  {row['runtime_seconds']:7.1f}",
                flush=True,
            )
            continue

        start = time.time()
        try:
            model = NeuralGLADIUS(
                n_actions=2,
                discount=args.discount,
                state_encoder=make_state_encoder(args.num_states),
                state_dim=1,
                feature_names=env.parameter_names,
                verbose=args.verbose,
                **kwargs,
            )
            model.fit(
                df,
                state="mileage_bin",
                action="replaced",
                id="id",
                features=np.asarray(env.feature_matrix),
            )
            model._n_states = args.num_states
            model._project_onto_features(np.asarray(env.feature_matrix), None, None, None)
            reward_matrix = jnp.array(model.reward_matrix_)
            policy = np.asarray(value_iteration(estimated_operator, reward_matrix).policy[:, 1])
            policy_mae = float(np.abs(policy - true_policy).mean())
            policy_corr = safe_corr(policy, true_policy)
            param_mae = float(np.abs(model.coef_ - true_params).mean())
            runtime_seconds = time.time() - start
            row = {
                "config_name": name,
                "status": "ok",
                "epochs": int(model.n_epochs_),
                "policy_mae": policy_mae,
                "policy_corr": policy_corr,
                "param_mae": param_mae,
                "runtime_seconds": runtime_seconds,
                "kwargs": kwargs,
                "params": model.params_,
            }
            print(
                f"{name:<35s}  {model.n_epochs_:>7d}  {policy_mae:7.4f}  "
                f"{policy_corr:7.4f}  {param_mae:7.4f}  {runtime_seconds:7.1f}",
                flush=True,
            )
        except Exception as exc:
            runtime_seconds = time.time() - start
            row = {
                "config_name": name,
                "status": "error",
                "runtime_seconds": runtime_seconds,
                "kwargs": kwargs,
                "error": repr(exc),
            }
            print(
                f"{name:<35s}  {'ERROR':>7s}  {repr(exc)[:55]}",
                flush=True,
            )

        results_by_name[name] = row
        ordered_results = [results_by_name[cfg_name] for cfg_name, _ in configs if cfg_name in results_by_name]
        save_results(output_path, args, nfxp_mae, ordered_results)

    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
