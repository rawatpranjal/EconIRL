#!/usr/bin/env python3
"""Compare NeuralAIRL with frozen linear AIRL on a nonlinear state reward."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "neural_airl_nonlinear.json"
FULL_REPLICATIONS = 10

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


THRESHOLDS = {
    "maximum_mean_tv_ratio_to_linear_airl": 0.75,
    "minimum_replications_improved": 8,
    "maximum_p95_policy_tv": 0.10,
    "maximum_p95_transfer_policy_tv": 0.12,
    "maximum_median_affine_reward_nrmse": 0.35,
}


def problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return inputs, original transitions, changed transitions, and truth."""
    n_states = 9
    x = np.linspace(-1.0, 1.0, n_states, dtype=np.float64)
    reward = 1.5 * np.cos(np.pi * x) - 0.35 * x
    transitions = np.zeros((2, n_states, n_states), dtype=np.float64)
    changed = np.zeros_like(transitions)
    for state in range(n_states):
        right = (state + 1) % n_states
        left = (state - 1) % n_states
        transitions[0, state, right] = 0.9
        transitions[0, state, state] = 0.1
        transitions[1, state, left] = 0.9
        transitions[1, state, state] = 0.1
        changed[0, state, right] = 0.7
        changed[0, state, state] = 0.3
        changed[1, state, left] = 0.65
        changed[1, state, state] = 0.35
    return x[:, None], transitions, changed, reward


def solve(reward: np.ndarray, transitions: np.ndarray) -> Any:
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration
    from econirl.core.types import DDCProblem

    ddc = DDCProblem(9, 2, 0.9, 1.0)
    return value_iteration(
        SoftBellmanOperator(ddc, jnp.asarray(transitions)),
        jnp.repeat(jnp.asarray(reward)[:, None], 2, axis=1),
        tol=1e-10,
        max_iter=5_000,
    )


def simulate(
    seed: int,
    *,
    smoke: bool,
    transitions: np.ndarray | None = None,
):
    from econirl.core.types import Panel, Trajectory

    _, default_transitions, _, reward = problem()
    if transitions is None:
        transitions = default_transitions
    oracle = solve(reward, transitions)
    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []
    individuals = 60 if smoke else 120
    periods = 18 if smoke else 25
    for individual in range(individuals):
        state = int(rng.integers(9))
        states: list[int] = []
        actions: list[int] = []
        next_states: list[int] = []
        for _ in range(periods):
            action = int(rng.choice(2, p=np.asarray(oracle.policy[state])))
            following = int(rng.choice(9, p=transitions[action, state]))
            states.append(state)
            actions.append(action)
            next_states.append(following)
            state = following
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
                individual_id=individual,
            )
        )
    return Panel(trajectories=trajectories)


def policy_tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(left) - np.asarray(right)).sum(axis=1).mean())


def affine_nrmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    design = np.column_stack([np.asarray(estimate), np.ones(len(estimate))])
    coefficient, *_ = np.linalg.lstsq(design, np.asarray(truth), rcond=None)
    aligned = design @ coefficient
    scale = np.ptp(truth)
    return float(np.sqrt(np.mean((aligned - truth) ** 2)) / scale)


def fit_once(replication: int, *, smoke: bool) -> dict[str, Any]:
    from econirl import AIRL, NeuralAIRL
    from econirl.evaluation.selfcheck import assert_effect

    inputs, transitions, changed, reward = problem()
    panel_seed = 24_000 + replication
    training_seed = 25_000 + replication
    panel = simulate(panel_seed, smoke=smoke)
    rounds = 90 if smoke else 160
    minimum = 45 if smoke else 70
    shared = dict(
        n_states=9,
        n_actions=2,
        discount=0.9,
        feature_matrix=inputs,
        max_rounds=rounds,
        min_rounds=minimum,
        discriminator_steps=3,
        policy_step_size=0.1,
        compute_se=False,
        seed=training_seed,
    )
    linear = AIRL(**shared).fit(panel, transitions=transitions)
    neural = NeuralAIRL(
        **shared,
        reward_hidden_dim=32,
        reward_num_layers=2,
        shaping_hidden_dim=32,
        policy_hidden_dim=32,
        policy_steps=15,
    ).fit(panel, transitions=transitions)
    oracle = solve(reward, transitions)
    transfer_oracle = solve(reward, changed)
    transfer = neural.counterfactual(transitions=changed)
    linear_tv = policy_tv(linear.policy_, oracle.policy)
    neural_tv = policy_tv(neural.policy_, oracle.policy)
    assert_effect(linear_tv - neural_tv, min_abs=0.005)
    return {
        "replication": replication,
        "panel_seed": panel_seed,
        "training_seed": training_seed,
        "n_observations": panel.num_observations,
        "linear_policy_tv": linear_tv,
        "neural_policy_tv": neural_tv,
        "tv_ratio": neural_tv / linear_tv,
        "improved": neural_tv < linear_tv,
        "affine_reward_nrmse": affine_nrmse(neural.reward_, reward),
        "transfer_policy_tv": policy_tv(
            transfer.counterfactual_policy,
            transfer_oracle.policy,
        ),
        "converged": bool(neural.converged_),
        "rounds": int(neural.n_iter_),
        "final_discriminator_loss": float(
            neural.diagnostics_["optimization"]["final_discriminator_loss"]
        ),
        "policy_distillation_tv": float(
            neural.diagnostics_["optimization"]["policy_distillation_tv"]
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "linear_policy_tv",
        "neural_policy_tv",
        "tv_ratio",
        "affine_reward_nrmse",
        "transfer_policy_tv",
        "policy_distillation_tv",
    )
    summary: dict[str, Any] = {
        "n_requested": len(records),
        "n_converged": sum(record["converged"] for record in records),
        "n_improved": sum(record["improved"] for record in records),
    }
    for field in fields:
        values = np.asarray([record[field] for record in records], dtype=float)
        summary[field] = {
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(values.max()),
        }
    summary["mean_tv_ratio_to_linear_airl"] = (
        summary["neural_policy_tv"]["mean"] / summary["linear_policy_tv"]["mean"]
    )
    return summary


def checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    specifications = {
        "all_fits_converged": (
            summary["n_converged"],
            "==",
            summary["n_requested"],
        ),
        "mean_tv_ratio_to_linear_airl": (
            summary["mean_tv_ratio_to_linear_airl"],
            "<=",
            THRESHOLDS["maximum_mean_tv_ratio_to_linear_airl"],
        ),
        "replications_improved": (
            summary["n_improved"],
            ">=",
            min(
                THRESHOLDS["minimum_replications_improved"],
                summary["n_requested"],
            ),
        ),
        "p95_policy_tv": (
            summary["neural_policy_tv"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_policy_tv"],
        ),
        "p95_transfer_policy_tv": (
            summary["transfer_policy_tv"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_transfer_policy_tv"],
        ),
        "median_affine_reward_nrmse": (
            summary["affine_reward_nrmse"]["median"],
            "<=",
            THRESHOLDS["maximum_median_affine_reward_nrmse"],
        ),
    }
    output: list[dict[str, Any]] = []
    for name, (value, operator, threshold) in specifications.items():
        passed = (
            value == threshold
            if operator == "=="
            else value <= threshold
            if operator == "<="
            else value >= threshold
        )
        output.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    n_replications = args.replications or (1 if args.smoke else FULL_REPLICATIONS)
    records = [fit_once(index, smoke=args.smoke) for index in range(n_replications)]
    summary = summarize(records)
    gate_results = checks(summary)
    receipt = {
        "estimator": "NeuralAIRL",
        "status": "ready" if all(gate["passed"] for gate in gate_results) else "failed",
        "paper_replication": False,
        "paper_boundary": (
            "Generated finite-state nonlinear recovery and transfer evidence. "
            "It is not a replication of Fu et al. continuous-control results."
        ),
        "target": "nonlinear state reward and changed-dynamics behavior",
        "configuration": {
            "states": 9,
            "actions": 2,
            "state_inputs": 1,
            "n_replications": n_replications,
            "individuals": 60 if args.smoke else 120,
            "periods": 18 if args.smoke else 25,
        },
        "frozen_thresholds": THRESHOLDS,
        "summary": summary,
        "checks": gate_results,
        "records": records,
        "environment": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("econirl", "jax", "jaxlib", "numpy", "optax")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "summary": summary}, indent=2))
    print(f"wrote {args.output}")
    return 0 if receipt["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
