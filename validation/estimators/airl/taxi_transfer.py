#!/usr/bin/env python3
"""Generated taxi-zone study of AIRL reward transfer across traffic dynamics."""

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
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "airl_taxi_transfer.json"
GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
DISCOUNT = 0.95
FEATURE_NAMES = ["downtown_access", "airport_access", "congestion_zone"]
TRUE_PARAMETERS = np.array([2.0, 1.0, -1.5], dtype=float)

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


THRESHOLDS = {
    "maximum_median_reward_nrmse": 0.25,
    "maximum_p95_baseline_policy_tv": 0.08,
    "maximum_p95_transfer_policy_tv": 0.10,
    "maximum_p95_transfer_flow_regret": 0.05,
    "minimum_oracle_policy_tv": 0.01,
    "minimum_state_action_coverage": 0.95,
}


def state_features() -> np.ndarray:
    """Return three state-only taxi-zone features."""
    features = np.zeros((N_STATES, 3), dtype=float)
    for state in range(N_STATES):
        row, col = divmod(state, GRID_SIZE)
        downtown_distance = abs(row - 0) + abs(col - 3)
        airport_distance = abs(row - 3) + abs(col - 0)
        features[state] = [
            -downtown_distance / 6.0,
            -airport_distance / 6.0,
            float((row, col) in {(1, 1), (1, 2), (2, 1), (2, 2)}),
        ]
    return features


def deterministic_transitions() -> np.ndarray:
    """Return taxi movement dynamics in (action, state, next_state) order."""
    transitions = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=float)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for action, (dr, dc) in enumerate(moves):
        for state in range(N_STATES):
            row, col = divmod(state, GRID_SIZE)
            next_row = min(max(row + dr, 0), GRID_SIZE - 1)
            next_col = min(max(col + dc, 0), GRID_SIZE - 1)
            successor = next_row * GRID_SIZE + next_col
            transitions[action, state, successor] = 1.0
    return transitions


def changed_transitions() -> np.ndarray:
    """Add directional congestion and a downtown eastbound closure."""
    base = deterministic_transitions()
    changed = np.zeros_like(base)
    reliability = np.array([0.65, 0.55, 0.90, 0.90])
    for action in range(N_ACTIONS):
        for state in range(N_STATES):
            successor = int(np.argmax(base[action, state]))
            success = reliability[action]
            if action == 1 and state in {1, 5, 9}:
                success = 0.10
            changed[action, state, successor] += success
            changed[action, state, state] += 1.0 - success
    return changed


def problem():
    from econirl.core.types import DDCProblem

    return DDCProblem(
        num_states=N_STATES,
        num_actions=N_ACTIONS,
        discount_factor=DISCOUNT,
        scale_parameter=1.0,
    )


def solve(transitions: np.ndarray, reward_matrix: np.ndarray):
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration

    result = value_iteration(
        SoftBellmanOperator(problem(), jnp.asarray(transitions)),
        jnp.asarray(reward_matrix),
        tol=1e-10,
        max_iter=10_000,
    )
    if not result.converged:
        raise RuntimeError("taxi-zone oracle did not converge")
    return result


def simulate_panel(policy: np.ndarray, seed: int, n_individuals: int, periods: int):
    from econirl.core.types import Panel, Trajectory

    transitions = deterministic_transitions()
    rng = np.random.default_rng(seed)
    trajectories = []
    for individual in range(n_individuals):
        state = int(rng.integers(N_STATES))
        states = np.empty(periods, dtype=np.int32)
        actions = np.empty(periods, dtype=np.int32)
        next_states = np.empty(periods, dtype=np.int32)
        for period in range(periods):
            action = int(rng.choice(N_ACTIONS, p=policy[state]))
            successor = int(rng.choice(N_STATES, p=transitions[action, state]))
            states[period] = state
            actions[period] = action
            next_states[period] = successor
            state = successor
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
                individual_id=individual,
            )
        )
    return Panel(trajectories)


def policy_tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(0.5 * np.abs(left - right).sum(axis=1)))


def fit_once(replication: int, *, smoke: bool) -> dict[str, Any]:
    from econirl import AIRL, RewardSpec
    from validation.known_truth import counterfactual_metrics, normalized_rmse

    features = state_features()
    reward_state = features @ TRUE_PARAMETERS
    reward_matrix = np.repeat(reward_state[:, None], N_ACTIONS, axis=1)
    baseline_oracle = solve(deterministic_transitions(), reward_matrix)
    transfer_oracle = solve(changed_transitions(), reward_matrix)
    panel = simulate_panel(
        np.asarray(baseline_oracle.policy),
        seed=61_000 + replication,
        n_individuals=100 if smoke else 300,
        periods=30 if smoke else 80,
    )
    spec = RewardSpec.state_dependent(jnp.asarray(features), FEATURE_NAMES, N_ACTIONS)
    model = AIRL(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        discount=DISCOUNT,
        max_rounds=80 if smoke else 200,
        min_rounds=50 if smoke else 150,
        discriminator_steps=5,
        policy_step_size=0.1,
        generator_reward="f",
        compute_se=False,
        seed=62_000 + replication,
    ).fit(panel, transitions=deterministic_transitions(), reward=spec)
    counterfactual = model.counterfactual(
        transitions=changed_transitions(),
        description="Directional congestion and downtown eastbound closure",
    )
    transfer_metrics = counterfactual_metrics(
        oracle_policy=transfer_oracle.policy,
        oracle_value=transfer_oracle.V,
        estimated_policy=counterfactual.counterfactual_policy,
        reward=reward_matrix,
        transitions=changed_transitions(),
        discount_factor=DISCOUNT,
        initial_distribution=np.ones(N_STATES) / N_STATES,
        scale_parameter=1.0,
    )
    states = np.asarray(panel.get_all_states(), dtype=int)
    actions = np.asarray(panel.get_all_actions(), dtype=int)
    coverage = len(np.unique(np.stack([states, actions], axis=1), axis=0)) / (N_STATES * N_ACTIONS)
    changed_policy = np.asarray(counterfactual.counterfactual_policy)
    baseline_policy = np.asarray(model.policy_)
    state_changes = 0.5 * np.abs(changed_policy - baseline_policy).sum(axis=1)
    most_changed_state = int(np.argmax(state_changes))
    return {
        "replication": replication,
        "panel_seed": 61_000 + replication,
        "training_seed": 62_000 + replication,
        "n_observations": panel.num_observations,
        "converged": bool(model.converged_),
        "reward_nrmse": normalized_rmse(model.reward_, reward_state),
        "baseline_policy_tv": policy_tv(baseline_policy, np.asarray(baseline_oracle.policy)),
        "transfer_policy_tv": float(transfer_metrics.policy.tv),
        "transfer_flow_regret": float((1.0 - DISCOUNT) * transfer_metrics.regret),
        "oracle_policy_tv": policy_tv(
            np.asarray(baseline_oracle.policy), np.asarray(transfer_oracle.policy)
        ),
        "fitted_policy_change_tv": policy_tv(baseline_policy, changed_policy),
        "state_action_coverage": coverage,
        "most_changed_state": most_changed_state,
        "most_changed_state_policy_tv": float(state_changes[most_changed_state]),
        "baseline_policy_at_most_changed_state": baseline_policy[most_changed_state].tolist(),
        "changed_policy_at_most_changed_state": changed_policy[most_changed_state].tolist(),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_replications": len(records),
        "n_converged": sum(record["converged"] for record in records),
    }
    for name in (
        "reward_nrmse",
        "baseline_policy_tv",
        "transfer_policy_tv",
        "transfer_flow_regret",
        "oracle_policy_tv",
        "fitted_policy_change_tv",
        "state_action_coverage",
    ):
        values = np.asarray([record[name] for record in records], dtype=float)
        summary[name] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(values.max()),
        }
    return summary


def gate_results(summary: dict[str, Any]) -> list[dict[str, Any]]:
    values = {
        "all_fits_converged": summary["n_converged"] == summary["n_replications"],
        "median_reward_nrmse": summary["reward_nrmse"]["median"],
        "p95_baseline_policy_tv": summary["baseline_policy_tv"]["p95"],
        "p95_transfer_policy_tv": summary["transfer_policy_tv"]["p95"],
        "p95_transfer_flow_regret": summary["transfer_flow_regret"]["p95"],
        "minimum_oracle_policy_tv": summary["oracle_policy_tv"]["minimum"],
        "minimum_state_action_coverage": summary["state_action_coverage"]["minimum"],
    }
    gates = [
        {
            "name": "all_fits_converged",
            "value": values["all_fits_converged"],
            "operator": "is",
            "threshold": True,
            "passed": bool(values["all_fits_converged"]),
        }
    ]
    specs = {
        "median_reward_nrmse": ("<=", THRESHOLDS["maximum_median_reward_nrmse"]),
        "p95_baseline_policy_tv": (
            "<=",
            THRESHOLDS["maximum_p95_baseline_policy_tv"],
        ),
        "p95_transfer_policy_tv": (
            "<=",
            THRESHOLDS["maximum_p95_transfer_policy_tv"],
        ),
        "p95_transfer_flow_regret": (
            "<=",
            THRESHOLDS["maximum_p95_transfer_flow_regret"],
        ),
        "minimum_oracle_policy_tv": (
            ">=",
            THRESHOLDS["minimum_oracle_policy_tv"],
        ),
        "minimum_state_action_coverage": (
            ">=",
            THRESHOLDS["minimum_state_action_coverage"],
        ),
    }
    for name, (operator, threshold) in specs.items():
        value = values[name]
        gates.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(value <= threshold if operator == "<=" else value >= threshold),
            }
        )
    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    n_replications = 1 if args.smoke else 3
    records = [fit_once(replication, smoke=args.smoke) for replication in range(n_replications)]
    summary = aggregate(records)
    checks = gate_results(summary)
    receipt = {
        "estimator": "AIRL",
        "status": "ready" if all(check["passed"] for check in checks) else "failed",
        "paper_replication": False,
        "study": "generated state-only taxi-zone dynamics transfer",
        "configuration": {
            "grid_size": GRID_SIZE,
            "states": N_STATES,
            "actions": N_ACTIONS,
            "features": FEATURE_NAMES,
            "true_parameters": TRUE_PARAMETERS.tolist(),
            "n_replications": n_replications,
            "individuals": 100 if args.smoke else 300,
            "periods": 30 if args.smoke else 80,
            "training_dynamics": "deterministic taxi grid",
            "transfer_dynamics": "directional congestion and downtown closure",
        },
        "diagnostics": {
            "feature_rank": int(np.linalg.matrix_rank(state_features())),
            "feature_condition_number": float(np.linalg.cond(state_features())),
            "transition_orientation": "(n_actions, n_states, n_states)",
            "maximum_training_row_sum_error": float(
                np.max(np.abs(deterministic_transitions().sum(axis=-1) - 1.0))
            ),
            "maximum_transfer_row_sum_error": float(
                np.max(np.abs(changed_transitions().sum(axis=-1) - 1.0))
            ),
        },
        "frozen_thresholds": THRESHOLDS,
        "summary": summary,
        "checks": checks,
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
    print(f"wrote {args.output}")
    return 0 if receipt["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
