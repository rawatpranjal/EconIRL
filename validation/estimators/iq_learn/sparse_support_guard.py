#!/usr/bin/env python3
"""Generate the local IQ-Learn sparse-support artifact.

This is deliberately tiny. It exercises IQ-Learn on a panel that observes only
one of three states and records that the non-smoke support gates fail. It is a
support-gate guard, not counterfactual-valid release evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "iq_learn_sparse_support_guard.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.types import DDCProblem, Panel, Trajectory  # noqa: E402
from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator  # noqa: E402
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from validation.known_truth import recovery_gates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = run_guard()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_finite_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


def run_guard() -> dict[str, Any]:
    problem_config = {
        "num_states": 3,
        "num_actions": 2,
        "discount_factor": 0.9,
        "scale_parameter": 1.0,
    }
    panel_config = {
        "n_individuals": 20,
        "n_periods": 5,
        "observed_states": [0],
        "observed_state_actions": [[0, 0]],
        "seed": 0,
    }
    solver_config = {
        "q_type": "tabular",
        "divergence": "chi2",
        "optimizer": "L-BFGS-B",
        "max_iter": 50,
        "verbose": False,
    }

    problem = DDCProblem(**problem_config)
    transitions = _deterministic_transitions(problem)
    utility = _action_indicator_reward(problem)
    panel = _sparse_panel(panel_config)

    estimator = IQLearnEstimator(config=IQLearnConfig(**solver_config))
    wall_start = time.time()
    summary = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=problem,
        transitions=transitions,
    )
    wall_time_seconds = time.time() - wall_start

    metric_fixture = _passing_metric_fixture()
    gates = recovery_gates("IQ-Learn", summary, metric_fixture["metrics"], smoke=False)
    gate_payloads = [asdict(gate) for gate in gates]
    failed_gates = [gate.name for gate in gates if not gate.passed]
    support_gate_names = {
        "expert_state_coverage",
        "expert_state_action_coverage",
    }
    failed_support_gates = [
        gate.name
        for gate in gates
        if gate.name in support_gate_names and not gate.passed
    ]
    non_support_gates_passed = all(
        gate.passed for gate in gates if gate.name not in support_gate_names
    )
    treated_as_counterfactual_valid = all(gate.passed for gate in gates)

    metadata = {
        "q_type": summary.metadata.get("q_type"),
        "divergence": summary.metadata.get("divergence"),
        "alpha": summary.metadata.get("alpha"),
        "final_objective": summary.metadata.get("final_objective"),
        "expert_state_coverage": summary.metadata.get("expert_state_coverage"),
        "expert_state_action_coverage": summary.metadata.get(
            "expert_state_action_coverage"
        ),
    }

    return {
        "artifact_name": "iq_learn_sparse_support_guard",
        "artifact_type": "local_sparse_support_guard",
        "estimator": "IQ-Learn",
        "generated_by": (
            "validation/estimators/iq_learn/"
            "iq_learn_sparse_support_guard.py"
        ),
        "release_status": "local_support_guard_not_counterfactual_release_evidence",
        "counterfactual_valid_certified": False,
        "does_not_replace": "full_known_truth_sparse_state_iq_learn_artifact",
        "purpose": (
            "Run IQ-Learn on a tiny sparse expert panel and evaluate the "
            "non-smoke support gates with otherwise passing diagnostic "
            "controls, so sparse support cannot be described as "
            "counterfactual-valid evidence."
        ),
        "problem_config": problem_config,
        "panel_config": panel_config,
        "solver_config": solver_config,
        "thresholds": {
            "expert_state_coverage_min": 1.0,
            "expert_state_action_coverage_min": 0.95,
        },
        "observed_support": {
            "states": [0],
            "state_action_pairs": [[0, 0]],
            "total_states": problem.num_states,
            "total_state_action_pairs": problem.num_states * problem.num_actions,
        },
        "metric_fixture": {
            "purpose": (
                "Passing control values isolate the support gates; they are "
                "not measured structural recovery metrics for this tiny cell."
            ),
            "values": metric_fixture["values"],
        },
        "gates": gate_payloads,
        "result": {
            "treated_as_counterfactual_valid": treated_as_counterfactual_valid,
            "counterfactual_valid_certified": treated_as_counterfactual_valid,
            "support_gates_passed": not failed_support_gates,
            "support_gate_passed": not failed_support_gates,
            "non_support_gates_passed": non_support_gates_passed,
            "failed_gates": failed_gates,
            "failed_support_gates": failed_support_gates,
            "converged": bool(summary.converged),
            "num_iterations": int(summary.num_iterations),
            "num_observations": int(summary.num_observations),
            "num_individuals": int(summary.num_individuals),
            "num_periods": int(summary.num_periods),
            "estimation_time": float(summary.estimation_time),
            "wall_time_seconds": float(wall_time_seconds),
            "convergence_message": summary.convergence_message,
            "metadata": metadata,
            "parameters": np.asarray(summary.parameters, dtype=float).tolist(),
        },
    }


def _deterministic_transitions(problem: DDCProblem) -> jnp.ndarray:
    transitions = jnp.zeros(
        (problem.num_actions, problem.num_states, problem.num_states)
    )
    for state in range(problem.num_states):
        transitions = transitions.at[0, state, state].set(1.0)
        transitions = transitions.at[1, state, (state + 1) % problem.num_states].set(
            1.0
        )
    return transitions


def _action_indicator_reward(problem: DDCProblem) -> ActionDependentReward:
    features = jnp.zeros((problem.num_states, problem.num_actions, 2))
    features = features.at[:, 0, 0].set(1.0)
    features = features.at[:, 1, 1].set(1.0)
    return ActionDependentReward(
        feature_matrix=features,
        parameter_names=["action_0_reward", "action_1_reward"],
    )


def _sparse_panel(panel_config: dict[str, Any]) -> Panel:
    trajectories = []
    observed_state = panel_config["observed_states"][0]
    observed_action = panel_config["observed_state_actions"][0][1]
    for individual_id in range(panel_config["n_individuals"]):
        states = jnp.full(panel_config["n_periods"], observed_state)
        actions = jnp.full(panel_config["n_periods"], observed_action)
        next_states = jnp.full(panel_config["n_periods"], observed_state)
        trajectories.append(
            Trajectory(
                states=states,
                actions=actions,
                next_states=next_states,
                individual_id=individual_id,
            )
        )
    return Panel(trajectories=trajectories)


def _passing_metric_fixture() -> dict[str, Any]:
    values = {
        "policy_tv": 0.01,
        "raw_bellman_reward_normalized_rmse": 0.01,
        "projected_reward_normalized_rmse": 0.01,
        "value_normalized_rmse": 0.01,
        "q_normalized_rmse": 0.01,
        "type_a_regret": 0.01,
        "type_b_regret": 0.01,
        "type_c_regret": 0.01,
    }
    return {
        "values": values,
        "metrics": {
            "policy": SimpleNamespace(tv=values["policy_tv"]),
            "raw_bellman_reward_normalized_rmse": values[
                "raw_bellman_reward_normalized_rmse"
            ],
            "projected_reward_normalized_rmse": values[
                "projected_reward_normalized_rmse"
            ],
            "value_normalized_rmse": values["value_normalized_rmse"],
            "q_normalized_rmse": values["q_normalized_rmse"],
            "counterfactuals": {
                "type_a": SimpleNamespace(regret=values["type_a_regret"]),
                "type_b": SimpleNamespace(regret=values["type_b_regret"]),
                "type_c": SimpleNamespace(regret=values["type_c_regret"]),
            },
        },
    }


def _finite_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _finite_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
