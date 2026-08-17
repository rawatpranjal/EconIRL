#!/usr/bin/env python3
"""Controlled state-only AIRL recovery under the paper's positive conditions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "airl_controlled_recovery.json"
FULL_REPLICATIONS = 3

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


THRESHOLDS = {
    "maximum_median_reward_nrmse": 0.15,
    "maximum_p95_reward_nrmse": 0.20,
    "maximum_p95_policy_tv": 0.05,
    "maximum_p95_value_nrmse": 0.20,
    "maximum_p95_q_nrmse": 0.20,
    "maximum_p95_transfer_policy_tv": 0.08,
    "maximum_p95_transfer_regret": 0.08,
    "minimum_state_action_coverage": 0.95,
}


def fit_once(replication: int, *, smoke: bool) -> dict[str, Any]:
    from econirl import AIRL, RewardSpec
    from econirl.core.bellman import SoftBellmanOperator
    from validation.known_truth import (
        build_known_truth_dgp,
        counterfactual_metrics,
        get_cell,
        normalized_rmse,
        simulate_known_truth_panel,
        solve_counterfactual_oracle,
        solve_known_truth,
    )

    cell = get_cell("airl_paper_identification")
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation = replace(
        cell.simulation_config,
        n_individuals=100 if smoke else 300,
        n_periods=30 if smoke else 80,
        seed=17_110 + replication,
        show_progress=False,
    )
    panel = simulate_known_truth_panel(dgp, simulation)
    reward = RewardSpec.state_dependent(
        dgp.feature_matrix[:, 0, :],
        dgp.parameter_names,
        dgp.problem.num_actions,
    )
    model = AIRL(
        n_states=dgp.problem.num_states,
        n_actions=dgp.problem.num_actions,
        discount=dgp.problem.discount_factor,
        max_rounds=80 if smoke else 200,
        min_rounds=50 if smoke else 150,
        discriminator_steps=5,
        policy_step_size=0.1,
        generator_reward="f",
        compute_se=False,
        seed=17_200 + replication,
    ).fit(panel, transitions=np.asarray(dgp.transitions), reward=reward)
    oracle = solve_known_truth(dgp)
    reward_matrix = np.asarray(model.reward_matrix_)
    fitted_bellman = SoftBellmanOperator(dgp.problem, dgp.transitions).apply(
        jnp.asarray(reward_matrix), jnp.asarray(model.value_)
    )
    transfer_oracle = solve_counterfactual_oracle(dgp, "type_b")
    changed_transitions = np.asarray(transfer_oracle.counterfactual.transitions)
    transfer = model.counterfactual(
        transitions=changed_transitions,
        description="paper-side Type B transition change",
    )
    transfer_metrics = counterfactual_metrics(
        oracle_policy=transfer_oracle.counterfactual_solution.policy,
        oracle_value=transfer_oracle.counterfactual_solution.V,
        estimated_policy=transfer.counterfactual_policy,
        reward=dgp.homogeneous_reward,
        transitions=changed_transitions,
        discount_factor=dgp.problem.discount_factor,
        initial_distribution=dgp.initial_distribution,
        scale_parameter=dgp.problem.scale_parameter,
    )
    states = np.asarray(panel.get_all_states(), dtype=int)
    actions = np.asarray(panel.get_all_actions(), dtype=int)
    coverage = len(np.unique(np.stack([states, actions], axis=1), axis=0)) / (
        dgp.problem.num_states * dgp.problem.num_actions
    )
    return {
        "replication": replication,
        "panel_seed": simulation.seed,
        "training_seed": 17_200 + replication,
        "n_observations": panel.num_observations,
        "converged": bool(model.converged_),
        "iterations": int(model.n_iter_),
        "final_discriminator_loss": float(
            model.diagnostics_["optimization"]["final_discriminator_loss"]
        ),
        "reward_nrmse": normalized_rmse(model.reward_, dgp.homogeneous_reward[:, 0]),
        "policy_tv": float(
            np.mean(0.5 * np.abs(np.asarray(model.policy_) - np.asarray(oracle.policy)).sum(axis=1))
        ),
        "value_nrmse": normalized_rmse(model.value_, oracle.V),
        "q_nrmse": normalized_rmse(fitted_bellman.Q, oracle.Q),
        "transfer_policy_tv": float(transfer_metrics.policy.tv),
        "transfer_regret": float(transfer_metrics.regret),
        "state_action_coverage": coverage,
        "summary": model.summary(),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_requested": len(records),
        "n_successful": len(records),
        "n_converged": sum(record["converged"] for record in records),
    }
    for name in (
        "reward_nrmse",
        "policy_tv",
        "value_nrmse",
        "q_nrmse",
        "transfer_policy_tv",
        "transfer_regret",
        "state_action_coverage",
        "final_discriminator_loss",
    ):
        values = np.asarray([record[name] for record in records], dtype=float)
        summary[name] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(values.max()),
        }
    return summary


def checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    specs = {
        "all_fits_converged": (
            summary["n_converged"] == summary["n_requested"],
            "is",
            True,
        ),
        "median_reward_nrmse": (
            summary["reward_nrmse"]["median"],
            "<=",
            THRESHOLDS["maximum_median_reward_nrmse"],
        ),
        "p95_reward_nrmse": (
            summary["reward_nrmse"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_reward_nrmse"],
        ),
        "p95_policy_tv": (
            summary["policy_tv"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_policy_tv"],
        ),
        "p95_value_nrmse": (
            summary["value_nrmse"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_value_nrmse"],
        ),
        "p95_q_nrmse": (
            summary["q_nrmse"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_q_nrmse"],
        ),
        "p95_transfer_policy_tv": (
            summary["transfer_policy_tv"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_transfer_policy_tv"],
        ),
        "p95_transfer_regret": (
            summary["transfer_regret"]["p95"],
            "<=",
            THRESHOLDS["maximum_p95_transfer_regret"],
        ),
        "minimum_state_action_coverage": (
            summary["state_action_coverage"]["minimum"],
            ">=",
            THRESHOLDS["minimum_state_action_coverage"],
        ),
    }
    return [
        {
            "name": name,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(
                value is threshold
                if operator == "is"
                else value <= threshold
                if operator == "<="
                else value >= threshold
            ),
        }
        for name, (value, operator, threshold) in specs.items()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    n_replications = 1 if args.smoke else FULL_REPLICATIONS
    records = [fit_once(index, smoke=args.smoke) for index in range(n_replications)]
    summary = summarize(records)
    gate_results = checks(summary)
    receipt = {
        "estimator": "AIRL",
        "status": "ready" if all(gate["passed"] for gate in gate_results) else "failed",
        "paper_replication": False,
        "paper_boundary": (
            "Fu et al. Section 7.1 begins from MaxEnt IRL. This generated "
            "adversarial study is a paper-supported controlled replica, not an "
            "exact replication of published AIRL numbers."
        ),
        "target": "state-only reward recovery and changed-dynamics transfer",
        "configuration": {
            "cell": "airl_paper_identification",
            "states": 16,
            "actions": 4,
            "features": 4,
            "n_replications": n_replications,
            "individuals": 100 if args.smoke else 300,
            "periods": 30 if args.smoke else 80,
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
    print(f"wrote {args.output}")
    return 0 if receipt["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
