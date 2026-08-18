#!/usr/bin/env python3
"""Calibrate AIRL2's label-aligned cluster bootstrap on known truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from econirl import AIRL2
from validation.known_truth import (
    ContentHeterogeneityKnownTruthConfig,
    SimulationConfig,
    build_known_truth_dgp,
    simulate_known_truth_panel,
    solve_known_truth,
)

ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = ROOT / "validation" / "results" / "airl2_bootstrap.json"

# Frozen before the held-out qualification run. The multiplier was selected
# after a separate six-panel pilot showed stable widths but undercoverage from
# unadjusted percentile intervals. The gates below are unchanged from that pilot.
N_PANELS = 6
N_BOOTSTRAP = 8
ALPHA = 0.10
GATES = {
    "panel_success_rate": (">=", 1.0),
    "bootstrap_success_rate": (">=", 1.0),
    "reward_coverage": (">=", 0.70),
    "policy_coverage": (">=", 0.75),
    "prior_coverage": (">=", 0.65),
    "max_width_cv": ("<=", 0.75),
}


def _interval_arrays(model: AIRL2) -> tuple[np.ndarray, ...]:
    named = model.conf_int(alpha=ALPHA)
    flat = np.asarray(list(named.values()), dtype=float)
    segment_cells = model.num_segments * model.n_states * model.n_actions
    reward = flat[:segment_cells].reshape(model.num_segments, model.n_states, model.n_actions, 2)
    policy = flat[segment_cells : 2 * segment_cells].reshape(
        model.num_segments, model.n_states, model.n_actions, 2
    )
    priors = flat[2 * segment_cells :].reshape(model.num_segments, 2)
    return (
        reward[..., 0],
        reward[..., 1],
        policy[..., 0],
        policy[..., 1],
        priors[:, 0],
        priors[:, 1],
    )


def _gate(name: str, value: float) -> dict[str, Any]:
    operator, threshold = GATES[name]
    passed = value >= threshold if operator == ">=" else value <= threshold
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        successful = sum(record["bootstrap_successful"] for record in payload["records"])
        requested = payload["n_panels"] * payload["n_bootstrap"]
        print(
            f"bootstrap result: {len(payload['records'])}/{payload['n_panels']} panels, "
            f"{successful}/{requested} refits"
        )
        return

    config = ContentHeterogeneityKnownTruthConfig(seed=4506)
    dgp = build_known_truth_dgp(config)
    true_rewards = np.asarray(dgp.reward_matrix)
    true_policies = np.stack(
        [
            np.asarray(solve_known_truth(dgp, segment_index=segment).policy)
            for segment in range(dgp.num_segments)
        ]
    )
    true_priors = np.asarray(dgp.segment_probabilities)
    reward_mask = np.ones(true_rewards.shape, dtype=bool)
    reward_mask[:, :, config.exit_action] = False
    reward_mask[:, config.absorbing_state, :] = False

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    reward_hits: list[np.ndarray] = []
    policy_hits: list[np.ndarray] = []
    prior_hits: list[np.ndarray] = []
    reward_widths: list[float] = []
    policy_widths: list[float] = []
    prior_widths: list[float] = []
    successful_draws = 0

    for panel_index in range(N_PANELS):
        panel_seed = 12100 + panel_index
        fit_seed = 13100 + panel_index
        bootstrap_seed = 14100 + panel_index
        try:
            panel = simulate_known_truth_panel(
                dgp,
                SimulationConfig(n_individuals=250, n_periods=16, seed=panel_seed),
            )
            model = AIRL2(
                n_states=dgp.problem.num_states,
                n_actions=dgp.problem.num_actions,
                exit_action=config.exit_action,
                absorbing_state=config.absorbing_state,
                discount=config.discount_factor,
                num_segments=dgp.num_segments,
                reward_type="linear",
                reward_lr=0.001,
                discriminator_steps=2,
                policy_step_size=0.1,
                generator_reward="f",
                max_airl_rounds=3,
                min_airl_rounds=1,
                max_em_iterations=8,
                airl_convergence_tol=0.01,
                em_convergence_tol=0.01,
                consistency_weight=1.0,
                prior_smoothing=0.01,
                prior_min=0.05,
                prior_damping=0.8,
                initialization="behavioral_anchor",
                initialization_smoothing=1.0,
                initialization_l2_penalty=10.0,
                compute_se=True,
                n_bootstrap=N_BOOTSTRAP,
                seed=fit_seed,
                se_seed=bootstrap_seed,
            )
            model.fit(
                panel,
                transitions=np.asarray(dgp.transitions),
                features=np.asarray(dgp.feature_matrix),
            )
            assert model.bootstrap_ is not None
            bootstrap = model.bootstrap_
            aligned_rewards, aligned_policies, aligned_priors = model._align_segment_draw(
                true_rewards,
                true_policies,
                true_priors,
            )

            assert bootstrap.segment_prior_draws is not None
            (
                reward_low,
                reward_high,
                policy_low,
                policy_high,
                prior_low,
                prior_high,
            ) = _interval_arrays(model)
            reward_hit = (reward_low <= aligned_rewards) & (aligned_rewards <= reward_high)
            policy_hit = (policy_low <= aligned_policies) & (aligned_policies <= policy_high)
            prior_hit = (prior_low <= aligned_priors) & (aligned_priors <= prior_high)

            reward_hits.append(reward_hit[reward_mask])
            policy_hits.append(policy_hit.ravel())
            prior_hits.append(prior_hit.ravel())
            reward_widths.append(float(np.mean((reward_high - reward_low)[reward_mask])))
            policy_widths.append(float(np.mean(policy_high - policy_low)))
            prior_widths.append(float(np.mean(prior_high - prior_low)))
            successful_draws += bootstrap.n_successful
            records.append(
                {
                    "panel_seed": panel_seed,
                    "fit_seed": fit_seed,
                    "bootstrap_seed": bootstrap_seed,
                    "point_converged": bool(model.converged_),
                    "point_iterations": int(model.n_iter_),
                    "bootstrap_successful": bootstrap.n_successful,
                    "bootstrap_requested": bootstrap.n_requested,
                    "reward_coverage": float(np.mean(reward_hit[reward_mask])),
                    "policy_coverage": float(np.mean(policy_hit)),
                    "prior_coverage": float(np.mean(prior_hit)),
                    "reward_mean_width": reward_widths[-1],
                    "policy_mean_width": policy_widths[-1],
                    "prior_mean_width": prior_widths[-1],
                }
            )
            print(
                f"panel {panel_index + 1}/{N_PANELS}: "
                f"{bootstrap.n_successful}/{bootstrap.n_requested} draws"
            )
        except Exception as exc:
            errors.append(f"panel {panel_index}: {type(exc).__name__}: {exc}")
            print(errors[-1])

    def coverage(values: list[np.ndarray]) -> float:
        return float(np.mean(np.concatenate(values))) if values else 0.0

    def width_cv(values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        return float(np.std(array, ddof=1) / np.mean(array)) if len(array) > 1 else float("inf")

    metrics = {
        "panel_success_rate": len(records) / N_PANELS,
        "bootstrap_success_rate": successful_draws / (N_PANELS * N_BOOTSTRAP),
        "reward_coverage": coverage(reward_hits),
        "policy_coverage": coverage(policy_hits),
        "prior_coverage": coverage(prior_hits),
        "reward_width_cv": width_cv(reward_widths),
        "policy_width_cv": width_cv(policy_widths),
        "prior_width_cv": width_cv(prior_widths),
    }
    metrics["max_width_cv"] = max(
        metrics["reward_width_cv"],
        metrics["policy_width_cv"],
        metrics["prior_width_cv"],
    )
    gates = [_gate(name, metrics[name]) for name in GATES]
    payload = {
        "estimator": "AIRL2",
        "method": "pairs_cluster_label_aligned",
        "unit": "individual",
        "n_panels": N_PANELS,
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "calibration_multiplier": AIRL2._BOOTSTRAP_CALIBRATION_MULTIPLIER,
        "calibration_note": (
            "A separate six-panel pilot on seeds 6100-6105 completed 48/48 "
            "refits but raw percentile coverage was 0.5597 for reward, 0.5515 "
            "for policy, and 0.0000 for priors. A first held-out block on seeds "
            "9100-9105 showed 0.9972 reward and 0.9923 policy coverage, but "
            "only 0.3333 prior coverage because EM prior damping suppressed "
            "refit variance. The 4.0 normal-SE multiplier and binomial "
            "individual-cluster SE floor were frozen before these final seeds."
        ),
        "metrics": metrics,
        "gates": gates,
        "records": records,
        "errors": errors,
        "status": "pass" if all(gate["passed"] for gate in gates) else "fail",
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {RESULT_PATH}")
    if args.enforce_gates and payload["status"] != "pass":
        failed = [gate["name"] for gate in gates if not gate["passed"]]
        raise SystemExit(f"AIRL2 bootstrap gates failed: {', '.join(failed)}")
    print(
        f"bootstrap result: {len(records)}/{N_PANELS} panels, "
        f"{successful_draws}/{N_PANELS * N_BOOTSTRAP} refits"
    )


if __name__ == "__main__":
    main()
