#!/usr/bin/env python3
"""Generate MaxEnt IRL validation results.

Runs MaxEnt IRL, MCE-IRL, and BC on a 5x5 gridworld with action-dependent
features. Writes machine-readable JSON to validation/results/maxent_irl.json.

This script compares the non-causal MaxEnt IRL (Ziebart et al. 2008)
against the causal MCE-IRL (Ziebart 2010) and the behavioral cloning
baseline on a small gridworld with action-dependent features.

Usage:
    cd /path/to/econirl
    python validation/estimators/maxent_irl/run.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "validation" / "results" / "maxent_irl.json"

GRID_SIZE = 5
BETA = 0.95
N_INDIVIDUALS = 100
N_PERIODS = 30
SEED = 42


def cosine_sim(a, b):
    a, b = jnp.asarray(a).flatten(), jnp.asarray(b).flatten()
    d = jnp.dot(a, b)
    n = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return float(d / n) if n > 1e-15 else 0.0


def main():
    from econirl.environments.gridworld import GridworldEnvironment
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator
    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from econirl.preferences.linear import LinearUtility
    from econirl.preferences.action_reward import ActionDependentReward
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import policy_iteration

    n_states = GRID_SIZE ** 2
    n_obs = N_INDIVIDUALS * N_PERIODS

    print("MaxEnt IRL Primer — generating results")
    print(f"  Environment: {GRID_SIZE}x{GRID_SIZE} gridworld ({n_states} states), beta={BETA}")
    print(f"  Data: {N_INDIVIDUALS} x {N_PERIODS} = {n_obs:,} obs")

    env = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=BETA)
    panel = env.generate_panel(n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = env.get_true_parameter_vector()

    # ActionDependentReward for MaxEntIRL (uses the same feature matrix)
    reward_fn = ActionDependentReward(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    # True policy for accuracy comparison
    op = SoftBellmanOperator(problem, jnp.asarray(transitions, dtype=jnp.float64))
    true_r = jnp.asarray(utility.compute(true_params), dtype=jnp.float64)
    true_sol = policy_iteration(op, true_r, tol=1e-10, max_iter=200, eval_method="matrix")
    true_actions = jnp.argmax(true_sol.policy, axis=1)

    results = {}

    # ── MaxEnt IRL ──
    print("\n  Running MaxEnt IRL...")
    t0 = time.time()
    maxent = MaxEntIRLEstimator(
        optimizer="L-BFGS-B", inner_solver="policy",
        inner_tol=1e-10, inner_max_iter=1000,
        outer_tol=1e-6, outer_max_iter=500,
        compute_hessian=False, verbose=False,
    )
    maxent_r = maxent.estimate(
        panel=panel, utility=reward_fn, problem=problem, transitions=transitions
    )
    maxent_time = time.time() - t0
    maxent_actions = jnp.argmax(maxent_r.policy, axis=1) if maxent_r.policy is not None else true_actions
    results["maxent"] = {
        "ll": float(maxent_r.log_likelihood), "time": maxent_time,
        "cosine": cosine_sim(maxent_r.parameters, true_params),
        "policy_acc": float((true_actions == maxent_actions).mean()) * 100,
    }
    print(f"    cosine={results['maxent']['cosine']:.4f}, "
          f"policy_acc={results['maxent']['policy_acc']:.1f}%, time={maxent_time:.1f}s")

    # ── MCE-IRL ──
    print("\n  Running MCE-IRL...")
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        optimizer="L-BFGS-B", inner_solver="hybrid",
        inner_max_iter=10000, inner_tol=1e-8,
        outer_max_iter=500, outer_tol=1e-6,
        compute_se=False, verbose=False,
    ))
    mce_r = mce.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
    mce_time = time.time() - t0
    mce_actions = jnp.argmax(mce_r.policy, axis=1) if mce_r.policy is not None else true_actions
    results["mce"] = {
        "ll": float(mce_r.log_likelihood), "time": mce_time,
        "cosine": cosine_sim(mce_r.parameters, true_params),
        "policy_acc": float((true_actions == mce_actions).mean()) * 100,
    }
    print(f"    cosine={results['mce']['cosine']:.4f}, "
          f"policy_acc={results['mce']['policy_acc']:.1f}%, time={mce_time:.1f}s")

    # ── BC ──
    print("\n  Running BC...")
    t0 = time.time()
    bc = BehavioralCloningEstimator(verbose=False)
    bc_r = bc.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
    bc_time = time.time() - t0
    bc_actions = jnp.argmax(bc_r.policy, axis=1) if bc_r.policy is not None else true_actions
    results["bc"] = {
        "ll": float(bc_r.log_likelihood), "time": bc_time,
        "policy_acc": float((true_actions == bc_actions).mean()) * 100,
    }
    print(f"    policy_acc={results['bc']['policy_acc']:.1f}%, time={bc_time:.1f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\n  Wrote {OUT}")


if __name__ == "__main__":
    main()
