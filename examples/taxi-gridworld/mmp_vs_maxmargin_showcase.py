"""Max Margin Planning vs Max Margin IRL: Scalable margin-based IRL.

Demonstrates the key contribution of Ratliff, Bagnell, and Zinkevich (2006):
1. Max Margin IRL (Abbeel and Ng 2004) solves a QP with constraint
   generation that grows at each iteration.
2. MMP replaces the QP with subgradient descent and loss-augmented
   inference. Per-iteration cost is one MDP solve, constant across
   iterations.
3. Both methods produce the same kind of margin-based solution, but
   MMP is designed to scale better.

The showcase compares both on a 5x5 gridworld alongside MCE-IRL as
a reference.

Usage:
    python examples/taxi-gridworld/mmp_vs_maxmargin_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.gridworld import GridworldEnvironment
from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator
from econirl.contrib.max_margin_planning import MaxMarginPlanningEstimator, MMPConfig
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


GRID_SIZE = 5


def policy_accuracy(est_policy, true_policy):
    """Fraction of states where argmax action matches."""
    est_actions = np.argmax(np.array(est_policy), axis=1)
    true_actions = np.argmax(np.array(true_policy), axis=1)
    return float(np.mean(est_actions == true_actions))


def reward_cosine(est_params, true_params):
    """Cosine similarity between estimated and true reward parameters."""
    e = np.array(est_params, dtype=float)
    t = np.array(true_params, dtype=float)
    denom = np.linalg.norm(e) * np.linalg.norm(t)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(e, t) / denom)


def run_diagnostics(panel, problem, utility):
    """Print pre-estimation diagnostics per CLAUDE.md."""
    features = np.array(utility.feature_matrix)
    n_states = problem.num_states
    n_actions = problem.num_actions
    n_features = features.shape[2]

    flat_features = features.reshape(n_states * n_actions, n_features)
    rank = np.linalg.matrix_rank(flat_features)
    nonzero_mask = np.any(flat_features != 0, axis=1)
    cond = np.linalg.cond(flat_features[nonzero_mask])

    all_states = panel.get_all_states()
    all_actions = panel.get_all_actions()
    covered = len(np.unique(all_states))

    print(f"\n  Pre-estimation diagnostics:")
    print(f"    Feature matrix rank: {rank} / {n_features}")
    print(f"    Condition number: {cond:.1f}")
    print(f"    State coverage: {covered} / {n_states}")


def main():
    print("=" * 70)
    print("MMP vs Max Margin IRL: Scalable Margin-Based IRL")
    print("Ratliff et al. (2006) vs Abbeel and Ng (2004)")
    print("=" * 70)
    print()
    print("Max Margin IRL: QP with constraint generation (grows per iter).")
    print("MMP: subgradient descent with loss-augmented inference (constant).")

    env = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=0.99)
    utility = LinearUtility.from_environment(env)
    ad_utility = ActionDependentReward(
        feature_matrix=env.feature_matrix,
        parameter_names=list(env.parameter_names),
    )
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = env.get_true_parameter_vector()

    # True policy
    operator = SoftBellmanOperator(
        problem, jnp.asarray(transitions, dtype=jnp.float64)
    )
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = policy_iteration(
        operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_policy = np.array(true_result.policy)

    panel = env.generate_panel(n_individuals=200, n_periods=50, seed=42)

    print(f"\n  Grid: {GRID_SIZE}x{GRID_SIZE}, States: {problem.num_states}")
    print(f"  Observations: {panel.num_observations}")

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- Max Margin IRL (QP) ----
    print(f"\n--- Max Margin IRL (QP + constraint generation) ---")
    t0 = time.time()
    mm = MaxMarginIRLEstimator(max_iterations=50, verbose=False)
    mm_result = mm.estimate(panel, ad_utility, problem, transitions)
    mm_time = time.time() - t0
    mm_cos = reward_cosine(mm_result.parameters, true_params)
    mm_acc = policy_accuracy(mm_result.policy, true_policy)
    print(f"  Time: {mm_time:.1f}s, Cosine: {mm_cos:.4f}, "
          f"Policy: {mm_acc:.2%}")
    results["max_margin"] = {
        "cosine": mm_cos, "policy_acc": mm_acc, "time": mm_time,
    }

    # ---- MMP (subgradient) ----
    print(f"\n--- MMP (subgradient + loss-augmented inference) ---")
    t0 = time.time()
    mmp = MaxMarginPlanningEstimator(config=MMPConfig(
        learning_rate=0.1,
        max_iterations=200,
        verbose=False,
    ))
    mmp_result = mmp.estimate(panel, ad_utility, problem, transitions)
    mmp_time = time.time() - t0
    mmp_cos = reward_cosine(mmp_result.parameters, true_params)
    mmp_acc = policy_accuracy(mmp_result.policy, true_policy)
    print(f"  Time: {mmp_time:.1f}s, Cosine: {mmp_cos:.4f}, "
          f"Policy: {mmp_acc:.2%}")
    results["mmp"] = {
        "cosine": mmp_cos, "policy_acc": mmp_acc, "time": mmp_time,
    }

    # ---- MCE-IRL (reference) ----
    print(f"\n--- MCE-IRL (entropy-based, reference) ---")
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05, outer_max_iter=300, verbose=False,
    ))
    mce_result = mce.estimate(panel, utility, problem, transitions)
    mce_time = time.time() - t0
    mce_cos = reward_cosine(mce_result.parameters, true_params)
    mce_acc = policy_accuracy(mce_result.policy, true_policy)
    print(f"  Time: {mce_time:.1f}s, Cosine: {mce_cos:.4f}, "
          f"Policy: {mce_acc:.2%}")
    results["mce_irl"] = {
        "cosine": mce_cos, "policy_acc": mce_acc, "time": mce_time,
    }

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Three Approaches to IRL")
    print("=" * 70)

    print(f"\n  {'Method':>15} {'Cosine':>8} {'Policy Acc':>12} {'Time':>8} "
          f"{'Algorithm':>25}")
    print(f"  {'-'*72}")
    for label, key, algo in [
        ("Max Margin", "max_margin", "QP + constraints"),
        ("MMP", "mmp", "Subgradient descent"),
        ("MCE-IRL", "mce_irl", "Feature matching"),
    ]:
        r = results[key]
        print(f"  {label:>15} {r['cosine']:>8.4f} {r['policy_acc']:>11.2%} "
              f"{r['time']:>7.1f}s {algo:>25}")

    print(f"""
Max Margin IRL and MMP produce the same kind of solution (margin-based
reward weights) via different algorithms. MMP replaces the QP with
subgradient descent, making per-iteration cost constant. On this
{GRID_SIZE}x{GRID_SIZE} grid the difference is small, but MMP scales
better because it avoids accumulating constraints.

MCE-IRL uses entropy regularization instead of margins, producing a
well-defined likelihood and proper standard errors.""")

    # Save
    path = Path(__file__).parent / "mmp_vs_maxmargin_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
