"""Max Margin IRL: The historical foundation of IRL.

Demonstrates the foundational contribution of Abbeel and Ng (2004, ICML):
1. This is the original IRL algorithm. It finds reward weights that
   make the expert's policy better than any other policy by a
   maximum margin.
2. Uses iterative constraint generation: find the most-violating
   policy, add a constraint, solve a QP.
3. Every other IRL method in the library improves on this in some way.

The showcase runs Max Margin IRL alongside MCE-IRL on a 5x5 gridworld
to show the conceptual difference: margin vs entropy.

Usage:
    python examples/taxi-gridworld/max_margin_textbook_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.gridworld import GridworldEnvironment
from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator
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
    single_action = 0
    for s in range(n_states):
        actions_at_s = all_actions[all_states == s]
        if len(actions_at_s) > 0 and len(np.unique(actions_at_s)) == 1:
            single_action += 1

    print(f"\n  Pre-estimation diagnostics:")
    print(f"    Feature matrix rank: {rank} / {n_features}")
    print(f"    Condition number: {cond:.1f}")
    print(f"    State coverage: {covered} / {n_states}")
    print(f"    Single-action states: {single_action}")


def main():
    print("=" * 70)
    print("Max Margin IRL: The Historical Foundation")
    print("Abbeel and Ng (2004, ICML)")
    print("=" * 70)
    print()
    print("The original IRL algorithm. Finds reward weights making the")
    print("expert's policy better than all others by a maximum margin.")
    print("Every other IRL method improves on this in some way.")

    env = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=0.99)
    utility = LinearUtility.from_environment(env)
    # Max Margin IRL needs ActionDependentReward, not LinearUtility
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
    print(f"  True: {[f'{float(p):.4f}' for p in true_params]}")
    print(f"  Parameters: {list(env.parameter_names)}")

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- Max Margin IRL ----
    print(f"\n--- Max Margin IRL (QP + constraint generation) ---")
    t0 = time.time()
    mm = MaxMarginIRLEstimator(
        max_iterations=50,
        margin_tol=1e-4,
        verbose=False,
    )
    mm_result = mm.estimate(panel, ad_utility, problem, transitions)
    mm_time = time.time() - t0
    mm_params = np.array(mm_result.parameters)
    mm_cos = reward_cosine(mm_params, true_params)
    mm_acc = policy_accuracy(mm_result.policy, true_policy)
    print(f"  Time: {mm_time:.1f}s")
    print(f"  Reward cosine: {mm_cos:.4f}")
    print(f"  Policy accuracy: {mm_acc:.2%}")
    print(f"  Params: {[f'{float(p):.4f}' for p in mm_params]}")
    n_constraints = mm_result.metadata.get("num_iterations", "?")
    print(f"  Constraint generation iterations: {n_constraints}")
    results["max_margin"] = {
        "params": mm_params.tolist(),
        "cosine": mm_cos,
        "policy_acc": mm_acc,
        "time": mm_time,
    }

    # ---- MCE-IRL (modern alternative) ----
    print(f"\n--- MCE-IRL (entropy maximization) ---")
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05,
        outer_max_iter=300,
        verbose=False,
    ))
    mce_result = mce.estimate(panel, utility, problem, transitions)
    mce_time = time.time() - t0
    mce_params = np.array(mce_result.parameters)
    mce_cos = reward_cosine(mce_params, true_params)
    mce_acc = policy_accuracy(mce_result.policy, true_policy)
    print(f"  Time: {mce_time:.1f}s")
    print(f"  Reward cosine: {mce_cos:.4f}")
    print(f"  Policy accuracy: {mce_acc:.2%}")
    print(f"  Params: {[f'{float(p):.4f}' for p in mce_params]}")
    results["mce_irl"] = {
        "params": mce_params.tolist(),
        "cosine": mce_cos,
        "policy_acc": mce_acc,
        "time": mce_time,
    }

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Margin vs Entropy: Two Philosophies of IRL")
    print("=" * 70)

    print(f"\n  {'Method':>15} {'Cosine':>8} {'Policy Acc':>12} {'Time':>8}")
    print(f"  {'-'*46}")
    for label, key in [("Max Margin", "max_margin"), ("MCE-IRL", "mce_irl")]:
        r = results[key]
        print(f"  {label:>15} {r['cosine']:>8.4f} {r['policy_acc']:>11.2%} "
              f"{r['time']:>7.1f}s")

    print(f"""
Max Margin IRL (Abbeel and Ng 2004) finds weights that make the
expert's policy better than any alternative by a margin. It solves
a QP with iterative constraint generation.

MCE-IRL (Ziebart 2010) finds weights that maximize the entropy of
the policy while matching expert feature expectations. It has a
well-defined likelihood and produces standard errors.

Max Margin IRL is the starting point. MCE-IRL improved it by adding
a probabilistic model (entropy regularization, proper likelihood).
Understanding Max Margin IRL is prerequisite for understanding why
entropy-based methods were invented.""")

    # Save
    path = Path(__file__).parent / "max_margin_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
