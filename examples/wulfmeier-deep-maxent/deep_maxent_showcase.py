"""Deep MaxEnt IRL: Neural reward beyond linear features.

Demonstrates the key contribution of Wulfmeier, Ondruska, and Posner (2016):
1. MCE-IRL assumes R = theta^T phi (linear in features). This cannot
   capture nonlinear reward structures.
2. Deep MaxEnt replaces the linear model with a neural network
   R = f_nn(s,a), keeping the maximum entropy framework but
   capturing nonlinear patterns.
3. The tradeoff: Deep MaxEnt loses interpretable parameters but
   gains flexibility for complex reward structures.

The showcase compares Deep MaxEnt IRL against linear MCE-IRL on an
Objectworld environment where the reward depends nonlinearly on
distances to colored objects.

Usage:
    python examples/wulfmeier-deep-maxent/deep_maxent_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.contrib.deep_maxent_irl import DeepMaxEntIRLEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.linear import LinearUtility
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


GRID_SIZE = 8


def policy_accuracy(est_policy, true_policy):
    """Fraction of states where argmax action matches."""
    est_actions = np.argmax(np.array(est_policy), axis=1)
    true_actions = np.argmax(np.array(true_policy), axis=1)
    return float(np.mean(est_actions == true_actions))


def run_diagnostics(panel, problem, utility):
    """Print pre-estimation diagnostics per CLAUDE.md."""
    features = np.array(utility.feature_matrix)
    n_states = problem.num_states
    n_actions = problem.num_actions
    n_features = features.shape[2]

    flat_features = features.reshape(n_states * n_actions, n_features)
    rank = np.linalg.matrix_rank(flat_features)
    nonzero_mask = np.any(flat_features != 0, axis=1)
    if nonzero_mask.sum() > 0:
        cond = np.linalg.cond(flat_features[nonzero_mask])
    else:
        cond = float("inf")

    all_states = panel.get_all_states()
    covered = len(np.unique(all_states))

    print(f"\n  Pre-estimation diagnostics:")
    print(f"    Feature matrix rank: {rank} / {n_features}")
    print(f"    Condition number: {cond:.1f}")
    print(f"    State coverage: {covered} / {n_states}")


def main():
    print("=" * 70)
    print("Deep MaxEnt IRL: Neural Reward Beyond Linear Features")
    print("Wulfmeier, Ondruska, and Posner (2016)")
    print("=" * 70)
    print()
    print("MCE-IRL: R = theta^T phi (linear, interpretable, limited).")
    print("Deep MaxEnt: R = f_nn(s,a) (neural, flexible, opaque).")

    env = ObjectworldEnvironment(
        grid_size=GRID_SIZE,
        n_colors=2,
        n_objects_per_color=3,
        discount_factor=0.9,
        feature_type="continuous",
        seed=42,
    )
    utility = LinearUtility.from_environment(env)
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
    print(f"  Features: {len(env.parameter_names)}")

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- MCE-IRL (linear) ----
    print(f"\n--- MCE-IRL (linear features) ---")
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05,
        outer_max_iter=300,
        verbose=False,
    ))
    mce_result = mce.estimate(panel, utility, problem, transitions)
    mce_time = time.time() - t0
    mce_acc = policy_accuracy(mce_result.policy, true_policy)
    print(f"  Time: {mce_time:.1f}s")
    print(f"  Policy accuracy: {mce_acc:.2%}")
    print(f"  Interpretable params: {len(env.parameter_names)}")
    results["mce_irl"] = {"policy_acc": mce_acc, "time": mce_time}

    # ---- Deep MaxEnt IRL (neural) ----
    print(f"\n--- Deep MaxEnt IRL (neural reward network) ---")
    t0 = time.time()
    deep = DeepMaxEntIRLEstimator(
        hidden_dims=[32, 32],
        embed_dim=8,
        lr=0.001,
        max_epochs=200,
        verbose=False,
    )
    deep_result = deep.estimate(panel, utility, problem, transitions)
    deep_time = time.time() - t0
    deep_acc = policy_accuracy(deep_result.policy, true_policy)
    print(f"  Time: {deep_time:.1f}s")
    print(f"  Policy accuracy: {deep_acc:.2%}")
    print(f"  Interpretable params: 0 (neural network)")
    results["deep_maxent"] = {"policy_acc": deep_acc, "time": deep_time}

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Linear vs Neural Reward")
    print("=" * 70)

    print(f"\n  {'Method':>20} {'Policy Acc':>12} {'Time':>8} {'Interpretable':>14}")
    print(f"  {'-'*58}")
    print(f"  {'MCE-IRL (linear)':>20} {results['mce_irl']['policy_acc']:>11.2%} "
          f"{results['mce_irl']['time']:>7.1f}s {'Yes':>14}")
    print(f"  {'Deep MaxEnt':>20} {results['deep_maxent']['policy_acc']:>11.2%} "
          f"{results['deep_maxent']['time']:>7.1f}s {'No':>14}")

    print(f"""
Deep MaxEnt keeps the maximum entropy framework (same occupancy-measure
gradient as MCE-IRL) but replaces the linear reward model with a neural
network. On Objectworld where the true reward is well-captured by the
continuous distance features, linear MCE-IRL may be sufficient. The
neural model's advantage appears when the feature specification is
wrong or incomplete. The cost of neural reward is loss of interpretable
parameters and standard errors.""")

    # Save
    path = Path(__file__).parent / "deep_maxent_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
