"""GCL: Trajectory-level IRL with neural cost functions.

Demonstrates the key contribution of Finn, Levine, and Abbeel (2016, ICML):
1. MCE-IRL uses per-step feature matching with hand-crafted linear features.
2. GCL uses a neural cost function trained from trajectory-level
   demonstrations via importance sampling. This enables nonlinear
   reward learning without feature specification.
3. GCL sits between linear MCE-IRL and full Deep MaxEnt: it learns
   nonlinear reward via neural networks but uses trajectory-level
   importance sampling as the training mechanism.

The showcase compares GCL against linear MCE-IRL on a gridworld where
both use the same environment but GCL uses a neural cost function.

Usage:
    python examples/wulfmeier-deep-maxent/gcl_nonlinear_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.gridworld import GridworldEnvironment
from econirl.contrib.gcl import GCLEstimator, GCLConfig
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.linear import LinearUtility
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


GRID_SIZE = 5


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
    cond = np.linalg.cond(flat_features[nonzero_mask])

    all_states = panel.get_all_states()
    covered = len(np.unique(all_states))

    print(f"\n  Pre-estimation diagnostics:")
    print(f"    Feature matrix rank: {rank} / {n_features}")
    print(f"    Condition number: {cond:.1f}")
    print(f"    State coverage: {covered} / {n_states}")


def main():
    print("=" * 70)
    print("GCL: Trajectory-Level IRL with Neural Cost Functions")
    print("Finn, Levine, and Abbeel (2016, ICML)")
    print("=" * 70)
    print()
    print("MCE-IRL: per-step feature matching, linear reward.")
    print("GCL: trajectory-level importance sampling, neural cost.")

    env = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=0.99)
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

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- MCE-IRL (linear features) ----
    print(f"\n--- MCE-IRL (linear features, per-step matching) ---")
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
    results["mce_irl"] = {"policy_acc": mce_acc, "time": mce_time}

    # ---- GCL (neural cost) ----
    print(f"\n--- GCL (neural cost, trajectory importance sampling) ---")
    t0 = time.time()
    gcl = GCLEstimator(config=GCLConfig(
        hidden_dims=[32, 32],
        cost_lr=0.001,
        max_iterations=100,
        n_sample_trajectories=50,
        verbose=False,
    ))
    gcl_result = gcl.estimate(panel, utility, problem, transitions)
    gcl_time = time.time() - t0
    gcl_acc = policy_accuracy(gcl_result.policy, true_policy)
    print(f"  Time: {gcl_time:.1f}s")
    print(f"  Policy accuracy: {gcl_acc:.2%}")
    results["gcl"] = {"policy_acc": gcl_acc, "time": gcl_time}

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Summary: Per-Step vs Trajectory-Level IRL")
    print("=" * 70)

    print(f"\n  {'Method':>20} {'Policy Acc':>12} {'Time':>8} "
          f"{'Reward':>20}")
    print(f"  {'-'*64}")
    print(f"  {'MCE-IRL (linear)':>20} {results['mce_irl']['policy_acc']:>11.2%} "
          f"{results['mce_irl']['time']:>7.1f}s {'linear, K params':>20}")
    print(f"  {'GCL (neural)':>20} {results['gcl']['policy_acc']:>11.2%} "
          f"{results['gcl']['time']:>7.1f}s {'neural network':>20}")

    print(f"""
GCL learns a neural cost function from trajectory-level demonstrations
using importance sampling to estimate the partition function. On this
linear-reward gridworld, MCE-IRL with correct features should dominate
because it has the right model specification. GCL's advantage appears
on problems with nonlinear reward structures where linear features
cannot capture the true cost.""")

    # Save
    path = Path(__file__).parent / "gcl_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
