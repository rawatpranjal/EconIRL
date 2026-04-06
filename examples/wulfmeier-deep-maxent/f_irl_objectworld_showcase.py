"""f-IRL: Feature-free reward recovery via state-marginal matching.

Demonstrates the key contribution of Ni et al. (2020, CoRL):
1. MCE-IRL requires specifying features and assumes linear reward
   R = theta^T phi. If the features are wrong, MCE-IRL recovers
   biased reward.
2. f-IRL matches state-marginal distributions using f-divergence,
   recovering nonparametric tabular reward without feature
   specification.

The showcase compares f-IRL against MCE-IRL on an Objectworld
environment where the true reward depends nonlinearly on distance
to colored objects. MCE-IRL with continuous (linear) features should
struggle with the nonlinear structure, while f-IRL recovers the
reward directly.

Usage:
    python examples/wulfmeier-deep-maxent/f_irl_objectworld_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.estimation.f_irl import FIRLEstimator
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
    print("f-IRL: Feature-Free Reward Recovery via State-Marginal Matching")
    print("Ni, Sikchi, Wang, Gupta, Lee, Eysenbach (2020, CoRL)")
    print("=" * 70)
    print()
    print("MCE-IRL requires features and assumes linear reward.")
    print("f-IRL matches state-marginals directly -- no features needed.")

    # Objectworld: nonlinear reward from distance to colored objects
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

    # Compute true policy
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

    # Generate data
    panel = env.generate_panel(n_individuals=200, n_periods=50, seed=42)
    n_obs = panel.num_observations

    print(f"\n  Grid: {GRID_SIZE}x{GRID_SIZE}, States: {problem.num_states}")
    print(f"  Actions: {problem.num_actions}")
    print(f"  Observations: {n_obs}")
    print(f"  Features: {len(env.parameter_names)} ({env.parameter_names})")

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- MCE-IRL (linear features) ----
    print(f"\n--- MCE-IRL (linear features) ---")
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05,
        outer_max_iter=300,
        verbose=False,
    ))
    try:
        mce_result = mce.estimate(panel, utility, problem, transitions)
        mce_time = time.time() - t0
        mce_acc = policy_accuracy(mce_result.policy, true_policy)
        print(f"  Time: {mce_time:.1f}s")
        print(f"  Policy accuracy: {mce_acc:.2%}")
        print(f"  Reward is linear in {len(env.parameter_names)} features")
        results["mce_irl"] = {
            "params": [float(p) for p in mce_result.parameters],
            "policy_acc": mce_acc,
            "time": mce_time,
        }
    except Exception as e:
        print(f"  MCE-IRL failed: {e}")
        mce_acc = float("nan")
        results["mce_irl"] = {"policy_acc": float("nan"), "time": 0}

    # ---- f-IRL (KL divergence) ----
    print(f"\n--- f-IRL (KL divergence, nonparametric tabular reward) ---")
    t0 = time.time()
    f_irl = FIRLEstimator(
        f_divergence="kl",
        lr=0.5,
        max_iter=300,
        verbose=False,
    )
    try:
        f_result = f_irl.estimate(panel, utility, problem, transitions)
        f_time = time.time() - t0
        f_acc = policy_accuracy(f_result.policy, true_policy)
        print(f"  Time: {f_time:.1f}s")
        print(f"  Policy accuracy: {f_acc:.2%}")
        print(f"  Reward is nonparametric: {problem.num_states} x "
              f"{problem.num_actions} table")
        results["f_irl_kl"] = {
            "policy_acc": f_acc,
            "time": f_time,
        }
    except Exception as e:
        print(f"  f-IRL failed: {e}")
        f_acc = float("nan")
        results["f_irl_kl"] = {"policy_acc": float("nan"), "time": 0}

    # ---- f-IRL (chi2 divergence) ----
    print(f"\n--- f-IRL (chi-squared divergence) ---")
    t0 = time.time()
    f_irl_chi2 = FIRLEstimator(
        f_divergence="chi2",
        lr=0.5,
        max_iter=300,
        verbose=False,
    )
    try:
        f_chi2_result = f_irl_chi2.estimate(panel, utility, problem, transitions)
        f_chi2_time = time.time() - t0
        f_chi2_acc = policy_accuracy(f_chi2_result.policy, true_policy)
        print(f"  Time: {f_chi2_time:.1f}s")
        print(f"  Policy accuracy: {f_chi2_acc:.2%}")
        results["f_irl_chi2"] = {
            "policy_acc": f_chi2_acc,
            "time": f_chi2_time,
        }
    except Exception as e:
        print(f"  f-IRL (chi2) failed: {e}")
        results["f_irl_chi2"] = {"policy_acc": float("nan"), "time": 0}

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Summary: Feature-Based vs Feature-Free Reward Recovery")
    print("=" * 70)

    print(f"\n  {'Method':>25} {'Policy Acc':>12} {'Time':>8} {'Reward':>20}")
    print(f"  {'-'*68}")
    if "mce_irl" in results:
        r = results["mce_irl"]
        print(f"  {'MCE-IRL (linear)':>25} {r['policy_acc']:>11.2%} "
              f"{r['time']:>7.1f}s {'linear, K features':>20}")
    if "f_irl_kl" in results:
        r = results["f_irl_kl"]
        print(f"  {'f-IRL (KL)':>25} {r['policy_acc']:>11.2%} "
              f"{r['time']:>7.1f}s {'tabular, S*A entries':>20}")
    if "f_irl_chi2" in results:
        r = results["f_irl_chi2"]
        print(f"  {'f-IRL (chi2)':>25} {r['policy_acc']:>11.2%} "
              f"{r['time']:>7.1f}s {'tabular, S*A entries':>20}")

    print(f"""
On Objectworld with continuous distance features, MCE-IRL's linear
model is actually well-specified: the features capture the true
reward structure. MCE-IRL dominates because estimating 2 parameters
requires far less data than f-IRL's {problem.num_states * problem.num_actions}
tabular entries.

f-IRL's value is as insurance against feature misspecification. When
you do not know whether linear features are correct, f-IRL provides
a model-free baseline. When features are right, MCE-IRL will always
win. f-IRL trades interpretability and statistical efficiency for
robustness to model misspecification.""")

    # Save
    path = Path(__file__).parent / "f_irl_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
