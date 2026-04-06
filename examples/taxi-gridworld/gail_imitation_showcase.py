"""GAIL: Direct policy imitation without reward recovery.

Demonstrates the key contribution of Ho and Ermon (2016, NeurIPS):
1. Traditional IRL recovers reward then derives policy (two-step).
2. GAIL directly matches the expert's occupancy measure via a GAN
   discriminator, producing a policy without explicit reward modeling.
3. GAIL is the right tool when the goal is imitation, not
   understanding (no counterfactuals or welfare analysis possible).

The showcase compares GAIL against MCE-IRL and Behavioral Cloning
on a 5x5 gridworld, measuring policy accuracy. BC is the lower bound
(no MDP structure), MCE-IRL is the structural alternative.

Usage:
    python examples/taxi-gridworld/gail_imitation_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.gridworld import GridworldEnvironment
from econirl.contrib.gail import GAILEstimator, GAILConfig
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
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
    print("GAIL: Direct Policy Imitation Without Reward Recovery")
    print("Ho and Ermon (2016, NeurIPS)")
    print("=" * 70)
    print()
    print("BC: supervised P(a|s) from frequencies. No MDP structure.")
    print("MCE-IRL: recover reward via feature matching, then derive policy.")
    print("GAIL: match occupancy measure via GAN. Policy without reward.")

    env = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=0.99)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    # True policy
    operator = SoftBellmanOperator(
        problem, jnp.asarray(transitions, dtype=jnp.float64)
    )
    true_params = env.get_true_parameter_vector()
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = policy_iteration(
        operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_policy = np.array(true_result.policy)

    panel = env.generate_panel(n_individuals=200, n_periods=50, seed=42)
    n_obs = panel.num_observations

    print(f"\n  Grid: {GRID_SIZE}x{GRID_SIZE}, States: {problem.num_states}")
    print(f"  Observations: {n_obs}")

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- Behavioral Cloning (lower bound) ----
    print(f"\n--- Behavioral Cloning (lower bound) ---")
    t0 = time.time()
    bc = BehavioralCloningEstimator(verbose=False)
    bc_result = bc.estimate(panel, utility, problem, transitions)
    bc_time = time.time() - t0
    bc_acc = policy_accuracy(bc_result.policy, true_policy)
    print(f"  Time: {bc_time:.1f}s")
    print(f"  Policy accuracy: {bc_acc:.2%}")
    print(f"  Interpretable reward: No (just action frequencies)")
    results["bc"] = {"policy_acc": bc_acc, "time": bc_time, "has_reward": False}

    # ---- MCE-IRL (structural) ----
    print(f"\n--- MCE-IRL (feature matching, reward + policy) ---")
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
    print(f"  Interpretable reward: Yes (linear, {len(env.parameter_names)} params)")
    results["mce_irl"] = {"policy_acc": mce_acc, "time": mce_time, "has_reward": True}

    # ---- GAIL ----
    print(f"\n--- GAIL (occupancy matching via GAN) ---")
    t0 = time.time()
    gail = GAILEstimator(config=GAILConfig(
        discriminator_type="tabular",
        discriminator_lr=0.01,
        discriminator_steps=5,
        max_rounds=100,
        compute_se=False,
        verbose=False,
    ))
    gail_result = gail.estimate(panel, utility, problem, transitions)
    gail_time = time.time() - t0
    gail_acc = policy_accuracy(gail_result.policy, true_policy)
    print(f"  Time: {gail_time:.1f}s")
    print(f"  Policy accuracy: {gail_acc:.2%}")
    print(f"  Interpretable reward: No (discriminator, not transferable)")
    results["gail"] = {"policy_acc": gail_acc, "time": gail_time, "has_reward": False}

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Summary: Three Approaches to Imitation")
    print("=" * 70)

    print(f"\n  {'Method':>15} {'Policy Acc':>12} {'Time':>8} {'Reward?':>10} "
          f"{'Transfer?':>10}")
    print(f"  {'-'*58}")
    for label, key, transfer in [
        ("BC", "bc", "No"),
        ("MCE-IRL", "mce_irl", "Yes"),
        ("GAIL", "gail", "No"),
    ]:
        r = results[key]
        reward = "Yes" if r["has_reward"] else "No"
        print(f"  {label:>15} {r['policy_acc']:>11.2%} {r['time']:>7.1f}s "
              f"{reward:>10} {transfer:>10}")

    print(f"""
On this small tabular problem, MCE-IRL with correct features
dominates. BC is surprisingly strong because 10,000 observations
give accurate frequency estimates. GAIL's adversarial training
adds overhead without benefit in the tabular setting.

GAIL was designed for continuous-state problems with neural
network policies (Mujoco, Atari). On tabular MDPs where exact
value iteration is feasible, MCE-IRL's feature matching is both
faster and more accurate. GAIL's value appears in high-dimensional
settings where MCE-IRL's feature specification is infeasible.""")

    # Save
    path = Path(__file__).parent / "gail_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
