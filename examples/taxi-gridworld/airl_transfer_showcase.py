"""AIRL: Reward transfer across dynamics.

Demonstrates the key contribution of Fu, Luo, and Levine (2018, ICLR):
1. GAIL learns a policy but the discriminator is entangled with the
   training dynamics. The policy does not transfer.
2. AIRL uses discriminator structure f = g(s,a) + beta*h(s') - h(s)
   that disentangles reward g from shaping h. The recovered reward
   transfers to environments with different dynamics.
3. Train on deterministic gridworld, evaluate on stochastic gridworld.
   AIRL's reward can be re-solved under new dynamics; GAIL has no
   reward to transfer.

Usage:
    python examples/taxi-gridworld/airl_transfer_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.gridworld import GridworldEnvironment
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
from econirl.contrib.gail import GAILEstimator, GAILConfig
from econirl.preferences.linear import LinearUtility
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


GRID_SIZE = 5
NOISE = 0.2  # Probability of random action in stochastic env


def policy_accuracy(est_policy, true_policy):
    """Fraction of states where argmax action matches."""
    est_actions = np.argmax(np.array(est_policy), axis=1)
    true_actions = np.argmax(np.array(true_policy), axis=1)
    return float(np.mean(est_actions == true_actions))


def make_stochastic_transitions(transitions, noise=0.2):
    """Add uniform noise to transition matrix to simulate stochastic dynamics."""
    n_actions, n_states, _ = transitions.shape
    uniform = np.ones((n_actions, n_states, n_states)) / n_states
    noisy = (1 - noise) * np.array(transitions) + noise * uniform
    return jnp.array(noisy)


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
    print("AIRL: Reward Transfer Across Dynamics")
    print("Fu, Luo, and Levine (2018, ICLR)")
    print("=" * 70)
    print()
    print("GAIL: policy only, no transferable reward.")
    print("AIRL: disentangled reward that transfers to new dynamics.")
    print(f"Train on deterministic grid, transfer to stochastic (noise={NOISE}).")

    # ---- Deterministic environment (training) ----
    env_det = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=0.99)
    utility = LinearUtility.from_environment(env_det)
    problem = env_det.problem_spec
    transitions_det = env_det.transition_matrices
    true_params = env_det.get_true_parameter_vector()

    # True policy in deterministic env
    operator_det = SoftBellmanOperator(
        problem, jnp.asarray(transitions_det, dtype=jnp.float64)
    )
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result_det = policy_iteration(
        operator_det, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_policy_det = np.array(true_result_det.policy)

    # ---- Stochastic environment (transfer) ----
    transitions_stoch = make_stochastic_transitions(transitions_det, NOISE)
    operator_stoch = SoftBellmanOperator(problem, transitions_stoch)
    true_result_stoch = policy_iteration(
        operator_stoch, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_policy_stoch = np.array(true_result_stoch.policy)

    # Generate training data from deterministic env
    panel = env_det.generate_panel(n_individuals=200, n_periods=50, seed=42)

    print(f"\n  Grid: {GRID_SIZE}x{GRID_SIZE}, States: {problem.num_states}")
    print(f"  Training observations: {panel.num_observations}")
    print(f"  Transfer: deterministic -> stochastic (noise={NOISE})")

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- AIRL (trains on deterministic) ----
    print(f"\n--- AIRL (potential-based shaping) ---")
    t0 = time.time()
    airl = AIRLEstimator(config=AIRLConfig(
        reward_type="tabular",
        reward_lr=0.01,
        max_rounds=100,
        use_shaping=True,
        compute_se=False,
        verbose=False,
    ))
    airl_result = airl.estimate(panel, utility, problem, transitions_det)
    airl_time = time.time() - t0
    airl_acc_det = policy_accuracy(airl_result.policy, true_policy_det)

    # Transfer: re-solve MDP under stochastic dynamics with AIRL reward
    # AIRL with tabular reward returns S*A parameters as a flat array
    n_states = problem.num_states
    n_actions = problem.num_actions
    if "reward_table" in airl_result.metadata:
        airl_reward = jnp.asarray(
            airl_result.metadata["reward_table"], dtype=jnp.float64
        )
    else:
        # Tabular reward: reshape flat (S*A,) to (S, A)
        params = np.array(airl_result.parameters)
        if len(params) == n_states * n_actions:
            airl_reward = jnp.asarray(
                params.reshape(n_states, n_actions), dtype=jnp.float64
            )
        else:
            airl_reward = jnp.asarray(
                utility.compute(jnp.asarray(airl_result.parameters, dtype=jnp.float32)),
                dtype=jnp.float64,
            )
    transfer_result = policy_iteration(
        operator_stoch, airl_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    airl_acc_transfer = policy_accuracy(transfer_result.policy, true_policy_stoch)

    print(f"  Time: {airl_time:.1f}s")
    print(f"  Original env policy accuracy: {airl_acc_det:.2%}")
    print(f"  Transfer env policy accuracy: {airl_acc_transfer:.2%}")
    results["airl"] = {
        "time": airl_time,
        "original_acc": airl_acc_det,
        "transfer_acc": airl_acc_transfer,
        "has_reward": True,
    }

    # ---- GAIL (trains on deterministic) ----
    print(f"\n--- GAIL (no transferable reward) ---")
    t0 = time.time()
    gail = GAILEstimator(config=GAILConfig(
        discriminator_type="tabular",
        discriminator_lr=0.01,
        max_rounds=100,
        compute_se=False,
        verbose=False,
    ))
    gail_result = gail.estimate(panel, utility, problem, transitions_det)
    gail_time = time.time() - t0
    gail_acc_det = policy_accuracy(gail_result.policy, true_policy_det)

    # GAIL cannot transfer -- no reward to re-solve under new dynamics
    print(f"  Time: {gail_time:.1f}s")
    print(f"  Original env policy accuracy: {gail_acc_det:.2%}")
    print(f"  Transfer env policy accuracy: N/A (no reward to transfer)")
    results["gail"] = {
        "time": gail_time,
        "original_acc": gail_acc_det,
        "transfer_acc": float("nan"),
        "has_reward": False,
    }

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Reward Transfer: AIRL vs GAIL")
    print("=" * 70)

    print(f"\n  {'Method':>10} {'Original':>12} {'Transfer':>12} {'Reward?':>10}")
    print(f"  {'-'*48}")
    for label, key in [("AIRL", "airl"), ("GAIL", "gail")]:
        r = results[key]
        transfer = f"{r['transfer_acc']:.2%}" if not np.isnan(r["transfer_acc"]) else "N/A"
        print(f"  {label:>10} {r['original_acc']:>11.2%} {transfer:>12} "
              f"{'Yes' if r['has_reward'] else 'No':>10}")

    print(f"""
AIRL's discriminator structure separates reward g(s,a) from
dynamics-dependent shaping h(s). The reward transfers to new
environments: re-solve the MDP under stochastic dynamics using
AIRL's recovered reward. GAIL produces only a policy, with no
reward to re-optimize. When dynamics change, GAIL must retrain
from scratch.""")

    # Save
    path = Path(__file__).parent / "airl_transfer_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
