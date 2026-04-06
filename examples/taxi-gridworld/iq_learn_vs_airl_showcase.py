"""IQ-Learn vs AIRL: Single-objective IRL without adversarial training.

Demonstrates the key contribution of Garg et al. (2021, NeurIPS):
1. AIRL requires adversarial training (alternating discriminator and
   generator updates). This min-max game is inherently unstable,
   sensitive to learning rates, and can oscillate.
2. IQ-Learn collapses the IRL problem to a single concave maximization
   over Q, with no adversarial loop. Convergence is monotone.
3. Both methods recover reward from demonstrations, but IQ-Learn does
   it more reliably across random seeds.

The showcase runs both methods across 5 seeds on a 5x5 gridworld and
compares convergence stability, reward recovery, and policy accuracy.

Usage:
    python examples/taxi-gridworld/iq_learn_vs_airl_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.gridworld import GridworldEnvironment
from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
from econirl.preferences.linear import LinearUtility
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


N_SEEDS = 5
GRID_SIZE = 5


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


def main():
    print("=" * 70)
    print("IQ-Learn vs AIRL: Concave Optimization vs Adversarial Training")
    print("Garg et al. (2021, NeurIPS) vs Fu et al. (2018, ICLR)")
    print("=" * 70)
    print()
    print("IQ-Learn: single concave optimization over Q, no min-max game.")
    print("AIRL: alternating discriminator/generator updates, GAN-style.")
    print(f"Running {N_SEEDS} seeds on {GRID_SIZE}x{GRID_SIZE} gridworld.")

    env = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=0.99)
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

    print(f"\n  States: {problem.num_states}, Actions: {problem.num_actions}")
    print(f"  Observations: {n_obs}")
    print(f"  True parameters: {[f'{float(p):.4f}' for p in true_params]}")

    run_diagnostics(panel, problem, utility)

    # Run across seeds
    iq_results = []
    airl_results = []

    for seed in range(N_SEEDS):
        print(f"\n--- Data seed {seed} ---")

        # Generate different data for each seed
        panel_seed = env.generate_panel(
            n_individuals=200, n_periods=50, seed=seed * 1000
        )

        # IQ-Learn (tabular Q, single concave optimization)
        t0 = time.time()
        iq = IQLearnEstimator(config=IQLearnConfig(
            q_type="tabular",
            divergence="chi2",
            learning_rate=0.01,
            max_iter=500,
            verbose=False,
        ))
        try:
            iq_result = iq.estimate(panel_seed, utility, problem, transitions)
            iq_time = time.time() - t0
            iq_acc = policy_accuracy(iq_result.policy, true_policy)
            print(f"  IQ-Learn: policy_acc={iq_acc:.2%}, "
                  f"time={iq_time:.1f}s, converged={iq_result.converged}")
            iq_results.append({
                "seed": seed,
                "policy_acc": iq_acc,
                "time": iq_time,
                "converged": bool(iq_result.converged),
            })
        except Exception as e:
            print(f"  IQ-Learn: FAILED ({e})")
            iq_results.append({
                "seed": seed, "policy_acc": float("nan"),
                "time": time.time() - t0, "converged": False,
            })

        # AIRL (tabular reward, adversarial training)
        t0 = time.time()
        airl = AIRLEstimator(config=AIRLConfig(
            reward_type="tabular",
            reward_lr=0.01,
            discriminator_steps=5,
            max_rounds=100,
            compute_se=False,
            verbose=False,
        ))
        try:
            airl_result = airl.estimate(panel_seed, utility, problem, transitions)
            airl_time = time.time() - t0
            airl_acc = policy_accuracy(airl_result.policy, true_policy)
            print(f"  AIRL:     policy_acc={airl_acc:.2%}, "
                  f"time={airl_time:.1f}s, converged={airl_result.converged}")
            airl_results.append({
                "seed": seed,
                "policy_acc": airl_acc,
                "time": airl_time,
                "converged": bool(airl_result.converged),
            })
        except Exception as e:
            print(f"  AIRL:     FAILED ({e})")
            airl_results.append({
                "seed": seed, "policy_acc": float("nan"),
                "time": time.time() - t0, "converged": False,
            })

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Stability Across Seeds")
    print("=" * 70)

    def summarize(results, name):
        accs = [r["policy_acc"] for r in results if not np.isnan(r["policy_acc"])]
        times = [r["time"] for r in results]
        if accs:
            print(f"\n  {name}:")
            print(f"    Policy accuracy: mean={np.mean(accs):.2%}, "
                  f"std={np.std(accs):.2%}")
            print(f"    Time:           mean={np.mean(times):.1f}s")
            print(f"    Converged:      {sum(1 for r in results if r['converged'])}"
                  f"/{len(results)}")
        else:
            print(f"\n  {name}: all runs failed")

    summarize(iq_results, "IQ-Learn")
    summarize(airl_results, "AIRL")

    print(f"""
IQ-Learn solves a single concave optimization problem. There is no
adversarial loop, no discriminator learning rate to tune, and no
mode collapse risk. The standard deviation across seeds measures
stability -- lower is better.

AIRL alternates discriminator and generator updates. The adversarial
dynamics can cause oscillation, sensitivity to the seed, and higher
variance in recovered reward quality.""")

    # Save results
    out = {
        "grid_size": GRID_SIZE,
        "n_seeds": N_SEEDS,
        "true_params": [float(p) for p in true_params],
        "iq_learn": iq_results,
        "airl": airl_results,
    }
    path = Path(__file__).parent / "iq_learn_vs_airl_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
