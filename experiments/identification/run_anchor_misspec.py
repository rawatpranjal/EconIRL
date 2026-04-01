"""Anchor misspecification robustness test.

What happens when the true exit payoff is epsilon != 0 but the AIRL
anchor still pins exit = 0? Tests how fast the recovered reward and
Type II counterfactual accuracy degrade as the anchor assumption
is violated.

This is an analytical (population-level) experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from identification.config import ExperimentConfig, DGPConfig
from identification.environment import SerializedContentEnvironment, EXIT
from identification.counterfactuals import solve_policy
from identification.metrics import reward_mse, ccp_error

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration


def main():
    epsilons = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]
    skip_k = 3

    print("Anchor misspecification: true exit payoff = epsilon, anchor assumes 0")
    print(f"{'epsilon':>8s}  {'Reward MSE':>12s}  {'Type II Err (k={skip_k})':>22s}")

    results = []
    for eps in epsilons:
        cfg = ExperimentConfig()
        env = SerializedContentEnvironment(cfg.dgp)
        problem = env.problem_spec
        transitions = env.transition_matrices
        absorbing = cfg.dgp.absorbing_state

        # True reward with non-zero exit payoff
        true_reward = np.array(env.reward_matrix)
        true_reward[:, EXIT] = eps
        true_reward[absorbing, :] = 0.0  # absorbing state stays zero
        true_reward = jnp.array(true_reward)

        # True policy under actual reward
        operator = SoftBellmanOperator(problem, transitions)
        vi = value_iteration(operator, true_reward, tol=1e-10, max_iter=10000)
        true_V = vi.V

        # What AIRL+anchors would recover: the reward that generates
        # the same policy but with r(s, exit) = 0 enforced.
        # Under anchor assumption, g*(s,a) = r*(s,a) - eps for non-exit actions.
        # Actually, the anchor pins exit=0, which shifts the reward level.
        # The recovered reward is: r_anchored = true_reward with exit zeroed out
        r_anchored = jnp.array(true_reward).at[:, EXIT].set(0.0)

        # Reward MSE: compare anchored recovery to true reward
        mse = reward_mse(r_anchored, true_reward, absorbing, EXIT)

        # Type II counterfactual
        new_trans = env.build_skip_transitions(skip_k)
        # Oracle: re-solve with true reward under new transitions
        oracle_vi = value_iteration(
            SoftBellmanOperator(problem, new_trans),
            true_reward, tol=1e-10, max_iter=10000,
        )
        oracle_cf = oracle_vi.policy

        # Anchored: re-solve with anchored reward under new transitions
        anchored_vi = value_iteration(
            SoftBellmanOperator(problem, new_trans),
            r_anchored, tol=1e-10, max_iter=10000,
        )
        anchored_cf = anchored_vi.policy

        err = ccp_error(anchored_cf, oracle_cf, absorbing)

        print(f"{eps:8.2f}  {mse:12.6f}  {err:22.6f}")
        results.append({
            "epsilon": float(eps),
            "reward_mse": float(mse),
            "type_ii_error": float(err),
        })

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "anchor_misspec.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'anchor_misspec.json'}")


if __name__ == "__main__":
    main()
