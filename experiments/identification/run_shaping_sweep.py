"""Shaping magnitude sweep: Type II error as a function of shaping strength.

Shows that without anchors, counterfactual accuracy depends on which
member of the identified set the estimator happens to find. With
delta = alpha * V* for alpha in [0, 1], the Type II error should grow
monotonically in alpha.

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

from identification.config import ExperimentConfig
from identification.environment import SerializedContentEnvironment
from identification.counterfactuals import solve_policy
from identification.metrics import ccp_error
from identification.estimators import oracle_counterfactual_ccps

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration


def main():
    cfg = ExperimentConfig()
    env = SerializedContentEnvironment(cfg.dgp)
    problem = env.problem_spec
    transitions = env.transition_matrices
    absorbing = cfg.dgp.absorbing_state
    true_reward = env.reward_matrix

    operator = SoftBellmanOperator(problem, transitions)
    vi = value_iteration(operator, true_reward, tol=1e-10, max_iter=10000)
    true_V = vi.V

    alphas = np.arange(0, 1.05, 0.1)
    skip_k = 3
    new_trans = env.build_skip_transitions(skip_k)
    oracle_cf = oracle_counterfactual_ccps(env, new_transitions=new_trans)

    print(f"Shaping magnitude sweep: Type II error (buy skips k={skip_k})")
    print(f"{'alpha':>8s}  {'Type II CCP Error':>18s}")

    results = []
    for alpha in alphas:
        # Construct shaped reward: r_shaped = r* + alpha*V(s) - beta*alpha*E[V(s')]
        h = alpha * true_V
        Eh = jnp.einsum("ast,t->as", transitions, h).T
        r_shaped = true_reward + h[:, None] - cfg.dgp.discount_factor * Eh

        # Re-solve under new transitions with the shaped reward
        cf_policy, _ = solve_policy(r_shaped, new_trans, problem)
        err = ccp_error(cf_policy, oracle_cf, absorbing)

        print(f"{alpha:8.2f}  {err:18.6f}")
        results.append({"alpha": float(alpha), "type_ii_error": float(err)})

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "shaping_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'shaping_sweep.json'}")


if __name__ == "__main__":
    main()
