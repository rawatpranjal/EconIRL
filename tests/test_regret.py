"""Tests for the counterfactual regret metric (Type A/B/C, reusing the package taxonomy).

Deterministic and fast: uses the true parameters / oracle policy and a frozen
policy, no estimator fits. Checks that a perfect transfer has near-zero regret
everywhere, that a frozen behavioral policy cannot adapt (large regret under the
action-set intervention), and the recovered-reward gauge guard.
"""

from __future__ import annotations

import numpy as np

from econirl.environments import random_mdp
from validation.benchmark.regret import (
    estimator_regret,
    recovered_reward,
    _solve,
)


def _env():
    return random_mdp(num_states=8, num_actions=2, num_features=2,
                      branching=3, discount_factor=0.9, seed=0)


def test_recovered_reward_gauge_guard():
    env = _env()
    assert recovered_reward(env, None) is None
    assert recovered_reward(env, [1.0, 2.0, 3.0]) is None  # wrong length (K=2)
    R = recovered_reward(env, env.get_true_parameter_vector())
    assert R is not None and tuple(R.shape) == (env.num_states, env.num_actions)


def test_perfect_transfer_has_near_zero_regret():
    env = _env()
    true_theta = np.asarray(env.get_true_parameter_vector())
    oracle_policy, _ = _solve(env.problem_spec, env.transition_matrices, env.true_reward_matrix)
    r = estimator_regret(env, true_theta, oracle_policy)
    assert r.transferred is True
    # A model that recovered the true reward re-solves every counterfactual exactly.
    for v in (r.baseline, r.type_a, r.type_b, r.type_c):
        assert abs(v) < 1e-2


def test_frozen_policy_cannot_adapt():
    env = _env()
    S, A = env.num_states, env.num_actions
    uniform = np.full((S, A), 1.0 / A)
    r = estimator_regret(env, None, uniform)  # no recovered reward -> frozen
    assert r.transferred is False
    # The action-set intervention (penalize an action) is catastrophic for a
    # policy that cannot re-solve: regret must be far above baseline.
    assert r.type_c > r.baseline + 1.0
