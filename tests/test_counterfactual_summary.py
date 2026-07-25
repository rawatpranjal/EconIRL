"""Tests for CounterfactualResult.summary().

Builds a small solved MDP, forms a Type-3 (reward parameter change)
counterfactual, and checks the two-policy summary renders with correct
welfare and oracle-recovery numbers. Small and fast (no estimator fit).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.inference.results import EstimationSummary
from econirl.preferences.linear import LinearUtility
from econirl.simulation.counterfactual import (
    compute_welfare_effect,
    counterfactual_policy,
)


def _setup():
    S, A, K = 5, 2, 2
    rng = np.random.default_rng(0)
    T = rng.random((A, S, S))
    T /= T.sum(axis=2, keepdims=True)
    phi = np.zeros((S, A, K), dtype=np.float32)
    phi[:, 1, 0] = 1.0  # action-varying feature
    for s in range(S):
        phi[s, :, 1] = s / S  # state feature
    util = LinearUtility(feature_matrix=jnp.asarray(phi), parameter_names=["x0", "x1"])
    prob = DDCProblem(num_states=S, num_actions=A, discount_factor=0.9, scale_parameter=1.0)
    params = jnp.array([1.0, -0.5])
    op = SoftBellmanOperator(prob, jnp.asarray(T))
    base = value_iteration(op, util.compute(params))
    result = EstimationSummary(
        parameters=params,
        parameter_names=["x0", "x1"],
        standard_errors=jnp.array([0.1, 0.1]),
        policy=base.policy,
        value_function=base.V,
    )
    return result, util, prob, jnp.asarray(T)


def test_counterfactual_summary_renders_and_matches_welfare():
    result, util, prob, T = _setup()
    cf = counterfactual_policy(result, {"x0": 1.5, "x1": -0.5}, util, prob, T)

    # transitions should have been threaded onto the result
    assert cf.transitions is not None

    out = cf.summary()
    assert "Counterfactual Summary" in out
    assert "Action rate a=0 (long-run)" in out
    assert "Expected value  E_mu[V]" in out
    assert "Policy shift" in out
    assert "Welfare change:" in out

    # welfare in the summary should equal the reused evaluator
    welfare = compute_welfare_effect(cf, T, use_stationary=True)
    total = float(welfare["total_welfare_change"])
    assert f"{total:+.2f} (stationary)" in out


def test_counterfactual_summary_oracle_self_recovery_is_zero():
    result, util, prob, T = _setup()
    cf = counterfactual_policy(result, {"x0": 1.5, "x1": -0.5}, util, prob, T)

    out = cf.summary(oracle=cf)  # comparing to itself -> perfect recovery
    assert "recovery vs oracle" in out
    assert "Policy TV vs truth:      0.0000" in out
    assert "Value RMSE vs truth:     0.0000" in out


def test_counterfactual_summary_irl_guard_hides_welfare_levels():
    result, util, prob, T = _setup()
    cf = counterfactual_policy(result, {"x0": 1.5, "x1": -0.5}, util, prob, T)

    out = cf.summary(reward_level_identified=False)
    assert "not identified in levels" in out
    assert "Expected value  E_mu[V]" not in out
