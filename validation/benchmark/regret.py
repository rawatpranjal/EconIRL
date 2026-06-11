"""Counterfactual regret for the benchmark, reusing the package's taxonomy.

The package already defines a three-tier counterfactual taxonomy and a regret
metric (``validation/known_truth.py``):

- Type A: payoff intervention (reward feature shift, transitions fixed).
- Type B: transition intervention (dynamics change, reward fixed).
- Type C: action-set intervention (penalize one action).
- regret = initial_distribution . (oracle_value - estimated_value), computed by
  ``counterfactual_metrics``. Lower is better; zero is perfect.

This module does NOT reinvent that. It reuses ``counterfactual_metrics`` verbatim
and applies the A/B/C intervention semantics to an abstract ``ArrayMDP`` (the
heavy ``evaluate_estimator_against_truth`` only accepts a structured
``KnownTruthDGP``).

The structural-vs-behavioral asymmetry is the honest point and falls out of the
data: an estimator that recovered a reward (finite theta in the env gauge) gets
that intervention applied to its recovered reward and re-solves it under the new
world (structural transfer); an estimator with no usable reward keeps its fixed
policy and cannot adapt.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.environments.random_mdp import _build_sparse_transitions
from validation.known_truth import counterfactual_metrics

# Intervention magnitudes mirror the package defaults (known_truth CounterfactualConfig).
TYPE_A_SHIFT = -0.25
TYPE_C_PENALTY = -10.0
TYPE_C_ACTION = 1
TYPE_B_CF_SEED = 99


@dataclass(frozen=True)
class RegretRow:
    """Welfare regret (lower better) at baseline and under each counterfactual."""

    baseline: float
    type_a: float
    type_b: float
    type_c: float
    transferred: bool  # True if the estimator re-solved a recovered reward


def recovered_reward(env, params) -> jnp.ndarray | None:
    """R_hat(s,a) = features . params when params are in the env gauge, else None."""
    if params is None:
        return None
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    features = np.asarray(env.feature_matrix, dtype=np.float64)
    if params.shape[0] != features.shape[2]:
        return None
    return jnp.asarray(np.einsum("sak,k->sa", features, params), dtype=jnp.float32)


def _solve(problem, transitions, reward):
    op = SoftBellmanOperator(problem, transitions)
    res = value_iteration(op, reward)
    return res.policy, res.V


def _interventions(env, true_reward: jnp.ndarray):
    """Return {name: (reward_delta (S,A), transitions_cf (A,S,S))} for each tier.

    reward_delta is added to BOTH the true reward (to form the true counterfactual
    world) and the estimator's recovered reward (the known intervention it can act
    on). transitions_cf is the counterfactual dynamics.
    """
    S, A = true_reward.shape
    T = np.asarray(env.transition_matrices, dtype=np.float64)
    zero = jnp.zeros((S, A), dtype=jnp.float32)

    # Type A: payoff shift along the normalized-state (progress) feature, applied
    # to non-outside actions (action 0 is the zeroed outside option).
    progress = np.arange(S, dtype=np.float64) / max(S - 1, 1)
    shift = np.zeros((S, A), dtype=np.float64)
    for a in range(1, A):
        shift[:, a] = TYPE_A_SHIFT * progress
    delta_a = jnp.asarray(shift, dtype=jnp.float32)

    # Type B: fresh dynamics (same shape, inferred branching), reward fixed.
    branching = int(np.max((T > 0).sum(axis=2)))
    rng = np.random.default_rng(TYPE_B_CF_SEED)
    T_b = _build_sparse_transitions(S, A, max(branching, 1), 0.0, rng)

    # Type C: penalize one action everywhere (action-set intervention).
    penalty = np.zeros((S, A), dtype=np.float64)
    action_c = min(TYPE_C_ACTION, A - 1)
    penalty[:, action_c] = TYPE_C_PENALTY
    delta_c = jnp.asarray(penalty, dtype=jnp.float32)

    return {
        "baseline": (zero, env.transition_matrices),
        "type_a": (delta_a, env.transition_matrices),
        "type_b": (zero, T_b),
        "type_c": (delta_c, env.transition_matrices),
    }


def estimator_regret(env, params, baseline_policy) -> RegretRow:
    """Compute baseline + Type A/B/C regret for one estimator.

    Args:
        env: the ArrayMDP the estimator was fit on.
        params: the estimator's recovered parameters (or None).
        baseline_policy: the estimator's recovered policy on the baseline world,
            used when it has no recovered reward (behavioral / cannot transfer).

    Returns:
        RegretRow. ``transferred`` is True when a recovered reward was re-solved.
    """
    problem = env.problem_spec
    discount = float(problem.discount_factor)
    scale = float(problem.scale_parameter)
    init = jnp.asarray(env._get_initial_state_distribution(), dtype=jnp.float32)
    true_reward = jnp.asarray(env.true_reward_matrix, dtype=jnp.float32)
    R_hat = recovered_reward(env, params)
    base_pol = None if baseline_policy is None else jnp.asarray(baseline_policy, dtype=jnp.float32)

    out = {}
    for name, (delta, T_cf) in _interventions(env, true_reward).items():
        true_cf_reward = true_reward + delta
        oracle_policy, oracle_value = _solve(problem, T_cf, true_cf_reward)
        if R_hat is not None:
            est_policy, _ = _solve(problem, T_cf, R_hat + delta)  # structural transfer
        elif base_pol is not None:
            est_policy = base_pol  # frozen: cannot adapt to the new world
        else:
            out[name] = float("nan")
            continue
        m = counterfactual_metrics(
            oracle_policy=oracle_policy, oracle_value=oracle_value,
            estimated_policy=est_policy, reward=true_cf_reward, transitions=T_cf,
            discount_factor=discount, initial_distribution=init, scale_parameter=scale,
        )
        out[name] = float(m.regret)

    return RegretRow(
        baseline=out["baseline"], type_a=out["type_a"],
        type_b=out["type_b"], type_c=out["type_c"],
        transferred=R_hat is not None,
    )
