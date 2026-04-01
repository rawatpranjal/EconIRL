"""Type I and Type II counterfactual evaluators.

Type I (state extrapolation): shift a state variable while keeping
    the MDP structure fixed. All methods that recover the correct
    advantage function should agree on counterfactual CCPs.

Type II (transition change): modify P(s'|s,a). Only methods that
    recover the structural reward (separated from continuation
    values) can re-solve the Bellman equation under new dynamics.
"""

from __future__ import annotations

import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem


def solve_policy(
    reward_matrix: jnp.ndarray,
    transitions: jnp.ndarray,
    problem: DDCProblem,
    absorbing_state: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve for optimal policy given reward and transitions.

    Optionally forces V(absorbing_state) = 0 after each VI step
    by setting absorbing state reward to make only exit viable.

    Returns (policy, V) where policy has shape (S, A) and V has shape (S,).
    """
    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, reward_matrix, tol=1e-10, max_iter=10000)
    policy = result.policy
    V = result.V
    return policy, V


def evaluate_type_i(
    reward_matrix: jnp.ndarray,
    transitions: jnp.ndarray,
    problem: DDCProblem,
    shifted_reward: jnp.ndarray,
) -> jnp.ndarray:
    """Compute counterfactual CCPs under a state-variable shift.

    For methods that recover the reward matrix, we substitute the
    shifted reward and re-solve for the policy. For reduced-form Q,
    the caller handles this differently (adjusting Q directly).

    Parameters
    ----------
    reward_matrix : (S, A)
        Recovered reward from estimation.
    transitions : (A, S, S')
        Original transition matrices (unchanged).
    problem : DDCProblem
        Problem specification.
    shifted_reward : (S, A)
        Reward matrix with shifted state features.

    Returns
    -------
    Counterfactual CCP matrix, shape (S, A).
    """
    policy, _ = solve_policy(shifted_reward, transitions, problem)
    return policy


def evaluate_type_ii(
    reward_matrix: jnp.ndarray,
    new_transitions: jnp.ndarray,
    problem: DDCProblem,
) -> jnp.ndarray:
    """Compute counterfactual CCPs under changed transition dynamics.

    Takes the recovered reward and re-solves the Bellman equation
    under the new transition kernel. This requires the structural
    reward separated from continuation values.

    Parameters
    ----------
    reward_matrix : (S, A)
        Recovered structural reward from estimation.
    new_transitions : (A, S, S')
        Modified transition matrices (e.g., buy skips episodes).
    problem : DDCProblem
        Problem specification.

    Returns
    -------
    Counterfactual CCP matrix, shape (S, A).
    """
    policy, _ = solve_policy(reward_matrix, new_transitions, problem)
    return policy


def evaluate_type_i_from_q(
    q_table: jnp.ndarray,
    feature_shift: jnp.ndarray,
    sigma: float = 1.0,
) -> jnp.ndarray:
    """Type I counterfactual for reduced-form Q estimation.

    Adjusts Q-values by the feature shift and recomputes softmax.
    This works because the Q-function is a smooth function of the
    state, so small perturbations to features produce small
    perturbations to Q (and hence to CCPs).

    Parameters
    ----------
    q_table : (S, A)
        Estimated Q-values from reduced-form logit.
    feature_shift : (S, A)
        Additive change to reward from the state shift.
        For reduced-form Q, we approximate: Q_new ~ Q_old + delta_r.
    sigma : float
        Logit scale parameter.

    Returns
    -------
    Counterfactual CCP matrix, shape (S, A).
    """
    q_shifted = q_table + feature_shift
    return jax.nn.softmax(q_shifted / sigma, axis=1)


def evaluate_type_ii_from_q(
    q_table: jnp.ndarray,
    sigma: float = 1.0,
) -> jnp.ndarray:
    """Type II counterfactual for reduced-form Q: just returns in-sample CCPs.

    Reduced-form Q cannot separate reward from continuation values,
    so under changed transitions it has nothing to re-solve with.
    The best it can do is return the original CCPs, which will
    diverge from the oracle as the transition change grows.

    Parameters
    ----------
    q_table : (S, A)
        Estimated Q-values (fitted to original environment).
    sigma : float
        Logit scale parameter.

    Returns
    -------
    CCP matrix from the original Q (unchanged), shape (S, A).
    """
    return jax.nn.softmax(q_table / sigma, axis=1)


import jax
