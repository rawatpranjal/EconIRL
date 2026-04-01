"""Occupancy measure utilities for MDPs.

Provides standalone functions for computing state and state-action visitation
frequencies (occupancy measures) under a given policy.

The key quantity is the discounted state visitation measure:

    D(s) = sum_{t=0}^{inf} gamma^t * P(S_t = s | pi, rho_0)

Which satisfies the fixed-point equation:

    D = rho_0 + gamma * P_pi^T @ D

where P_pi(s, s') = sum_a pi(a|s) P(s'|s, a) is the policy-weighted transition.

References:
    Gleave, A., & Toyer, S. (2022). A Primer on Maximum Causal Entropy Inverse
        Reinforcement Learning. arXiv:2203.11409.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from econirl.core.types import DDCProblem

_SVF_TOL = 1e-8
_SVF_MAX_ITER = 1000


def compute_state_visitation(
    policy: jnp.ndarray,
    transitions: jnp.ndarray,
    problem: DDCProblem,
    initial_dist: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute discounted state visitation frequencies.

    Solves D = (I - gamma * P_pi^T)^{-1} rho_0 via direct linear solve.
    Falls back to fixed-point iteration if the solve fails (e.g., singular
    matrix, negative result, or non-finite values).

    The returned D is normalized to sum to 1 for use as a probability
    distribution over states.

    Parameters
    ----------
    policy : jnp.ndarray
        Choice probabilities pi(a|s), shape (n_states, n_actions).
    transitions : jnp.ndarray
        Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
    problem : DDCProblem
        Problem specification providing num_states and discount_factor.
    initial_dist : jnp.ndarray, optional
        Initial state distribution rho_0, shape (n_states,).
        If None, uses uniform distribution.

    Returns
    -------
    D : jnp.ndarray
        Normalized state visitation frequencies, shape (n_states,).
        Sums to 1.
    """
    n_states = problem.num_states
    gamma = problem.discount_factor

    if initial_dist is None:
        rho0 = jnp.ones(n_states, dtype=jnp.float64) / n_states
    else:
        rho0 = jnp.array(initial_dist, dtype=jnp.float64)

    # Policy-weighted transition: P_pi[s, s'] = sum_a pi(a|s) P(s'|s, a)
    P_pi = jnp.einsum("sa,ast->st", policy, transitions)

    # Try direct linear solve: (I - gamma * P_pi^T) D = rho_0
    A = jnp.eye(n_states, dtype=jnp.float64) - gamma * P_pi.T
    D = jnp.linalg.solve(A, rho0)

    # Check if solve succeeded
    solve_ok = jnp.isfinite(D).all() & (D >= -1e-6).all()

    if solve_ok:
        D = jnp.maximum(D, 0.0)
        total = D.sum()
        if total > 0:
            return D / total

    # Fallback: fixed-point iteration D = rho_0 + gamma * P_pi^T @ D
    D = rho0

    def body_fn(carry):
        D, error, k = carry
        D_new = rho0 + gamma * (P_pi.T @ D)
        error = jnp.max(jnp.abs(D_new - D))
        return D_new, error, k + 1

    def cond_fn(carry):
        _, error, k = carry
        return jnp.logical_and(error > _SVF_TOL, k < _SVF_MAX_ITER)

    D, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (D, jnp.float64(_SVF_TOL + 1.0), jnp.int32(0))
    )

    total = D.sum()
    return jnp.where(total > 0, D / total, D)


def compute_state_action_visitation(
    policy: jnp.ndarray,
    transitions: jnp.ndarray,
    problem: DDCProblem,
    initial_dist: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute discounted state-action visitation frequencies.

    D_sa[s, a] = D[s] * pi(a|s), where D is the discounted state visitation
    from compute_state_visitation.

    Parameters
    ----------
    policy : jnp.ndarray
        Choice probabilities pi(a|s), shape (n_states, n_actions).
    transitions : jnp.ndarray
        Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
    problem : DDCProblem
        Problem specification.
    initial_dist : jnp.ndarray, optional
        Initial state distribution rho_0. If None, uses uniform.

    Returns
    -------
    D_sa : jnp.ndarray
        State-action visitation frequencies, shape (n_states, n_actions).
    """
    D = compute_state_visitation(policy, transitions, problem, initial_dist)
    return D[:, None] * policy
