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

import torch

from econirl.core.types import DDCProblem

_SVF_TOL = 1e-8
_SVF_MAX_ITER = 1000


def compute_state_visitation(
    policy: torch.Tensor,
    transitions: torch.Tensor,
    problem: DDCProblem,
    initial_dist: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted state visitation frequencies.

    Solves D = (I - gamma * P_pi^T)^{-1} rho_0 via direct linear solve.
    Falls back to fixed-point iteration if the solve fails (e.g., singular
    matrix, negative result, or non-finite values).

    The returned D is normalized to sum to 1 for use as a probability
    distribution over states.

    Parameters
    ----------
    policy : torch.Tensor
        Choice probabilities pi(a|s), shape (n_states, n_actions).
    transitions : torch.Tensor
        Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
    problem : DDCProblem
        Problem specification providing num_states and discount_factor.
    initial_dist : torch.Tensor, optional
        Initial state distribution rho_0, shape (n_states,).
        If None, uses uniform distribution.

    Returns
    -------
    D : torch.Tensor
        Normalized state visitation frequencies, shape (n_states,).
        Sums to 1.
    """
    n_states = problem.num_states
    gamma = problem.discount_factor

    if initial_dist is None:
        rho0 = torch.ones(n_states, dtype=policy.dtype) / n_states
    else:
        rho0 = initial_dist.clone().to(policy.dtype)

    # Policy-weighted transition: P_pi[s, s'] = sum_a pi(a|s) P(s'|s, a)
    # transitions: (n_actions, n_states, n_states) = [a, from_s, to_s]
    P_pi = torch.einsum("sa,ast->st", policy, transitions)

    # Try direct linear solve: (I - gamma * P_pi^T) D = rho_0
    try:
        A = torch.eye(n_states, dtype=policy.dtype) - gamma * P_pi.T
        D = torch.linalg.solve(A, rho0)
        if torch.isfinite(D).all() and (D >= -1e-6).all():
            D = D.clamp(min=0.0)
            total = D.sum()
            if total > 0:
                return D / total
    except Exception:
        pass

    # Fallback: fixed-point iteration D = rho_0 + gamma * P_pi^T @ D
    D = rho0.clone()
    for _ in range(_SVF_MAX_ITER):
        D_new = rho0 + gamma * (P_pi.T @ D)
        if torch.abs(D_new - D).max().item() < _SVF_TOL:
            D = D_new
            break
        D = D_new

    total = D.sum()
    if total > 0:
        return D / total
    return D


def compute_state_action_visitation(
    policy: torch.Tensor,
    transitions: torch.Tensor,
    problem: DDCProblem,
    initial_dist: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted state-action visitation frequencies.

    D_sa[s, a] = D[s] * pi(a|s), where D is the discounted state visitation
    from :func:`compute_state_visitation`.

    Parameters
    ----------
    policy : torch.Tensor
        Choice probabilities pi(a|s), shape (n_states, n_actions).
    transitions : torch.Tensor
        Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
    problem : DDCProblem
        Problem specification.
    initial_dist : torch.Tensor, optional
        Initial state distribution rho_0. If None, uses uniform.

    Returns
    -------
    D_sa : torch.Tensor
        State-action visitation frequencies, shape (n_states, n_actions).
    """
    D = compute_state_visitation(policy, transitions, problem, initial_dist)
    return D.unsqueeze(1) * policy
