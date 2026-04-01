"""Solvers for the dynamic programming fixed point.

This module provides iterative methods to solve for the value function
and optimal policy in dynamic discrete choice models with logit shocks.

Available solvers:
- value_iteration: Simple fixed-point iteration (guaranteed convergence)
- policy_iteration: Often faster convergence via policy evaluation step
- hybrid_iteration: Contraction + Newton-Kantorovich (best for high beta)
- optimistix_solve: Optimistix-based solver with implicit differentiation

The hybrid solver implements Rust (1987, 2000)'s recommended approach:
1. Start with cheap contraction iterations
2. Switch to Newton-Kantorovich when close to solution
3. Achieve quadratic convergence in the final phase

For high discount factors (beta > 0.99), the hybrid solver can be
10-100x faster than pure contraction (value_iteration).

All iteration-based solvers use jax.lax.while_loop for compiled
convergence loops, eliminating Python dispatch overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from econirl.core.bellman import BellmanResult, SoftBellmanOperator, bellman_operator_fn
from econirl.core.types import DDCProblem


@dataclass
class SolverResult:
    """Result from solving the dynamic programming problem.

    Attributes:
        Q: Converged action-value function, shape (num_states, num_actions)
        V: Converged value function, shape (num_states,)
        policy: Optimal choice probabilities, shape (num_states, num_actions)
        converged: Whether the solver converged within max_iter
        num_iterations: Number of iterations performed
        final_error: Final convergence error (sup norm of value change)
    """

    Q: jnp.ndarray
    V: jnp.ndarray
    policy: jnp.ndarray
    converged: bool
    num_iterations: int
    final_error: float


def value_iteration(
    operator: SoftBellmanOperator,
    utility: jnp.ndarray,
    V_init: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult:
    """Solve for the fixed point using value iteration with jax.lax.while_loop.

    The entire convergence loop is compiled to a single XLA kernel,
    eliminating Python dispatch overhead per iteration.

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix, shape (num_states, num_actions)
        V_init: Initial value function guess. If None, starts at zeros.
        tol: Convergence tolerance (sup norm)
        max_iter: Maximum number of iterations

    Returns:
        SolverResult with converged value function and policy
    """
    num_states = operator.problem.num_states
    beta = operator.problem.discount_factor
    sigma = operator.problem.scale_parameter
    transitions = operator.transitions

    if V_init is None:
        V = jnp.zeros(num_states, dtype=jnp.float64)
    else:
        V = jnp.array(V_init)

    def body_fn(carry):
        V, iteration, error = carry
        V_new = bellman_operator_fn(V, (utility, transitions, beta, sigma))
        error = jnp.max(jnp.abs(V_new - V))
        return V_new, iteration + 1, error

    def cond_fn(carry):
        _, iteration, error = carry
        return jnp.logical_and(error > tol, iteration < max_iter)

    init_state = (V, jnp.int32(0), jnp.float64(tol + 1.0))
    V_final, n_iter, final_error = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # Get final Q and policy
    final_result = operator.apply(utility, V_final)

    return SolverResult(
        Q=final_result.Q,
        V=final_result.V,
        policy=final_result.policy,
        converged=bool(final_error <= tol),
        num_iterations=int(n_iter),
        final_error=float(final_error),
    )


def policy_iteration(
    operator: SoftBellmanOperator,
    utility: jnp.ndarray,
    V_init: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 100,
    eval_method: Literal["matrix", "iterative"] = "matrix",
    eval_tol: float = 1e-12,
    eval_max_iter: int = 1000,
) -> SolverResult:
    """Solve for the fixed point using policy iteration.

    Policy iteration alternates between:
    1. Policy evaluation: Given policy, solve for V
    2. Policy improvement: Update policy using new V

    This can converge faster than value iteration, especially for
    high discount factors.

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix, shape (num_states, num_actions)
        V_init: Initial value function guess. If None, starts at zeros.
        tol: Convergence tolerance for policy probabilities
        max_iter: Maximum number of policy iterations
        eval_method: "matrix" for direct solve, "iterative" for fixed-point
        eval_tol: Tolerance for iterative policy evaluation
        eval_max_iter: Max iterations for iterative policy evaluation

    Returns:
        SolverResult with converged value function and policy
    """
    problem = operator.problem
    num_states = problem.num_states
    beta = problem.discount_factor
    sigma = problem.scale_parameter

    if V_init is None:
        V = jnp.zeros(num_states, dtype=jnp.float64)
    else:
        V = jnp.array(V_init)

    # Initial policy
    result = operator.apply(utility, V)
    policy = result.policy

    converged = False
    final_error = float("inf")

    for iteration in range(max_iter):
        # Policy evaluation: solve for V given current policy
        if eval_method == "matrix":
            V = _policy_evaluation_matrix(
                utility, policy, operator.transitions, beta, sigma
            )
        else:
            V = _policy_evaluation_iterative(
                utility, policy, operator.transitions, beta, sigma, eval_tol, eval_max_iter
            )

        # Policy improvement
        new_result = operator.apply(utility, V)
        new_policy = new_result.policy

        # Check convergence of policy
        policy_error = float(jnp.abs(new_policy - policy).max())

        policy = new_policy

        if policy_error < tol:
            converged = True
            final_error = policy_error
            break

        final_error = policy_error

    # Final values
    final_result = operator.apply(utility, V)

    return SolverResult(
        Q=final_result.Q,
        V=final_result.V,
        policy=final_result.policy,
        converged=converged,
        num_iterations=iteration + 1,
        final_error=final_error,
    )


def _policy_evaluation_matrix(
    utility: jnp.ndarray,
    policy: jnp.ndarray,
    transitions: jnp.ndarray,
    beta: float,
    sigma: float,
) -> jnp.ndarray:
    """Evaluate policy by solving linear system.

    Given policy pi, solves:
        (I - beta*P_pi) V = r_pi

    where r_pi = E_pi[u] + sigma*H(pi) is expected flow utility plus entropy.
    """
    num_states = utility.shape[0]

    # Expected flow utility under policy (including entropy bonus)
    log_policy = jnp.log(policy + 1e-10)
    expected_utility = (policy * utility).sum(axis=1)
    entropy_bonus = -sigma * (policy * log_policy).sum(axis=1)
    r_pi = expected_utility + entropy_bonus

    # Policy-weighted transition matrix P_pi[s,s'] = sum_a pi(a|s) P(s'|s,a)
    P_pi = jnp.einsum("sa,ast->st", policy, transitions)

    # Solve (I - beta*P_pi) V = r_pi
    A = jnp.eye(num_states) - beta * P_pi
    V = jnp.linalg.solve(A, r_pi)

    return V


def _policy_evaluation_iterative(
    utility: jnp.ndarray,
    policy: jnp.ndarray,
    transitions: jnp.ndarray,
    beta: float,
    sigma: float,
    tol: float,
    max_iter: int,
) -> jnp.ndarray:
    """Evaluate policy by fixed-point iteration.

    Useful when state space is large and matrix solve is expensive.
    """
    num_states = utility.shape[0]

    log_policy = jnp.log(policy + 1e-10)
    expected_utility = (policy * utility).sum(axis=1)
    entropy_bonus = -sigma * (policy * log_policy).sum(axis=1)
    r_pi = expected_utility + entropy_bonus

    P_pi = jnp.einsum("sa,ast->st", policy, transitions)

    # Iterate: V_{k+1} = r_pi + beta * P_pi @ V_k
    V = jnp.zeros(num_states)

    def body_fn(carry):
        V, error, k = carry
        V_new = r_pi + beta * (P_pi @ V)
        error = jnp.max(jnp.abs(V_new - V))
        return V_new, error, k + 1

    def cond_fn(carry):
        _, error, k = carry
        return jnp.logical_and(error > tol, k < max_iter)

    V, _, _ = jax.lax.while_loop(cond_fn, body_fn, (V, jnp.float64(tol + 1.0), jnp.int32(0)))
    return V


def _newton_kantorovich_step(
    V: jnp.ndarray,
    utility: jnp.ndarray,
    operator: SoftBellmanOperator,
) -> tuple[jnp.ndarray, float]:
    """Perform a single Newton-Kantorovich iteration.

    The NK update is:
        V_{k+1} = V_k + (I - beta*P)^{-1} [T(V_k) - V_k]

    This achieves quadratic convergence near the fixed point.

    Reference:
        Rust (2000) NFXP Manual, Section 3.2
    """
    # Apply Bellman operator to get T(V_k)
    result = operator.apply(utility, V)
    V_bellman = result.V
    residual = V_bellman - V

    # Policy-weighted transition matrix
    P_pi = jnp.einsum("sa,ast->st", result.policy, operator.transitions)

    # Solve (I - beta*P) delta = residual for the Newton correction
    beta = operator.problem.discount_factor
    num_states = len(V)
    A = jnp.eye(num_states) - beta * P_pi
    delta = jnp.linalg.solve(A, residual)

    # Apply Newton correction
    V_new = V + delta

    # Compute post-update residual for convergence check
    result_new = operator.apply(utility, V_new)
    post_residual_norm = float(jnp.abs(result_new.V - V_new).max())

    return V_new, post_residual_norm


def hybrid_iteration(
    operator: SoftBellmanOperator,
    utility: jnp.ndarray,
    V_init: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    switch_tol: float = 1e-3,
    max_nk_iter: int = 20,
) -> SolverResult:
    """Solve for the fixed point using hybrid contraction + Newton-Kantorovich.

    This implements the hybrid algorithm from Rust (1987, 2000):
    1. Run contraction iterations until error < switch_tol
    2. Switch to Newton-Kantorovich iterations for quadratic convergence

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix, shape (num_states, num_actions)
        V_init: Initial value function guess. If None, starts at zeros.
        tol: Final convergence tolerance (sup norm)
        max_iter: Maximum total iterations (contraction + NK)
        switch_tol: Switch from contraction to NK when error < this
        max_nk_iter: Maximum NK iterations after switch

    Returns:
        SolverResult with converged value function and policy

    Reference:
        Rust, J. (2000). "NFXP Manual"
        Iskhakov et al. (2016). "The Endurance of First-Price Auctions"
    """
    num_states = operator.problem.num_states
    beta = operator.problem.discount_factor
    sigma = operator.problem.scale_parameter
    transitions = operator.transitions

    if V_init is None:
        V = jnp.zeros(num_states, dtype=jnp.float64)
    else:
        V = jnp.array(V_init)

    # Phase 1: Contraction iterations via while_loop
    def sa_body(carry):
        V, error, k = carry
        V_new = bellman_operator_fn(V, (utility, transitions, beta, sigma))
        error = jnp.max(jnp.abs(V_new - V))
        return V_new, error, k + 1

    def sa_cond(carry):
        _, error, k = carry
        return jnp.logical_and(error > switch_tol, k < max_iter)

    init_state = (V, jnp.float64(switch_tol + 1.0), jnp.int32(0))
    V, sa_error, sa_iters = jax.lax.while_loop(sa_cond, sa_body, init_state)

    converged = bool(sa_error <= tol)
    final_error = float(sa_error)
    total_iters = int(sa_iters)

    # Phase 2: Newton-Kantorovich iterations
    if not converged and float(sa_error) <= switch_tol:
        for nk_iter in range(max_nk_iter):
            V, error = _newton_kantorovich_step(V, utility, operator)
            total_iters += 1

            if error < tol:
                converged = True
                final_error = error
                break

            final_error = error

    # Get final Q and policy
    final_result = operator.apply(utility, V)

    return SolverResult(
        Q=final_result.Q,
        V=final_result.V,
        policy=final_result.policy,
        converged=converged,
        num_iterations=total_iters,
        final_error=final_error,
    )


def backward_induction(
    operator: SoftBellmanOperator,
    utility_sequence: list[jnp.ndarray] | jnp.ndarray,
    terminal_V: jnp.ndarray | None = None,
) -> SolverResult:
    """Solve finite-horizon problem via backward induction.

    Args:
        operator: SoftBellmanOperator instance
        utility_sequence: List of utility matrices, one per period (T periods),
            or stacked array of shape (T, num_states, num_actions)
        terminal_V: Terminal value function. If None, uses zeros.

    Returns:
        SolverResult with period-0 values (Q, V, policy)
    """
    num_states = operator.problem.num_states

    if terminal_V is None:
        V = jnp.zeros(num_states, dtype=jnp.float64)
    else:
        V = jnp.array(terminal_V)

    if isinstance(utility_sequence, list):
        utility_sequence = jnp.stack(utility_sequence)

    T = utility_sequence.shape[0]

    # Backward pass using scan (reversed)
    def step(V_next, utility_t):
        result = operator.apply(utility_t, V_next)
        return result.V, (result.Q, result.V, result.policy)

    V_final, (all_Q, all_V, all_policy) = jax.lax.scan(
        step, V, utility_sequence[::-1]
    )

    # Reverse to get period-0 first
    all_Q = all_Q[::-1]
    all_V = all_V[::-1]
    all_policy = all_policy[::-1]

    return SolverResult(
        Q=all_Q[0],
        V=all_V[0],
        policy=all_policy[0],
        converged=True,
        num_iterations=T,
        final_error=0.0,
    )


def optimistix_solve(
    problem: DDCProblem,
    transitions: jnp.ndarray,
    utility: jnp.ndarray,
    tol: float = 1e-10,
    max_steps: int = 10000,
) -> jnp.ndarray:
    """Solve Bellman fixed point using optimistix with implicit differentiation.

    This is the recommended solver for use with jax.grad, jax.hessian, etc.
    The implicit function theorem is applied automatically through the
    ImplicitAdjoint, giving machine-precision gradients of V* w.r.t.
    structural parameters at O(n) memory cost.

    Args:
        problem: DDCProblem specification
        transitions: Transition matrices, shape (A, S, S)
        utility: Flow utility matrix, shape (S, A)
        tol: Convergence tolerance
        max_steps: Maximum number of iterations

    Returns:
        Converged value function V*, shape (S,). Differentiable via IFT.
    """
    import optimistix as optx

    args = (utility, transitions, problem.discount_factor, problem.scale_parameter)
    V_init = jnp.zeros(problem.num_states)
    solver = optx.FixedPointIteration(rtol=tol, atol=tol)
    sol = optx.fixed_point(
        bellman_operator_fn, solver, V_init, args=args,
        max_steps=max_steps,
        adjoint=optx.ImplicitAdjoint(),
        throw=False,
    )
    return sol.value


def solve(
    operator: SoftBellmanOperator,
    utility: jnp.ndarray,
    V_init: jnp.ndarray | None = None,
    method: Literal["value", "policy", "hybrid"] = "hybrid",
    tol: float = 1e-10,
    max_iter: int = 1000,
    **kwargs,
) -> SolverResult:
    """Convenience dispatcher for different solver methods.

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix
        V_init: Initial value function guess
        method: "value", "policy", or "hybrid"
        tol: Convergence tolerance
        max_iter: Maximum iterations
        **kwargs: Additional arguments passed to the specific solver

    Returns:
        SolverResult
    """
    if method == "value":
        return value_iteration(operator, utility, V_init, tol, max_iter)
    elif method == "policy":
        return policy_iteration(operator, utility, V_init, tol, max_iter, **kwargs)
    elif method == "hybrid":
        return hybrid_iteration(operator, utility, V_init, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'value', 'policy', or 'hybrid'.")
