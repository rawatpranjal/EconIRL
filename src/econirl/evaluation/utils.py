"""Shared utilities for evaluation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem


def compute_policy(
    theta: jnp.ndarray,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    feature_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Compute optimal policy given parameters and environment.

    Args:
        theta: Parameter vector
        problem: DDC problem specification
        transitions: Transition tensor (n_actions, n_states, n_states)
        feature_matrix: Feature tensor (n_states, n_actions, n_features)

    Returns:
        Policy tensor of shape (n_states, n_actions)
    """
    # Compute utility matrix
    utility = jnp.einsum("sak,k->sa", feature_matrix, theta)

    # Create Bellman operator and solve
    operator = SoftBellmanOperator(problem=problem, transitions=transitions)
    result = value_iteration(operator, utility)

    return result.policy
