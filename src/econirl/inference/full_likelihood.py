"""Joint full-likelihood scores for Rust-style replacement models."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.preferences.base import UtilityFunction


def build_rust_transition_tensor(
    n_states: int,
    transition_probabilities: jnp.ndarray | np.ndarray,
) -> jnp.ndarray:
    """Build ``P(s' | s, a)`` for the Rust residual-probability model."""
    probabilities = np.asarray(transition_probabilities, dtype=np.float64)
    if n_states < 1:
        raise ValueError("n_states must be positive")
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("transition_probabilities must contain at least two values")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("transition_probabilities must be finite")
    if not np.all(probabilities > 0.0):
        raise ValueError("transition_probabilities must be strictly positive")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("transition_probabilities must sum to one")

    transitions = np.zeros((2, n_states, n_states), dtype=np.float64)
    for state in range(n_states):
        for increment, probability in enumerate(probabilities):
            transitions[0, state, min(state + increment, n_states - 1)] += probability
            transitions[1, state, min(increment, n_states - 1)] += probability
    return jnp.asarray(transitions, dtype=jnp.float64)


def rust_transition_derivative_tensor(
    n_states: int,
    n_transition_probabilities: int,
) -> jnp.ndarray:
    """Differentiate Rust mileage transitions with respect to free probabilities."""
    if n_transition_probabilities < 2:
        raise ValueError("n_transition_probabilities must be at least two")

    n_free = n_transition_probabilities - 1
    residual_increment = n_transition_probabilities - 1
    derivative = np.zeros((n_free, 2, n_states, n_states), dtype=np.float64)

    for k in range(n_free):
        for state in range(n_states):
            keep_positive = min(state + k, n_states - 1)
            keep_residual = min(state + residual_increment, n_states - 1)
            derivative[k, 0, state, keep_positive] += 1.0
            derivative[k, 0, state, keep_residual] -= 1.0

            replace_positive = min(k, n_states - 1)
            replace_residual = min(residual_increment, n_states - 1)
            derivative[k, 1, state, replace_positive] += 1.0
            derivative[k, 1, state, replace_residual] -= 1.0

    return jnp.asarray(derivative, dtype=jnp.float64)


def _validate_rust_score_inputs(
    *,
    panel: Panel,
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    value_function: jnp.ndarray,
    policy: jnp.ndarray,
    transition_probabilities: jnp.ndarray,
    transition_increments: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Validate the model-specific contract and normalize nuisance inputs."""
    n_states = problem.num_states
    if problem.num_actions != 2:
        raise ValueError(
            "full_likelihood_bhhh currently supports the Rust-style two-action replacement model."
        )
    if not hasattr(utility, "feature_matrix"):
        raise ValueError("full_likelihood_bhhh requires a linear utility")

    transition_array = np.asarray(transitions, dtype=np.float64)
    expected_shape = (2, n_states, n_states)
    if transition_array.shape != expected_shape:
        raise ValueError(
            "transitions must have orientation (n_actions, n_states, n_states) "
            f"and shape {expected_shape}, got {transition_array.shape}"
        )
    if not np.all(np.isfinite(transition_array)):
        raise ValueError("transitions must be finite")

    probabilities = np.asarray(transition_probabilities, dtype=np.float64)
    expected_transitions = np.asarray(
        build_rust_transition_tensor(n_states, probabilities),
        dtype=np.float64,
    )
    if not np.allclose(transition_array, expected_transitions, atol=1e-7, rtol=0.0):
        raise ValueError(
            "transitions do not match the Rust residual-probability replacement "
            "structure implied by transition_probabilities"
        )

    increments_raw = np.asarray(transition_increments)
    if increments_raw.ndim != 1:
        raise ValueError("transition_increments must be one-dimensional")
    if increments_raw.shape[0] != panel.num_observations:
        raise ValueError("transition_increments must have one entry per panel observation")
    if not np.all(np.isfinite(increments_raw)):
        raise ValueError("transition_increments must be finite")
    if not np.all(increments_raw == np.floor(increments_raw)):
        raise ValueError("transition_increments must contain integer category labels")
    increments = increments_raw.astype(np.int32)
    if np.any(increments < 0) or np.any(increments >= probabilities.size):
        raise ValueError(
            "transition_increments must lie between zero and the residual transition category"
        )

    features = np.asarray(getattr(utility, "feature_matrix"), dtype=np.float64)
    if features.ndim != 3 or features.shape[:2] != (n_states, 2):
        raise ValueError(
            "linear utility features must have shape (n_states, n_actions, n_parameters)"
        )
    values = np.asarray(value_function, dtype=np.float64)
    if values.shape != (n_states,) or not np.all(np.isfinite(values)):
        raise ValueError("value_function must be a finite vector with one value per state")
    policy_array = np.asarray(policy, dtype=np.float64)
    if policy_array.shape != (n_states, 2) or not np.all(np.isfinite(policy_array)):
        raise ValueError("policy must be a finite (n_states, two_actions) matrix")
    if np.any(policy_array <= 0.0) or not np.allclose(
        policy_array.sum(axis=1),
        1.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("policy rows must be strictly positive and sum to one")

    states = np.asarray(panel.get_all_states())
    actions = np.asarray(panel.get_all_actions())
    if np.any(states < 0) or np.any(states >= n_states):
        raise ValueError("panel states fall outside the model state space")
    if np.any(actions < 0) or np.any(actions >= 2):
        raise ValueError("panel actions must be zero or one")

    return (
        jnp.asarray(probabilities, dtype=jnp.float64),
        jnp.asarray(increments, dtype=jnp.int32),
    )


def compute_rust_full_likelihood_bhhh_score(
    *,
    panel: Panel,
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    value_function: jnp.ndarray,
    policy: jnp.ndarray,
    transition_probabilities: jnp.ndarray,
    transition_increments: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Compute joint BHHH scores for a Rust-style replacement likelihood.

    Score columns contain structural utility parameters first, followed by
    the free transition probabilities. The final transition probability is
    the residual probability.
    """
    probs, increments = _validate_rust_score_inputs(
        panel=panel,
        utility=utility,
        problem=problem,
        transitions=transitions,
        value_function=value_function,
        policy=policy,
        transition_probabilities=transition_probabilities,
        transition_increments=transition_increments,
    )
    beta = problem.discount_factor
    sigma = problem.scale_parameter
    transitions = jnp.asarray(transitions, dtype=jnp.float64)
    features = jnp.asarray(getattr(utility, "feature_matrix"), dtype=jnp.float64)
    value_function = jnp.asarray(value_function, dtype=jnp.float64)
    policy = jnp.asarray(policy, dtype=jnp.float64)
    n_states = problem.num_states

    policy_transition = jnp.einsum("sa,ast->st", policy, transitions)
    fixed_point_jacobian = jnp.eye(n_states, dtype=jnp.float64) - beta * policy_transition

    expected_features = jnp.einsum("sa,sak->sk", policy, features)
    value_derivative = jnp.linalg.solve(fixed_point_jacobian, expected_features)
    expected_value_derivative = jnp.einsum("ast,tk->ask", transitions, value_derivative)
    choice_value_derivative = features + beta * jnp.transpose(
        expected_value_derivative,
        (1, 0, 2),
    )
    policy_mean_derivative = jnp.einsum("sa,sak->sk", policy, choice_value_derivative)

    states = panel.get_all_states()
    actions = panel.get_all_actions()
    structural_scores = (
        choice_value_derivative[states, actions] - policy_mean_derivative[states]
    ) / sigma

    transition_derivative = rust_transition_derivative_tensor(n_states, int(probs.shape[0]))
    transition_value_effect = jnp.einsum("kast,t->kas", transition_derivative, value_function)
    transition_rhs = beta * jnp.einsum("sa,kas->sk", policy, transition_value_effect)
    transition_value_derivative = jnp.linalg.solve(fixed_point_jacobian, transition_rhs)
    expected_transition_derivative = jnp.einsum(
        "ast,tk->ask",
        transitions,
        transition_value_derivative,
    )
    choice_transition_derivative = beta * (
        jnp.transpose(transition_value_effect, (2, 1, 0))
        + jnp.transpose(expected_transition_derivative, (1, 0, 2))
    )
    policy_mean_transition_derivative = jnp.einsum(
        "sa,sak->sk",
        policy,
        choice_transition_derivative,
    )
    transition_choice_scores = (
        choice_transition_derivative[states, actions] - policy_mean_transition_derivative[states]
    ) / sigma

    free_probs = probs[:-1]
    residual_prob = probs[-1]
    n_free = int(free_probs.shape[0])
    transition_density_scores = jnp.zeros(
        (panel.num_observations, n_free),
        dtype=jnp.float64,
    )
    for k in range(n_free):
        transition_density_scores = transition_density_scores.at[increments == k, k].set(
            1.0 / free_probs[k]
        )
        transition_density_scores = transition_density_scores.at[
            increments == n_free,
            k,
        ].set(-1.0 / residual_prob)

    transition_scores = transition_choice_scores + transition_density_scores
    joint_scores = jnp.concatenate([structural_scores, transition_scores], axis=1)
    transition_counts = {int(k): int(jnp.sum(increments == k)) for k in range(int(probs.shape[0]))}

    return joint_scores, {
        "joint_parameter_names": list(utility.parameter_names)
        + [f"transition_p{k}" for k in range(n_free)],
        "transition_probabilities": [float(p) for p in np.asarray(probs)],
        "transition_counts": transition_counts,
        "transition_score_columns": [f"transition_p{k}" for k in range(n_free)],
        "transition_model": "rust_residual_probability",
        "transition_orientation": "(n_actions, n_states, n_states)",
    }
