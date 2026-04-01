"""Base class for adversarial imitation learning methods.

This module provides shared functionality for GAIL, AIRL, and similar
adversarial methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import jax
import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator


class AdversarialEstimatorBase(BaseEstimator):
    """Base class for adversarial imitation learning estimators.

    Provides shared utilities for sampling, policy computation, and
    initial state distribution estimation.
    """

    def _sample_from_panel(
        self,
        panel: Panel,
        batch_size: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample state-action pairs from expert demonstrations."""
        states = panel.get_all_states()
        actions = panel.get_all_actions()

        if batch_size is not None and batch_size > 0 and batch_size < len(states):
            indices = jax.random.permutation(jax.random.key(0), len(states))[:batch_size]
            return states[indices], actions[indices]

        return states, actions

    def _sample_transitions_from_panel(
        self,
        panel: Panel,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample (s, a, s') transitions from expert demonstrations."""
        return (
            panel.get_all_states(),
            panel.get_all_actions(),
            panel.get_all_next_states(),
        )

    def _sample_from_policy(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        n_samples: int,
        initial_dist: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample state-action pairs from current policy."""
        n_states, n_actions = policy.shape
        states = []
        actions = []

        state = jax.random.categorical(jax.random.key(0), jnp.log(initial_dist, 1).item()

        for _ in range(n_samples):
            action = jax.random.categorical(jax.random.key(0), jnp.log(policy[state], 1).item()
            states.append(state)
            actions.append(action)

            next_state_dist = transitions[action, state, :]
            state = jax.random.categorical(jax.random.key(0), jnp.log(next_state_dist, 1).item()

        return jnp.array(states, dtype=jnp.int32), jnp.array(
            actions, dtype=jnp.int32
        )

    def _sample_transitions_from_policy(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        n_samples: int,
        initial_dist: jnp.ndarray,
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample (s, a, s') transitions from current policy via lax.scan.

        Uses compiled trajectory sampling instead of a Python for-loop,
        which is 5-17x faster than sequential multinomial sampling.
        """
        if key is None:
            key = jax.random.key(0)

        def step_fn(carry, step_key):
            state = carry
            k1, k2 = jax.random.split(step_key)
            action = jax.random.categorical(k1, jnp.log(policy[state] + 1e-10))
            next_state = jax.random.categorical(k2, jnp.log(transitions[action, state] + 1e-10))
            return next_state.astype(jnp.int32), (
                state, action.astype(jnp.int32), next_state.astype(jnp.int32)
            )

        init_key, scan_key = jax.random.split(key)
        init_state = jax.random.categorical(
            init_key, jnp.log(initial_dist + 1e-10)
        ).astype(jnp.int32)
        keys = jax.random.split(scan_key, n_samples)
        _, (states, actions, next_states) = jax.lax.scan(
            step_fn, init_state, keys
        )

        return states, actions, next_states

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute initial state distribution from data."""
        counts = jnp.zeros(n_states, dtype=jnp.float32)
        init_states = jnp.array(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )
        counts.scatter_add_(0, init_states, jnp.ones_like(init_states, dtype=jnp.float32))

        if counts.sum() > 0:
            return counts / counts.sum()
        return jnp.ones(n_states) / n_states

    def _compute_policy(
        self,
        reward_matrix: jnp.ndarray,
        operator: SoftBellmanOperator,
        solver: Literal["value", "hybrid"] = "hybrid",
        tol: float = 1e-8,
        max_iter: int = 5000,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute optimal policy given reward matrix."""
        if solver == "hybrid":
            result = hybrid_iteration(operator, reward_matrix, tol=tol, max_iter=max_iter)
        else:
            result = value_iteration(operator, reward_matrix, tol=tol, max_iter=max_iter)
        return result.policy, result.V
