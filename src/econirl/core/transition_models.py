"""Compact transition representations for tabular dynamic models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DeterministicTransitions:
    """Deterministic tabular transitions with state-local action slots.

    ``next_state[s, a]`` is the successor of action ``a`` in state ``s``.
    ``valid_action[s, a]`` marks whether that state-action pair belongs to the
    choice set. Invalid successor entries are normalized to state zero and are
    never used by Bellman or occupancy operations.
    """

    next_state: jnp.ndarray | np.ndarray
    valid_action: jnp.ndarray | np.ndarray | None = None

    def __post_init__(self) -> None:
        next_state = np.asarray(self.next_state)
        if next_state.ndim != 2:
            raise ValueError(
                f"next_state must have shape (n_states, n_actions), got {next_state.shape}"
            )
        if not np.issubdtype(next_state.dtype, np.integer):
            raise TypeError("next_state must contain integer state indices")

        if self.valid_action is None:
            valid_action = np.ones(next_state.shape, dtype=bool)
        else:
            valid_action = np.asarray(self.valid_action, dtype=bool)
            if valid_action.shape != next_state.shape:
                raise ValueError(
                    "next_state and valid_action must have the same shape, "
                    f"got {next_state.shape} and {valid_action.shape}"
                )

        n_states = next_state.shape[0]
        valid_successors = next_state[valid_action]
        if valid_successors.size and (
            np.any(valid_successors < 0) or np.any(valid_successors >= n_states)
        ):
            raise ValueError(
                f"valid next states must lie in [0, {n_states}), "
                f"got range [{valid_successors.min()}, {valid_successors.max()}]"
            )

        safe_next_state = np.where(valid_action, next_state, 0)
        object.__setattr__(self, "next_state", jnp.asarray(safe_next_state, dtype=jnp.int32))
        object.__setattr__(self, "valid_action", jnp.asarray(valid_action, dtype=bool))

    @property
    def num_states(self) -> int:
        """Number of states."""
        return int(self.next_state.shape[0])

    @property
    def num_actions(self) -> int:
        """Number of padded action slots."""
        return int(self.next_state.shape[1])


TransitionModel: TypeAlias = jnp.ndarray | DeterministicTransitions


def expected_values(transitions: TransitionModel, value: jnp.ndarray) -> jnp.ndarray:
    """Return ``E[V(s') | s, a]`` with shape ``(S, A)``."""
    if isinstance(transitions, DeterministicTransitions):
        assert transitions.valid_action is not None
        successor_value = value[transitions.next_state]
        return jnp.where(transitions.valid_action, successor_value, 0.0)
    return jnp.einsum("ast,t->as", transitions, value).T


def advance_distribution(
    transitions: TransitionModel,
    state_action_mass: jnp.ndarray,
) -> jnp.ndarray:
    """Push state-action probability mass through the transition model."""
    if isinstance(transitions, DeterministicTransitions):
        assert transitions.valid_action is not None
        mass = jnp.where(transitions.valid_action, state_action_mass, 0.0)
        result = jnp.zeros(transitions.num_states, dtype=mass.dtype)
        return result.at[transitions.next_state.reshape(-1)].add(mass.reshape(-1))
    return jnp.einsum("sa,ast->t", state_action_mass, transitions)
