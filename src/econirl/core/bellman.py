"""Soft Bellman operator for logit discrete choice models.

This module implements the contraction mapping at the heart of dynamic
discrete choice estimation. The soft (logit) Bellman operator accounts
for the extreme value distribution of preference shocks.

Key equations (following Rust 1987):
- Q(s,a) = u(s,a) + beta * sum_s' P(s'|s,a) V(s')
- V(s) = sigma * log(sum_a exp(Q(s,a)/sigma))  [log-sum-exp / social surplus]

The operator is a contraction with modulus beta, guaranteeing
convergence of value iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.transition_models import (
    DeterministicTransitions,
    TransitionModel,
    expected_values,
)
from econirl.core.types import DDCProblem


class BellmanResult(NamedTuple):
    """Result of applying the Bellman operator.

    Attributes:
        Q: Action-value function, shape (num_states, num_actions)
        V: Value function, shape (num_states,)
        policy: Choice probabilities, shape (num_states, num_actions)
    """

    Q: jnp.ndarray
    V: jnp.ndarray
    policy: jnp.ndarray


@dataclass
class SoftBellmanOperator:
    """Soft Bellman operator for logit discrete choice models.

    Implements the fixed-point operator for solving the dynamic programming
    problem with extreme value (Type I) preference shocks. This is the
    inner loop of NFXP estimation.

    The operator computes:
        Q(s,a) = u(s,a) + beta * sum_s' P(s'|s,a) V(s')
        V(s) = sigma * log(sum_a exp(Q(s,a)/sigma))

    where sigma is the scale parameter of the extreme value distribution.

    Attributes:
        problem: DDCProblem specification
        transitions: Transition probability matrices, shape (num_actions, num_states, num_states)
                    where transitions[a, s, s'] = P(s' | s, a)
    """

    problem: DDCProblem
    transitions: TransitionModel
    terminal_states: jnp.ndarray | np.ndarray | None = None

    def __post_init__(self) -> None:
        if isinstance(self.transitions, DeterministicTransitions):
            deterministic_shape = (self.problem.num_states, self.problem.num_actions)
            if self.transitions.next_state.shape != deterministic_shape:
                raise ValueError(
                    "deterministic transitions must have shape "
                    f"{deterministic_shape}, got {self.transitions.next_state.shape}"
                )
        else:
            dense_shape = (
                self.problem.num_actions,
                self.problem.num_states,
                self.problem.num_states,
            )
            if self.transitions.shape != dense_shape:
                raise ValueError(
                    f"transitions must have shape {dense_shape}, got {self.transitions.shape}"
                )

        if self.terminal_states is not None:
            terminal_states = jnp.asarray(self.terminal_states, dtype=bool)
            if terminal_states.shape != (self.problem.num_states,):
                raise ValueError(
                    "terminal_states must have shape "
                    f"({self.problem.num_states},), got {terminal_states.shape}"
                )
            self.terminal_states = terminal_states

    def apply(self, utility: jnp.ndarray, V: jnp.ndarray) -> BellmanResult:
        """Apply the Bellman operator once.

        Args:
            utility: Flow utility matrix, shape (num_states, num_actions)
            V: Current value function, shape (num_states,)

        Returns:
            BellmanResult with updated Q, V, and policy
        """
        beta = self.problem.discount_factor
        sigma = self.problem.scale_parameter

        # Q(s,a) = u(s,a) + beta * sum_s' P(s'|s,a) V(s')
        EV = expected_values(self.transitions, V)

        # Q shape: (num_states, num_actions)
        Q = utility + beta * EV

        if isinstance(self.transitions, DeterministicTransitions):
            valid_action = self.transitions.valid_action
            assert valid_action is not None
            Q = jnp.where(valid_action, Q, -jnp.inf)

        if self.terminal_states is not None:
            terminal_q = jnp.full_like(Q, -jnp.inf).at[:, 0].set(0.0)
            Q = jnp.where(self.terminal_states[:, None], terminal_q, Q)

        # V(s) = sigma * log(sum_a exp(Q(s,a)/sigma)) using log-sum-exp
        V_new = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)

        # Choice probabilities via softmax
        policy = jax.nn.softmax(Q / sigma, axis=1)

        return BellmanResult(Q=Q, V=V_new, policy=policy)

    def compute_expected_value(self, V: jnp.ndarray) -> jnp.ndarray:
        """Compute expected continuation value E[V(s') | s, a].

        Args:
            V: Value function, shape (num_states,)

        Returns:
            Expected values, shape (num_states, num_actions)
        """
        return expected_values(self.transitions, V)

    def compute_choice_probabilities(self, utility: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """Compute choice probabilities P(a|s).

        Args:
            utility: Flow utility matrix, shape (num_states, num_actions)
            V: Value function, shape (num_states,)

        Returns:
            Choice probabilities, shape (num_states, num_actions)
        """
        result = self.apply(utility, V)
        return result.policy

    def compute_log_choice_probabilities(self, utility: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """Compute log choice probabilities log P(a|s).

        Args:
            utility: Flow utility matrix, shape (num_states, num_actions)
            V: Value function, shape (num_states,)

        Returns:
            Log choice probabilities, shape (num_states, num_actions)
        """
        beta = self.problem.discount_factor
        sigma = self.problem.scale_parameter
        EV = expected_values(self.transitions, V)
        Q = utility + beta * EV
        if isinstance(self.transitions, DeterministicTransitions):
            valid_action = self.transitions.valid_action
            assert valid_action is not None
            Q = jnp.where(valid_action, Q, -jnp.inf)
        if self.terminal_states is not None:
            terminal_q = jnp.full_like(Q, -jnp.inf).at[:, 0].set(0.0)
            Q = jnp.where(self.terminal_states[:, None], terminal_q, Q)
        return jax.nn.log_softmax(Q / sigma, axis=1)

    def compute_social_surplus(self, utility: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """Compute expected max utility (inclusive value / social surplus).

        Args:
            utility: Flow utility matrix, shape (num_states, num_actions)
            V: Value function, shape (num_states,)

        Returns:
            Social surplus, shape (num_states,)
        """
        result = self.apply(utility, V)
        return result.V


def compute_flow_utility(params: jnp.ndarray, feature_matrix: jnp.ndarray) -> jnp.ndarray:
    """Compute flow utility matrix from parameters and features.

    U(s,a) = sum_k theta_k * phi(s,a,k)

    Args:
        params: Parameter vector, shape (K,)
        feature_matrix: Feature tensor, shape (S, A, K)

    Returns:
        Flow utility, shape (S, A)
    """
    return jnp.einsum("sak,k->sa", feature_matrix, params)


# ---------------------------------------------------------------------------
# Standalone Bellman function for optimistix.fixed_point
# ---------------------------------------------------------------------------


def bellman_operator_fn(V: jnp.ndarray, args: tuple) -> jnp.ndarray:
    """Bellman fixed-point map V -> T(V) for use with optimistix.

    This standalone function is compatible with optimistix.fixed_point,
    enabling implicit differentiation through the Bellman fixed point
    via the implicit function theorem.

    Args:
        V: Current value function, shape (num_states,)
        args: Tuple of (utility, transitions, beta, sigma) where
            utility: (S, A) flow utility matrix
            transitions: (A, S, S) transition matrices
            beta: discount factor
            sigma: scale parameter

    Returns:
        Updated value function T(V), shape (num_states,)
    """
    utility, transitions, beta, sigma = args
    EV = jnp.einsum("ast,t->as", transitions, V)
    Q = utility + beta * EV.T
    return sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)
