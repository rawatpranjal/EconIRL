"""Pre-computed sufficient statistics for tabular estimators.

Tabular estimators (NFXP, CCP) only need summary statistics from panel data,
not individual trajectories. Pre-computing these makes estimation O(1) in
dataset size after the initial O(N) computation pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class SufficientStats:
    """Pre-computed statistics for tabular estimators.

    Tabular estimators (NFXP, CCP) only need these summary statistics,
    not individual trajectories. This makes estimation O(1) in dataset size.

    Attributes:
        state_action_counts: Array of shape (S, A) with raw observation counts.
        transitions: Array of shape (A, S, S) with estimated P(s'|s,a).
        empirical_ccps: Array of shape (S, A) with empirical P(a|s).
        initial_distribution: Array of shape (S,) with empirical starting
            state distribution across individuals.
        n_observations: Total number of (s, a, s') observations.
        n_individuals: Number of distinct individuals in the panel.
    """

    state_action_counts: jnp.ndarray  # (S, A)
    transitions: jnp.ndarray  # (A, S, S)
    empirical_ccps: jnp.ndarray  # (S, A)
    initial_distribution: jnp.ndarray  # (S,)
    n_observations: int
    n_individuals: int

    def __post_init__(self) -> None:
        S_sa, A_sa = self.state_action_counts.shape
        A_tr, S_tr1, S_tr2 = self.transitions.shape
        S_ccp, A_ccp = self.empirical_ccps.shape
        (S_init,) = self.initial_distribution.shape

        if not (S_sa == S_tr1 == S_tr2 == S_ccp == S_init):
            raise ValueError(
                f"Inconsistent state dimensions: "
                f"state_action_counts ({S_sa}), transitions ({S_tr1}, {S_tr2}), "
                f"empirical_ccps ({S_ccp}), initial_distribution ({S_init})"
            )
        if not (A_sa == A_tr == A_ccp):
            raise ValueError(
                f"Inconsistent action dimensions: "
                f"state_action_counts ({A_sa}), transitions ({A_tr}), "
                f"empirical_ccps ({A_ccp})"
            )

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.state_action_counts.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return self.state_action_counts.shape[1]
