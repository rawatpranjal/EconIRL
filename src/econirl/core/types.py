"""Core data types for dynamic discrete choice models.

This module defines the fundamental data structures used throughout econirl:
- DDCProblem: Specification of a discrete choice problem
- Trajectory: A single individual's state-action-state sequence
- Panel: Collection of trajectories (panel data)
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class DDCProblem:
    """Specification of a Dynamic Discrete Choice problem.

    This dataclass contains the structural parameters that define the
    decision environment, following the notation in Rust (1987).

    Attributes:
        num_states: Number of discrete states |S|
        num_actions: Number of discrete actions |A|
        discount_factor: Time discount factor β ∈ [0, 1)
        scale_parameter: Logit scale parameter σ > 0 (extreme value shock scale)

    Example:
        >>> problem = DDCProblem(
        ...     num_states=90,
        ...     num_actions=2,
        ...     discount_factor=0.9999,
        ...     scale_parameter=1.0
        ... )
    """

    num_states: int
    num_actions: int
    discount_factor: float = 0.9999
    scale_parameter: float = 1.0
    num_periods: int | None = None  # None = infinite horizon, int = finite horizon
    state_dim: int | None = None
    state_encoder: Callable[[torch.Tensor], torch.Tensor] | None = field(
        default=None, hash=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.num_states < 1:
            raise ValueError(f"num_states must be positive, got {self.num_states}")
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be positive, got {self.num_actions}")
        if not 0 <= self.discount_factor < 1:
            raise ValueError(
                f"discount_factor must be in [0, 1), got {self.discount_factor}"
            )
        if self.scale_parameter <= 0:
            raise ValueError(
                f"scale_parameter must be positive, got {self.scale_parameter}"
            )


@dataclass
class Trajectory:
    """A single individual's observed decision trajectory.

    Represents the sequence of states, actions, and next states observed
    for one decision-maker over time. This is the fundamental unit of
    observation in dynamic discrete choice estimation.

    Attributes:
        states: Tensor of shape (T,) containing state indices at each period
        actions: Tensor of shape (T,) containing chosen action at each period
        next_states: Tensor of shape (T,) containing state after transition
        individual_id: Optional identifier for the individual
        metadata: Optional dictionary for additional trajectory-level data

    Example:
        >>> traj = Trajectory(
        ...     states=torch.tensor([0, 5, 12, 18]),
        ...     actions=torch.tensor([0, 0, 0, 1]),
        ...     next_states=torch.tensor([5, 12, 18, 0]),
        ...     individual_id="bus_001"
        ... )
    """

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    individual_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions):
            raise ValueError(
                f"states and actions must have same length, "
                f"got {len(self.states)} and {len(self.actions)}"
            )
        if len(self.states) != len(self.next_states):
            raise ValueError(
                f"states and next_states must have same length, "
                f"got {len(self.states)} and {len(self.next_states)}"
            )

    def __len__(self) -> int:
        """Return the number of time periods in this trajectory."""
        return len(self.states)

    @property
    def num_periods(self) -> int:
        """Number of time periods observed."""
        return len(self.states)

    def to(self, device: torch.device | str) -> Trajectory:
        """Move trajectory tensors to specified device."""
        return Trajectory(
            states=self.states.to(device),
            actions=self.actions.to(device),
            next_states=self.next_states.to(device),
            individual_id=self.individual_id,
            metadata=self.metadata,
        )


@dataclass
class Panel:
    """Collection of individual trajectories forming a panel dataset.

    A Panel represents the complete dataset used for estimation, containing
    trajectories from multiple individuals observed over (potentially varying)
    time periods. This is the primary data structure passed to estimators.

    Attributes:
        trajectories: List of Trajectory objects, one per individual
        metadata: Optional dictionary for panel-level metadata

    Example:
        >>> panel = Panel(trajectories=[traj1, traj2, traj3])
        >>> print(f"Panel with {panel.num_individuals} individuals")
        >>> print(f"Total observations: {panel.num_observations}")
    """

    trajectories: list[Trajectory]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("Panel must contain at least one trajectory")

    def __len__(self) -> int:
        """Return the number of individuals in the panel."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        """Get trajectory by index."""
        return self.trajectories[idx]

    def __iter__(self):
        """Iterate over trajectories."""
        return iter(self.trajectories)

    @property
    def num_individuals(self) -> int:
        """Number of individuals in the panel."""
        return len(self.trajectories)

    @property
    def num_observations(self) -> int:
        """Total number of state-action observations across all individuals."""
        return sum(len(traj) for traj in self.trajectories)

    @property
    def num_periods_per_individual(self) -> list[int]:
        """List of number of periods for each individual."""
        return [len(traj) for traj in self.trajectories]

    def get_all_states(self) -> torch.Tensor:
        """Concatenate all states into a single tensor."""
        return torch.cat([traj.states for traj in self.trajectories])

    def get_all_actions(self) -> torch.Tensor:
        """Concatenate all actions into a single tensor."""
        return torch.cat([traj.actions for traj in self.trajectories])

    def get_all_next_states(self) -> torch.Tensor:
        """Concatenate all next_states into a single tensor."""
        return torch.cat([traj.next_states for traj in self.trajectories])

    def to(self, device: torch.device | str) -> Panel:
        """Move all trajectory tensors to specified device."""
        return Panel(
            trajectories=[traj.to(device) for traj in self.trajectories],
            metadata=self.metadata,
        )

    def compute_state_frequencies(self, num_states: int) -> torch.Tensor:
        """Compute empirical state visit frequencies.

        Args:
            num_states: Total number of possible states

        Returns:
            Tensor of shape (num_states,) with visit counts
        """
        all_states = self.get_all_states()
        frequencies = torch.zeros(num_states, dtype=torch.float32)
        for state in all_states:
            frequencies[state.item()] += 1
        return frequencies / frequencies.sum()

    def compute_choice_frequencies(
        self, num_states: int, num_actions: int
    ) -> torch.Tensor:
        """Compute empirical choice frequencies by state.

        This gives the empirical conditional choice probabilities (CCPs)
        that can be used for CCP-based estimation methods.

        Args:
            num_states: Total number of possible states
            num_actions: Total number of possible actions

        Returns:
            Tensor of shape (num_states, num_actions) with empirical CCPs
        """
        all_states = self.get_all_states()
        all_actions = self.get_all_actions()

        counts = torch.zeros((num_states, num_actions), dtype=torch.float32)
        for state, action in zip(all_states, all_actions):
            counts[state.item(), action.item()] += 1

        # Normalize to get probabilities (add small epsilon to avoid division by zero)
        row_sums = counts.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        return counts / row_sums

    @classmethod
    def from_numpy(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        individual_ids: np.ndarray | None = None,
    ) -> Panel:
        """Create Panel from numpy arrays with individual grouping.

        Args:
            states: Array of shape (N,) with state indices
            actions: Array of shape (N,) with action indices
            next_states: Array of shape (N,) with next state indices
            individual_ids: Array of shape (N,) with individual identifiers.
                           If None, all observations treated as one individual.

        Returns:
            Panel object with trajectories grouped by individual
        """
        if individual_ids is None:
            individual_ids = np.zeros(len(states), dtype=np.int64)

        unique_ids = np.unique(individual_ids)
        trajectories = []

        for ind_id in unique_ids:
            mask = individual_ids == ind_id
            traj = Trajectory(
                states=torch.tensor(states[mask], dtype=torch.long),
                actions=torch.tensor(actions[mask], dtype=torch.long),
                next_states=torch.tensor(next_states[mask], dtype=torch.long),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return cls(trajectories=trajectories)


class TrajectoryPanel(Panel):
    """Enhanced panel with efficient tensor operations and DataFrame I/O.

    TrajectoryPanel keeps the list-of-Trajectory interface from Panel but
    adds efficient stacked tensor operations, DataFrame conversion, bootstrap
    resampling, transition iteration, and sufficient statistics computation.

    All existing Panel methods and properties are inherited unchanged.
    """

    # ------------------------------------------------------------------
    # Lazy-cached stacked tensors
    # ------------------------------------------------------------------

    @functools.cached_property
    def all_states(self) -> torch.Tensor:
        """Concatenated states tensor of shape (N,)."""
        return self.get_all_states()

    @functools.cached_property
    def all_actions(self) -> torch.Tensor:
        """Concatenated actions tensor of shape (N,)."""
        return self.get_all_actions()

    @functools.cached_property
    def all_next_states(self) -> torch.Tensor:
        """Concatenated next_states tensor of shape (N,)."""
        return self.get_all_next_states()

    @functools.cached_property
    def offsets(self) -> torch.Tensor:
        """Cumulative individual lengths of shape (I+1,).

        ``offsets[i]`` is the start index of individual ``i`` in the
        concatenated tensors; ``offsets[-1] == num_observations``.
        """
        lengths = [len(traj) for traj in self.trajectories]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.long)

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df,  # pd.DataFrame — imported lazily
        state: str,
        action: str,
        id: str,
        next_state: str | None = None,
    ) -> TrajectoryPanel:
        """Create a TrajectoryPanel from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data with at least ``state``, ``action``, and ``id`` columns.
        state : str
            Column name for state indices.
        action : str
            Column name for action indices.
        id : str
            Column name for individual identifiers.
        next_state : str or None
            Column name for next-state indices.  If ``None``, next states are
            inferred from sequential rows within each individual: for all but
            the last row, ``next_state = states[t+1]``.  For the last row,
            ``next_state = min(state + 1, max_state)`` when ``action == 0``,
            or ``0`` when ``action == 1``.

        Returns
        -------
        TrajectoryPanel
        """
        import pandas as pd  # noqa: F811 — lazy import

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        trajectories: list[Trajectory] = []
        max_state = int(df[state].max())

        for ind_id, group in df.groupby(id, sort=True):
            group = group.sort_index()
            states_arr = torch.tensor(group[state].values, dtype=torch.long)
            actions_arr = torch.tensor(group[action].values, dtype=torch.long)

            if next_state is not None:
                next_states_arr = torch.tensor(
                    group[next_state].values, dtype=torch.long
                )
            else:
                # Infer next_states from sequential rows
                n = len(states_arr)
                next_states_arr = torch.empty(n, dtype=torch.long)
                if n > 1:
                    next_states_arr[:-1] = states_arr[1:]
                # Last row: heuristic
                last_s = states_arr[-1].item()
                last_a = actions_arr[-1].item()
                if last_a == 0:
                    next_states_arr[-1] = min(last_s + 1, max_state)
                else:
                    next_states_arr[-1] = 0

            trajectories.append(
                Trajectory(
                    states=states_arr,
                    actions=actions_arr,
                    next_states=next_states_arr,
                    individual_id=ind_id,
                )
            )

        return cls(trajectories=trajectories)

    @classmethod
    def from_panel(cls, panel: Panel) -> TrajectoryPanel:
        """Wrap an existing Panel as a TrajectoryPanel.

        Parameters
        ----------
        panel : Panel
            Existing panel to wrap.

        Returns
        -------
        TrajectoryPanel
        """
        return cls(trajectories=panel.trajectories, metadata=panel.metadata)

    # ------------------------------------------------------------------
    # Sufficient statistics
    # ------------------------------------------------------------------

    def sufficient_stats(self, n_states: int, n_actions: int):
        """Compute sufficient statistics for tabular estimators.

        Parameters
        ----------
        n_states : int
            Total number of states in the MDP.
        n_actions : int
            Total number of actions in the MDP.

        Returns
        -------
        SufficientStats
            Pre-computed state-action counts, transitions, empirical CCPs,
            and initial state distribution.
        """
        from econirl.core.sufficient_stats import SufficientStats

        states = self.all_states
        actions = self.all_actions
        next_states = self.all_next_states

        # --- state-action counts ---
        state_action_counts = torch.zeros(
            (n_states, n_actions), dtype=torch.float64
        )
        for s, a in zip(states, actions):
            state_action_counts[s.item(), a.item()] += 1

        # --- empirical CCPs ---
        row_sums = state_action_counts.sum(dim=1, keepdim=True)
        # Avoid division by zero: states with no observations get uniform
        row_sums_safe = torch.where(
            row_sums > 0, row_sums, torch.ones_like(row_sums)
        )
        empirical_ccps = state_action_counts / row_sums_safe
        # States with zero observations: uniform over actions
        zero_mask = (row_sums.squeeze(1) == 0)
        if zero_mask.any():
            empirical_ccps[zero_mask] = 1.0 / n_actions

        # --- transition matrix (A, S, S) ---
        transition_counts = torch.zeros(
            (n_actions, n_states, n_states), dtype=torch.float64
        )
        for s, a, sp in zip(states, actions, next_states):
            transition_counts[a.item(), s.item(), sp.item()] += 1

        # Normalize rows; add epsilon smoothing for zero-count rows
        eps = 1e-10
        transition_row_sums = transition_counts.sum(dim=2, keepdim=True)
        # For (a, s) pairs with zero observations, fall back to uniform
        transition_row_sums_safe = torch.where(
            transition_row_sums > 0,
            transition_row_sums,
            torch.ones_like(transition_row_sums),
        )
        transitions = transition_counts / transition_row_sums_safe
        # Zero-count rows get uniform
        zero_transition_mask = (transition_row_sums.squeeze(2) == 0)
        if zero_transition_mask.any():
            transitions[zero_transition_mask] = 1.0 / n_states

        # Add epsilon smoothing to all rows, then re-normalize
        transitions = transitions + eps
        transitions = transitions / transitions.sum(dim=2, keepdim=True)

        # --- initial distribution ---
        initial_dist = torch.zeros(n_states, dtype=torch.float64)
        for traj in self.trajectories:
            initial_dist[traj.states[0].item()] += 1
        initial_total = initial_dist.sum()
        if initial_total > 0:
            initial_dist = initial_dist / initial_total
        else:
            initial_dist[:] = 1.0 / n_states

        # Cast to float32 for downstream use
        return SufficientStats(
            state_action_counts=state_action_counts.float(),
            transitions=transitions.float(),
            empirical_ccps=empirical_ccps.float(),
            initial_distribution=initial_dist.float(),
            n_observations=int(states.shape[0]),
            n_individuals=len(self.trajectories),
        )

    # ------------------------------------------------------------------
    # Bootstrap resampling
    # ------------------------------------------------------------------

    def resample_individuals(
        self, n: int | None = None, seed: int | None = None
    ) -> TrajectoryPanel:
        """Bootstrap resample of individuals (trajectories).

        Parameters
        ----------
        n : int or None
            Number of individuals in the resampled panel.  Defaults to the
            same number as the original panel.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        TrajectoryPanel
            New panel with resampled trajectories (sampled with replacement).
        """
        if n is None:
            n = len(self.trajectories)

        rng = np.random.RandomState(seed)
        indices = rng.choice(len(self.trajectories), size=n, replace=True)
        resampled = [self.trajectories[i] for i in indices]
        return TrajectoryPanel(trajectories=resampled, metadata=self.metadata)

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def iter_transitions(
        self, batch_size: int = 512
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over (state, action, next_state) mini-batches.

        Shuffles all transitions and yields them in batches for SGD-style
        training loops.

        Parameters
        ----------
        batch_size : int
            Number of transitions per batch.

        Yields
        ------
        tuple[Tensor, Tensor, Tensor]
            ``(states, actions, next_states)`` each of shape ``(B,)`` where
            ``B <= batch_size``.
        """
        states = self.all_states
        actions = self.all_actions
        next_states = self.all_next_states
        n = states.shape[0]

        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            yield states[idx], actions[idx], next_states[idx]

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Convert panel to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``id``, ``period``, ``state``, ``action``,
            ``next_state``.
        """
        import pandas as pd  # noqa: F811 — lazy import

        rows = []
        for traj in self.trajectories:
            ind_id = traj.individual_id
            for t in range(len(traj)):
                rows.append(
                    {
                        "id": ind_id,
                        "period": t,
                        "state": traj.states[t].item(),
                        "action": traj.actions[t].item(),
                        "next_state": traj.next_states[t].item(),
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Override to() to return TrajectoryPanel
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> TrajectoryPanel:
        """Move all trajectory tensors to specified device."""
        return TrajectoryPanel(
            trajectories=[traj.to(device) for traj in self.trajectories],
            metadata=self.metadata,
        )


# Backward-compatible alias: Panel now points to TrajectoryPanel so new code
# that creates Panel(...) automatically gets the enhanced interface.
Panel = TrajectoryPanel  # type: ignore[misc]
