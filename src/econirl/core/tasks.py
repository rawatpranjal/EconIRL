"""Task-conditioned views over a shared deterministic MDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import jax.numpy as jnp
import numpy as np

from econirl.core.transition_models import DeterministicTransitions
from econirl.core.types import Panel, Trajectory


@dataclass(frozen=True)
class MCEIRLTask:
    """One finite-horizon task over a shared deterministic transition system.

    States and action slots retain their global meanings. ``active_states`` and
    ``action_mask`` define a compact task-specific subgraph. Reward parameters
    remain shared across every task in a fit.
    """

    task_id: Hashable
    terminal_states: np.ndarray
    horizon: int
    initial_state: int | None = None
    initial_state_dist: np.ndarray | None = None
    active_states: np.ndarray | None = None
    action_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if (self.initial_state is None) == (self.initial_state_dist is None):
            raise ValueError("supply exactly one of initial_state or initial_state_dist")


@dataclass(frozen=True)
class CompiledMCEIRLTasks:
    """Disjoint compact MDP produced from a task collection."""

    transitions: DeterministicTransitions
    terminal_states: jnp.ndarray
    feature_matrix: jnp.ndarray
    panel: Panel
    horizon: int
    task_slices: dict[Hashable, slice]
    global_to_local: dict[Hashable, dict[int, int]]
    local_to_global: dict[Hashable, np.ndarray]
    initial_state_dist: jnp.ndarray
    task_initial_state_dist: dict[Hashable, np.ndarray]


def _terminal_indices(values: np.ndarray, n_states: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool:
        if arr.shape != (n_states,):
            raise ValueError(
                f"boolean terminal_states must have shape ({n_states},), got {arr.shape}"
            )
        return np.flatnonzero(arr)
    result = np.asarray(arr, dtype=int).reshape(-1)
    if result.size == 0:
        raise ValueError("each task must define at least one terminal state")
    if np.any(result < 0) or np.any(result >= n_states):
        raise ValueError("terminal state index is outside the shared state space")
    return np.unique(result)


def compile_mce_irl_tasks(
    tasks: list[MCEIRLTask],
    transitions: DeterministicTransitions,
    feature_matrix: np.ndarray | jnp.ndarray,
    panel: Panel,
    *,
    validate_demonstrations: bool = True,
) -> CompiledMCEIRLTasks:
    """Compile task-specific subgraphs into one compact disjoint MDP."""
    if not tasks:
        raise ValueError("tasks must contain at least one MCEIRLTask")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_id values must be unique")
    horizons = {task.horizon for task in tasks}
    if len(horizons) != 1:
        raise ValueError("all tasks in one fit must use the same horizon")

    n_states = transitions.num_states
    n_actions = transitions.num_actions
    features = np.asarray(feature_matrix)
    if features.ndim not in (2, 3) or features.shape[0] != n_states:
        raise ValueError(
            "feature_matrix must have shape (S,K) or (S,A,K) and align "
            "with the shared transition states"
        )
    if features.ndim == 3 and features.shape[1] != n_actions:
        raise ValueError("state-action features must align with action slots")

    next_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    feature_blocks: list[np.ndarray] = []
    terminal_blocks: list[np.ndarray] = []
    initial_blocks: list[np.ndarray] = []
    task_slices: dict[Hashable, slice] = {}
    global_to_local: dict[Hashable, dict[int, int]] = {}
    local_to_global: dict[Hashable, np.ndarray] = {}
    task_initial_state_dist: dict[Hashable, np.ndarray] = {}
    offset = 0

    for task in tasks:
        terminals = _terminal_indices(task.terminal_states, n_states)
        if task.active_states is None:
            active = np.arange(n_states, dtype=int)
        else:
            active = np.unique(np.asarray(task.active_states, dtype=int))
            if active.size == 0:
                raise ValueError(f"task {task.task_id!r} has no active states")
            if np.any(active < 0) or np.any(active >= n_states):
                raise ValueError(f"task {task.task_id!r} has an active state outside the model")
        if not np.all(np.isin(terminals, active)):
            raise ValueError(f"task {task.task_id!r} terminal states must be active")

        mapping = {int(state): i for i, state in enumerate(active)}
        global_to_local[task.task_id] = mapping
        local_to_global[task.task_id] = active
        task_slices[task.task_id] = slice(offset, offset + len(active))

        task_mask = np.asarray(transitions.valid_action, dtype=bool)[active].copy()
        if task.action_mask is not None:
            supplied_mask = np.asarray(task.action_mask, dtype=bool)
            if supplied_mask.shape != (n_states, n_actions):
                raise ValueError(
                    f"task action_mask must have shape {(n_states, n_actions)}, "
                    f"got {supplied_mask.shape}"
                )
            task_mask &= supplied_mask[active]

        local_next = np.zeros((len(active), n_actions), dtype=np.int32)
        global_next = np.asarray(transitions.next_state)[active]
        for local_state, global_state in enumerate(active):
            for action in range(n_actions):
                successor = int(global_next[local_state, action])
                if not task_mask[local_state, action] or successor not in mapping:
                    task_mask[local_state, action] = False
                    continue
                local_next[local_state, action] = offset + mapping[successor]

        local_terminal = np.isin(active, terminals)
        terminal_blocks.append(local_terminal)
        for local_state_value in np.flatnonzero(local_terminal):
            local_state = int(local_state_value)
            task_mask[local_state] = False
            task_mask[local_state, 0] = True
            local_next[local_state, 0] = offset + local_state

        nonterminal_without_action = (~local_terminal) & (~task_mask.any(axis=1))
        if np.any(nonterminal_without_action):
            bad = active[np.flatnonzero(nonterminal_without_action)[:5]]
            raise ValueError(
                f"task {task.task_id!r} has nonterminal states without a legal "
                f"action: {bad.tolist()}"
            )

        if task.initial_state is not None:
            if task.initial_state not in mapping:
                raise ValueError(f"task {task.task_id!r} initial_state must be active")
            local_initial = np.zeros(len(active), dtype=float)
            local_initial[mapping[int(task.initial_state)]] = 1.0
        else:
            supplied_initial = np.asarray(task.initial_state_dist, dtype=float)
            if supplied_initial.shape != (n_states,):
                raise ValueError(
                    f"initial_state_dist must have shape ({n_states},), "
                    f"got {supplied_initial.shape}"
                )
            if np.any(supplied_initial < 0) or not np.isclose(supplied_initial.sum(), 1.0):
                raise ValueError("initial_state_dist must be nonnegative and sum to one")
            if supplied_initial[np.setdiff1d(np.arange(n_states), active)].sum() > 0:
                raise ValueError("initial_state_dist places mass outside active_states")
            local_initial = supplied_initial[active]

        next_blocks.append(local_next)
        valid_blocks.append(task_mask)
        local_features = features[active].copy()
        local_features[local_terminal] = 0.0
        feature_blocks.append(local_features)
        initial_blocks.append(local_initial)
        task_initial_state_dist[task.task_id] = local_initial
        offset += len(active)

    trajectories: list[Trajectory] = []
    for trajectory in panel.trajectories:
        task_id = trajectory.metadata.get("task_id")
        if task_id not in global_to_local:
            raise ValueError("every trajectory must set metadata['task_id'] to a supplied task")
        mapping = global_to_local[task_id]
        task_slice = task_slices[task_id]
        try:
            states = np.array(
                [task_slice.start + mapping[int(s)] for s in trajectory.states],
                dtype=np.int32,
            )
            next_states = np.array(
                [task_slice.start + mapping[int(s)] for s in trajectory.next_states],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise ValueError(f"trajectory for task {task_id!r} leaves its active subgraph") from exc
        actions = np.asarray(trajectory.actions, dtype=np.int32)
        local_states = states - task_slice.start
        task_index = task_ids.index(task_id)
        task_mask = valid_blocks[task_index]
        if validate_demonstrations and np.any(~task_mask[local_states, actions]):
            raise ValueError(f"trajectory for task {task_id!r} uses an invalid action")
        expected_next = next_blocks[task_index][local_states, actions]
        if validate_demonstrations and not np.array_equal(expected_next, next_states):
            raise ValueError(
                f"trajectory for task {task_id!r} disagrees with deterministic transitions"
            )
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
                individual_id=trajectory.individual_id,
                metadata=dict(trajectory.metadata),
            )
        )

    task_counts = {
        task_id: sum(
            trajectory.metadata.get("task_id") == task_id for trajectory in panel.trajectories
        )
        for task_id in task_ids
    }
    empty_tasks = [task_id for task_id, count in task_counts.items() if count == 0]
    if empty_tasks:
        raise ValueError(f"every supplied task needs demonstrations; missing {empty_tasks!r}")
    total_trajectories = sum(task_counts.values())
    weighted_initial_blocks = [
        block * (task_counts[task.task_id] / total_trajectories)
        for task, block in zip(tasks, initial_blocks)
    ]
    initial_state_dist = np.concatenate(weighted_initial_blocks)
    return CompiledMCEIRLTasks(
        transitions=DeterministicTransitions(
            next_state=jnp.asarray(np.concatenate(next_blocks)),
            valid_action=jnp.asarray(np.concatenate(valid_blocks)),
        ),
        terminal_states=jnp.asarray(np.concatenate(terminal_blocks)),
        feature_matrix=jnp.asarray(np.concatenate(feature_blocks)),
        panel=Panel(trajectories=trajectories, metadata=dict(panel.metadata)),
        horizon=horizons.pop(),
        task_slices=task_slices,
        global_to_local=global_to_local,
        local_to_global=local_to_global,
        initial_state_dist=jnp.asarray(initial_state_dist),
        task_initial_state_dist=task_initial_state_dist,
    )
