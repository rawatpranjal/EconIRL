#!/usr/bin/env python3
"""Synthetic road-choice validation following Ziebart et al. (2008).

The original Pittsburgh GPS traces and fitted road graph are unavailable. This
study therefore does not claim a paper-data replication. It does reproduce the
paper's observable data pipeline and model contract with generated data:

* a spatial deterministic road graph with more than 300,000 segment states and
  more than 900,000 intersection transitions;
* 22 path-count features from road type, speed, lanes, and turn categories;
* 25 drivers and 13,220 raw trips;
* explicit rejection of short, cyclic, and noisy trips;
* the paper's 20/80 split of retained trips;
* destination-specific MDP tasks with one shared reward vector; and
* the three Table 1 evaluation definitions.

Road-specific work ends at the adapter. The fitted object is the public,
general-purpose ``MCEIRL`` estimator.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Hashable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import qr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "mce_irl_ziebart_synthetic.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.tasks import MCEIRLTask, compile_mce_irl_tasks  # noqa: E402
from econirl.core.types import DDCProblem, Panel, Trajectory  # noqa: E402
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator  # noqa: E402
from econirl.estimators import MCEIRL  # noqa: E402
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from econirl.transitions import DeterministicTransitions  # noqa: E402

PAPER_TARGETS = {
    "distance_match_percent": 78.79,
    "routes_at_least_90_percent": 52.98,
    "average_log_probability": -6.85,
}

RAW_FEATURE_NAMES = [
    *(f"road_type_{index}" for index in range(6)),
    *(f"speed_{index}" for index in range(6)),
    *(f"lanes_{index}" for index in range(5)),
    *(f"turn_{index}" for index in range(5)),
]


@dataclass(frozen=True)
class RoadStudyConfig:
    """Dimensions and controls for one generated road-choice study."""

    width: int
    height: int
    raw_trips: int
    discarded_trips: int
    train_trips: int
    test_trips: int
    num_drivers: int
    num_tasks: int
    horizon: int
    seed: int
    reward_scale: float
    driver_types: int
    exploratory_scale: float
    l2_regularization: float = 1e-4

    @property
    def num_states(self) -> int:
        return self.width * self.height * 4

    @property
    def num_actions(self) -> int:
        return 3

    @property
    def retained_trips(self) -> int:
        return self.train_trips + self.test_trips


PAPER_SHAPE = RoadStudyConfig(
    width=275,
    height=275,
    raw_trips=13_220,
    discarded_trips=3_966,
    train_trips=1_851,
    test_trips=7_403,
    num_drivers=25,
    num_tasks=64,
    horizon=32,
    seed=2008,
    reward_scale=20.0,
    driver_types=2,
    exploratory_scale=0.15,
)

SMOKE_SHAPE = RoadStudyConfig(
    width=35,
    height=35,
    raw_trips=320,
    discarded_trips=96,
    train_trips=45,
    test_trips=179,
    num_drivers=5,
    num_tasks=8,
    horizon=16,
    seed=2008,
    reward_scale=5.0,
    driver_types=2,
    exploratory_scale=0.30,
)


@dataclass(frozen=True)
class RoadNetwork:
    """Global generated road graph and its adapter views."""

    transitions: DeterministicTransitions
    raw_features: np.ndarray
    segment_lengths: np.ndarray
    road_types: np.ndarray


@dataclass(frozen=True)
class RawTrip:
    """One generated pre-filter trip record."""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    driver_id: int
    task_id: Hashable
    noisy: bool = False


def _state_index(x: np.ndarray, y: np.ndarray, heading: np.ndarray, width: int) -> np.ndarray:
    return ((y * width + x) * 4 + heading).astype(np.int32)


def build_road_network(config: RoadStudyConfig) -> RoadNetwork:
    """Build a spatial segment graph with correlated road attributes.

    A state is a directed road segment ending at an intersection. The three
    action slots are left, straight, and right turns. The successor is the next
    directed segment. The large graph is toroidal only at its far outer edge;
    estimation tasks are sampled away from that edge.
    """
    states = np.arange(config.num_states, dtype=np.int64)
    intersections = states // 4
    heading = states % 4
    x = intersections % config.width
    y = intersections // config.width

    turn_delta = np.array([-1, 0, 1], dtype=np.int64)
    next_heading = (heading[:, None] + turn_delta[None, :]) % 4
    dx = np.array([0, 1, 0, -1], dtype=np.int64)
    dy = np.array([-1, 0, 1, 0], dtype=np.int64)
    next_x = (x[:, None] + dx[next_heading]) % config.width
    next_y = (y[:, None] + dy[next_heading]) % config.height
    next_state = _state_index(next_x, next_y, next_heading, config.width)
    transitions = DeterministicTransitions(next_state=next_state)

    spatial_hash = (
        x * np.int64(73_856_093)
        ^ y * np.int64(19_349_663)
        ^ heading * np.int64(83_492_791)
        ^ np.int64(config.seed)
    )
    axis_coordinate = np.where(np.isin(heading, (0, 2)), x, y)
    road_type = np.full(config.num_states, 3, dtype=np.int8)
    road_type[axis_coordinate % 7 == 0] = 2
    road_type[axis_coordinate % 15 == 0] = 1
    road_type[axis_coordinate % 55 == 0] = 0
    road_type[spatial_hash % 97 == 0] = 4
    road_type[spatial_hash % 193 == 0] = 5

    speed = np.clip(
        np.array([5, 4, 3, 2, 2, 1], dtype=np.int8)[road_type]
        + ((spatial_hash % 5) == 0).astype(np.int8)
        - ((spatial_hash % 11) == 0).astype(np.int8),
        0,
        5,
    )
    lanes = np.clip(
        np.array([4, 3, 2, 1, 1, 0], dtype=np.int8)[road_type]
        + ((spatial_hash % 13) == 0).astype(np.int8),
        0,
        4,
    )
    segment_length = 0.06 + (spatial_hash % 15).astype(np.float32) / 100.0

    successor_road_type = road_type[np.asarray(next_state)]
    successor_speed = speed[np.asarray(next_state)]
    successor_lanes = lanes[np.asarray(next_state)]
    successor_length = segment_length[np.asarray(next_state)]
    intersection_hash = (
        x[:, None] * np.int64(2_654_435_761)
        ^ y[:, None] * np.int64(805_459_861)
        ^ np.arange(config.num_actions, dtype=np.int64)[None, :]
    )
    turn_category = np.broadcast_to(
        np.array([1, 2, 3], dtype=np.int8),
        (config.num_states, config.num_actions),
    ).copy()
    turn_category[:, 0] = np.where(intersection_hash[:, 0] % 7 == 0, 0, 1)
    turn_category[:, 2] = np.where(intersection_hash[:, 2] % 7 == 0, 4, 3)

    raw_features = np.zeros(
        (config.num_states, config.num_actions, len(RAW_FEATURE_NAMES)),
        dtype=np.float32,
    )
    rows = np.arange(config.num_states)[:, None]
    actions = np.arange(config.num_actions)[None, :]
    raw_features[rows, actions, successor_road_type] = successor_length
    raw_features[rows, actions, 6 + successor_speed] = successor_length
    raw_features[rows, actions, 12 + successor_lanes] = successor_length
    raw_features[rows, actions, 17 + turn_category] = 1.0
    return RoadNetwork(
        transitions=transitions,
        raw_features=raw_features,
        segment_lengths=segment_length,
        road_types=road_type,
    )


def build_tasks(config: RoadStudyConfig) -> list[MCEIRLTask]:
    """Create acyclic classes of reasonably good origin-destination paths."""
    margin = 3
    if config.width >= 100:
        prefix_steps, branch_x, branch_y, suffix_steps = 8, 6, 6, 12
    else:
        prefix_steps, branch_x, branch_y, suffix_steps = 4, 4, 4, 4
    if prefix_steps + branch_x + branch_y + suffix_steps != config.horizon:
        raise ValueError("task path phases must sum to the configured horizon")
    x_span = prefix_steps + branch_x + suffix_steps
    usable_x = config.width - 2 * margin - x_span
    usable_y = config.height - 2 * margin - branch_y
    tasks = []
    for task_index in range(config.num_tasks):
        origin_x = margin + (task_index * 37) % usable_x
        origin_y = margin + (task_index * 53) % usable_y
        branch_start_x = origin_x + prefix_steps
        branch_end_x = branch_start_x + branch_x
        branch_end_y = origin_y + branch_y
        destination_x = branch_end_x + suffix_steps
        destination_y = branch_end_y

        prefix = _state_index(
            np.arange(origin_x, branch_start_x + 1, dtype=np.int64),
            np.full(prefix_steps + 1, origin_y, dtype=np.int64),
            np.full(prefix_steps + 1, 1, dtype=np.int64),
            config.width,
        )
        xx, yy, hh = np.meshgrid(
            np.arange(branch_start_x, branch_end_x + 1, dtype=np.int64),
            np.arange(origin_y, branch_end_y + 1, dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            indexing="xy",
        )
        branch = _state_index(xx.ravel(), yy.ravel(), hh.ravel(), config.width)
        suffix = _state_index(
            np.arange(branch_end_x, destination_x + 1, dtype=np.int64),
            np.full(suffix_steps + 1, branch_end_y, dtype=np.int64),
            np.full(suffix_steps + 1, 1, dtype=np.int64),
            config.width,
        )
        active = np.unique(np.concatenate([prefix, branch, suffix]))
        terminal = _state_index(
            np.array([destination_x]),
            np.array([destination_y]),
            np.array([1]),
            config.width,
        )
        initial = int(
            _state_index(
                np.array([origin_x]),
                np.array([origin_y]),
                np.array([1]),
                config.width,
            )[0]
        )
        tasks.append(
            MCEIRLTask(
                task_id=f"od_{task_index:03d}",
                initial_state=initial,
                terminal_states=terminal,
                active_states=active,
                horizon=config.horizon,
            )
        )
    return tasks


def identified_feature_basis(
    config: RoadStudyConfig,
    network: RoadNetwork,
    tasks: list[MCEIRLTask],
) -> tuple[np.ndarray, list[str], np.ndarray, list[int], np.ndarray]:
    """Select a full-rank action-contrast basis from the 22 observed counts."""
    compiled = compile_mce_irl_tasks(
        tasks,
        network.transitions,
        network.raw_features,
        _placeholder_panel(tasks, network.transitions),
    )
    compiled_features = np.asarray(compiled.feature_matrix)
    contrasts = (compiled_features[:, 1:, :] - compiled_features[:, :1, :]).reshape(
        -1, len(RAW_FEATURE_NAMES)
    )
    column_scale = np.sqrt(np.mean(np.square(contrasts.astype(np.float64)), axis=0))
    column_scale[column_scale == 0] = 1.0
    normalized = contrasts / column_scale
    rank_source = normalized.astype(np.float32)
    _, _, pivots = qr(normalized, mode="economic", pivoting=True)
    rank = int(np.linalg.matrix_rank(rank_source))
    selected_unsorted: list[int] = []
    for pivot in map(int, pivots):
        candidate = selected_unsorted + [pivot]
        if np.linalg.matrix_rank(rank_source[:, candidate]) > len(selected_unsorted):
            selected_unsorted.append(pivot)
        if len(selected_unsorted) == rank:
            break
    selected = sorted(selected_unsorted)
    selected_scale = column_scale[selected].astype(np.float32)
    fit_features = network.raw_features[:, :, selected] / selected_scale
    fit_names = [RAW_FEATURE_NAMES[index] for index in selected]

    raw_costs = np.array(
        [
            0.20,
            0.28,
            0.38,
            0.54,
            0.66,
            0.82,
            0.80,
            0.68,
            0.56,
            0.44,
            0.34,
            0.26,
            0.72,
            0.57,
            0.46,
            0.36,
            0.29,
            0.42,
            0.24,
            0.03,
            0.27,
            0.48,
        ],
        dtype=np.float32,
    )
    theta = -config.reward_scale * raw_costs[selected] * selected_scale
    return fit_features, fit_names, theta, selected, selected_scale


def _placeholder_panel(tasks: list[MCEIRLTask], transitions: DeterministicTransitions) -> Panel:
    trajectories = []
    global_next = np.asarray(transitions.next_state)
    for task in tasks:
        active = set(np.asarray(task.active_states, dtype=int))
        start = int(task.initial_state)
        action = next(
            action
            for action, successor in enumerate(global_next[start])
            if int(successor) in active
        )
        successor = int(global_next[start, action])
        trajectories.append(
            Trajectory(
                states=jnp.array([start]),
                actions=jnp.array([action]),
                next_states=jnp.array([successor]),
                metadata={"task_id": task.task_id},
            )
        )
    return Panel(trajectories)


def _expert_policies(
    config: RoadStudyConfig,
    network: RoadNetwork,
    tasks: list[MCEIRLTask],
    fit_features: np.ndarray,
    fit_names: list[str],
    theta: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray]:
    placeholder = _placeholder_panel(tasks, network.transitions)
    compiled = compile_mce_irl_tasks(
        tasks,
        network.transitions,
        fit_features,
        placeholder,
    )
    problem = DDCProblem(
        num_states=compiled.transitions.num_states,
        num_actions=config.num_actions,
        discount_factor=1.0,
        num_periods=config.horizon,
    )
    reward = ActionDependentReward(
        feature_matrix=compiled.feature_matrix,
        parameter_names=fit_names,
    )
    operator = SoftBellmanOperator(
        problem,
        compiled.transitions,
        terminal_states=compiled.terminal_states,
    )
    solver = MCEIRLEstimator(MCEIRLConfig(compute_se=False))
    if config.driver_types < 1:
        raise ValueError("driver_types must be positive")
    if config.driver_types == 1:
        factors = np.ones(1)
    else:
        factors = np.linspace(1.0, config.exploratory_scale, config.driver_types)
    driver_parameters = theta[None, :] * factors[:, None]
    policies = []
    for driver_theta in driver_parameters:
        _, policy, _ = solver._soft_value_iteration(
            operator,
            reward.compute(jnp.asarray(driver_theta)),
            num_periods=config.horizon,
        )
        policies.append(np.asarray(policy))
    return compiled, np.asarray(policies), driver_parameters


def _sample_valid_route(
    *,
    rng: np.random.Generator,
    config: RoadStudyConfig,
    compiled: Any,
    policy: np.ndarray,
    terminal_states: np.ndarray,
    next_state: np.ndarray,
    task: MCEIRLTask,
    driver_id: int,
) -> RawTrip | None:
    task_slice = compiled.task_slices[task.task_id]
    local_to_global = compiled.local_to_global[task.task_id]
    state = task_slice.start + compiled.global_to_local[task.task_id][int(task.initial_state)]
    states = []
    actions = []
    next_states = []
    for period in range(config.horizon):
        if terminal_states[state]:
            break
        action = int(rng.choice(config.num_actions, p=policy[period, state]))
        successor = int(next_state[state, action])
        states.append(int(local_to_global[state - task_slice.start]))
        actions.append(action)
        next_states.append(int(local_to_global[successor - task_slice.start]))
        state = successor
    reached_terminal = bool(terminal_states[state])
    if not reached_terminal or len(states) < 10 or len(set(states)) != len(states):
        return None
    return RawTrip(
        states=np.asarray(states, dtype=np.int32),
        actions=np.asarray(actions, dtype=np.int32),
        next_states=np.asarray(next_states, dtype=np.int32),
        driver_id=driver_id,
        task_id=task.task_id,
    )


def generate_raw_trips(
    config: RoadStudyConfig,
    network: RoadNetwork,
    tasks: list[MCEIRLTask],
    fit_features: np.ndarray,
    fit_names: list[str],
    theta: np.ndarray,
) -> tuple[list[RawTrip], np.ndarray]:
    """Generate raw records, including records that the paper-style filter rejects."""
    compiled, policies, driver_parameters = _expert_policies(
        config,
        network,
        tasks,
        fit_features,
        fit_names,
        theta,
    )
    rng = np.random.default_rng(config.seed)
    terminal_states = np.asarray(compiled.terminal_states)
    next_state = np.asarray(compiled.transitions.next_state)
    retained = []
    trip_index = 0
    while len(retained) < config.retained_trips:
        task = tasks[len(retained) % len(tasks)]
        driver_id = trip_index % config.num_drivers
        route = _sample_valid_route(
            rng=rng,
            config=config,
            compiled=compiled,
            policy=policies[driver_id % config.driver_types],
            terminal_states=terminal_states,
            next_state=next_state,
            task=task,
            driver_id=driver_id,
        )
        trip_index += 1
        if route is not None:
            retained.append(route)
        if trip_index > config.retained_trips * 50:
            raise RuntimeError("could not generate enough valid road trips")

    discarded_per_reason = config.discarded_trips // 3
    if discarded_per_reason * 3 != config.discarded_trips:
        raise ValueError("discarded_trips must be divisible by three")
    discarded = []
    for index in range(discarded_per_reason):
        base = retained[index % len(retained)]
        length = min(9, len(base.states))
        discarded.append(
            RawTrip(
                states=base.states[:length],
                actions=base.actions[:length],
                next_states=base.next_states[:length],
                driver_id=base.driver_id,
                task_id=base.task_id,
            )
        )
    for index in range(discarded_per_reason):
        base = retained[(index + discarded_per_reason) % len(retained)]
        cycle_states = base.states.copy()
        cycle_states[-1] = cycle_states[0]
        discarded.append(
            RawTrip(
                states=cycle_states,
                actions=base.actions.copy(),
                next_states=base.next_states.copy(),
                driver_id=base.driver_id,
                task_id=base.task_id,
            )
        )
    for index in range(discarded_per_reason):
        base = retained[(index + 2 * discarded_per_reason) % len(retained)]
        discarded.append(
            RawTrip(
                states=base.states.copy(),
                actions=base.actions.copy(),
                next_states=base.next_states.copy(),
                driver_id=base.driver_id,
                task_id=base.task_id,
                noisy=True,
            )
        )

    raw = retained + discarded
    rng.shuffle(raw)
    return raw, driver_parameters


def filter_raw_trips(raw: list[RawTrip]) -> tuple[list[RawTrip], dict[str, int]]:
    """Apply the three exclusion rules described in the paper."""
    retained = []
    reasons: Counter[str] = Counter()
    for trip in raw:
        if trip.noisy:
            reasons["noisy"] += 1
        elif len(trip.states) < 10:
            reasons["too_short"] += 1
        elif len(set(map(int, trip.states))) != len(trip.states):
            reasons["cyclic"] += 1
        else:
            retained.append(trip)
    return retained, dict(reasons)


def _to_trajectory(trip: RawTrip, individual_id: int) -> Trajectory:
    return Trajectory(
        states=jnp.asarray(trip.states),
        actions=jnp.asarray(trip.actions),
        next_states=jnp.asarray(trip.next_states),
        individual_id=individual_id,
        metadata={"task_id": trip.task_id, "driver_id": trip.driver_id},
    )


def split_trips(
    config: RoadStudyConfig,
    retained: list[RawTrip],
) -> tuple[Panel, Panel]:
    """Create the exact 20/80 retained split while keeping every task in train."""
    rng = np.random.default_rng(config.seed + 1)
    by_task: dict[Hashable, list[int]] = defaultdict(list)
    for index, trip in enumerate(retained):
        by_task[trip.task_id].append(index)
    mandatory = []
    for indices in by_task.values():
        mandatory.append(int(rng.choice(indices)))
    remaining = np.setdiff1d(np.arange(len(retained)), np.asarray(mandatory))
    extra = rng.choice(
        remaining,
        size=config.train_trips - len(mandatory),
        replace=False,
    )
    train_indices = np.concatenate([np.asarray(mandatory), extra])
    test_indices = np.setdiff1d(np.arange(len(retained)), train_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train = Panel(
        [_to_trajectory(retained[index], position) for position, index in enumerate(train_indices)]
    )
    test = Panel(
        [_to_trajectory(retained[index], position) for position, index in enumerate(test_indices)]
    )
    return train, test


def evaluate_routes(
    model: MCEIRL,
    *,
    path_panel: Panel,
    probability_panel: Panel,
    segment_lengths: np.ndarray,
) -> dict[str, float]:
    """Compute the Table 1 metrics with the paper's train/test semantics."""
    if model.task_policy_ is None or model._compiled_tasks is None:
        raise RuntimeError("task policies are unavailable")
    compiled = model._compiled_tasks
    terminal_states = np.asarray(compiled.terminal_states)
    next_state = np.asarray(compiled.transitions.next_state)

    log_probabilities = []
    for trajectory in probability_panel.trajectories:
        task_id = trajectory.metadata["task_id"]
        mapping = compiled.global_to_local[task_id]
        task_policy = model.task_policy_[task_id]
        log_probability = 0.0
        for period, (state, action) in enumerate(
            zip(np.asarray(trajectory.states), np.asarray(trajectory.actions))
        ):
            probability = task_policy[period, mapping[int(state)], int(action)]
            log_probability += float(np.log(max(probability, 1e-300)))
        log_probabilities.append(log_probability)

    distance_matches = []
    for trajectory in path_panel.trajectories:
        task_id = trajectory.metadata["task_id"]
        actual_states = np.asarray(trajectory.states, dtype=int)
        task_slice = compiled.task_slices[task_id]
        mapping = compiled.global_to_local[task_id]
        local_to_global = compiled.local_to_global[task_id]
        task_policy = model.task_policy_[task_id]
        state = int(actual_states[0])
        predicted_states = []
        for period in range(task_policy.shape[0]):
            local_state = mapping[state]
            compiled_state = task_slice.start + local_state
            if terminal_states[compiled_state]:
                break
            predicted_states.append(state)
            action = int(np.argmax(task_policy[period, local_state]))
            successor = int(next_state[compiled_state, action])
            state = int(local_to_global[successor - task_slice.start])

        actual_distance = float(segment_lengths[actual_states].sum())
        shared_states = np.intersect1d(actual_states, np.asarray(predicted_states))
        shared_distance = float(segment_lengths[shared_states].sum())
        distance_matches.append(shared_distance / actual_distance)

    matches = np.asarray(distance_matches)
    return {
        "distance_match_percent": 100.0 * float(matches.mean()),
        "routes_at_least_90_percent": 100.0 * float(np.mean(matches >= 0.9)),
        "average_log_probability": float(np.mean(log_probabilities)),
    }


def shape_checks(
    config: RoadStudyConfig,
    network: RoadNetwork,
    fit_features: np.ndarray,
    filter_counts: dict[str, int],
) -> dict[str, Any]:
    """Return paper-contract checks that distinguish structure from scale."""
    paper_scale = config == PAPER_SHAPE
    road_type_counts = np.bincount(network.road_types, minlength=6)
    checks = {
        "states_over_300000": network.transitions.num_states > 300_000,
        "actions_over_900000": int(network.transitions.valid_action.sum()) > 900_000,
        "spatial_segment_topology": True,
        "deterministic": True,
        "feature_families": 4,
        "raw_feature_counts": network.raw_features.shape[-1],
        "identified_fit_features": fit_features.shape[-1],
        "all_road_types_present": bool(np.all(road_type_counts > 0)),
        "raw_trips_approximately_13000": 12_000 <= config.raw_trips <= 14_000,
        "discard_fraction": config.discarded_trips / config.raw_trips,
        "training_fraction_of_retained": config.train_trips / config.retained_trips,
        "test_trips": config.test_trips,
        "drivers": config.num_drivers,
        "filter_reasons": filter_counts,
        "paper_scale": paper_scale,
    }
    checks["passed"] = bool(
        checks["raw_feature_counts"] == 22
        and checks["identified_fit_features"] >= 12
        and checks["all_road_types_present"]
        and (
            not paper_scale
            or (
                checks["states_over_300000"]
                and checks["actions_over_900000"]
                and abs(checks["discard_fraction"] - 0.30) < 1e-12
                and abs(checks["training_fraction_of_retained"] - 0.20) < 1e-3
                and checks["test_trips"] == 7_403
                and checks["drivers"] == 25
                and sum(filter_counts.values()) == 3_966
                and set(filter_counts) == {"too_short", "cyclic", "noisy"}
            )
        )
    )
    return checks


def run(config: RoadStudyConfig, *, output: Path) -> dict[str, Any]:
    """Run the synthetic study and write a machine-readable receipt."""
    started = time.perf_counter()
    network = build_road_network(config)
    tasks = build_tasks(config)
    (
        fit_features,
        fit_names,
        theta,
        selected_columns,
        feature_scale,
    ) = identified_feature_basis(
        config,
        network,
        tasks,
    )
    raw, driver_parameters = generate_raw_trips(
        config,
        network,
        tasks,
        fit_features,
        fit_names,
        theta,
    )
    retained, filter_counts = filter_raw_trips(raw)
    if len(raw) != config.raw_trips or len(retained) != config.retained_trips:
        raise RuntimeError("generated raw trip counts do not match the configured pipeline")
    train, test = split_trips(config, retained)

    model = MCEIRL(
        n_states=config.num_states,
        n_actions=config.num_actions,
        discount=1.0,
        horizon=config.horizon,
        feature_matrix=fit_features,
        feature_names=fit_names,
        compute_se=False,
        l2_regularization=config.l2_regularization,
    )
    fit_started = time.perf_counter()
    model.fit(train, transitions=network.transitions, tasks=tasks)
    fit_seconds = time.perf_counter() - fit_started
    compiled_features = np.asarray(model._reward_fn.feature_matrix)
    compiled_contrast = (compiled_features[:, 1:, :] - compiled_features[:, :1, :]).reshape(
        -1, compiled_features.shape[-1]
    )
    contrast_rank = int(np.linalg.matrix_rank(compiled_contrast))
    synthetic_metrics = evaluate_routes(
        model,
        path_panel=test,
        probability_panel=train,
        segment_lengths=network.segment_lengths,
    )
    coefficient_rmse = float(np.sqrt(np.mean((model.coef_ - theta) ** 2)))
    target_gaps = {name: synthetic_metrics[name] - target for name, target in PAPER_TARGETS.items()}

    config_payload = asdict(config)
    config_payload.update(
        {
            "num_states": config.num_states,
            "num_actions": config.num_actions,
        }
    )
    receipt = {
        "claim": (
            "Paper-faithful generated-data validation of the road-choice adapter. "
            "This does not reproduce Table 1 from the unavailable Pittsburgh data."
        ),
        "paper": {
            "citation": "Ziebart, Maas, Bagnell, and Dey (2008)",
            "reported_table_1_targets": PAPER_TARGETS,
        },
        "config": config_payload,
        "network": {
            "state_definition": "directed road segment ending at an intersection",
            "action_definition": "left, straight, or right intersection transition",
            "transition_orientation": "next_state[state, action]",
            "num_states": network.transitions.num_states,
            "num_action_slots": int(network.transitions.valid_action.sum()),
            "raw_feature_names": RAW_FEATURE_NAMES,
            "fit_feature_names": fit_names,
            "normalization": {
                "selected_columns": selected_columns,
                "rule": "pivoted full-rank action-contrast basis",
                "feature_scale": feature_scale.tolist(),
            },
            "shape_checks": shape_checks(
                config,
                network,
                fit_features,
                filter_counts,
            ),
        },
        "data": {
            "raw_trips": len(raw),
            "discarded_trips": sum(filter_counts.values()),
            "discard_reasons": filter_counts,
            "retained_trips": len(retained),
            "train_trips": train.num_individuals,
            "test_trips": test.num_individuals,
            "train_observations": train.num_observations,
            "test_observations": test.num_observations,
            "drivers": config.num_drivers,
            "driver_types": config.driver_types,
            "driver_parameter_spread": float(np.mean(np.std(driver_parameters, axis=0))),
            "tasks": config.num_tasks,
        },
        "fit": {
            "estimator": "econirl.estimators.MCEIRL",
            "road_specific_estimator_logic": False,
            "compiled_states": int(model._compiled_tasks.transitions.num_states),
            "contrast_rank": contrast_rank,
            "num_parameters": len(fit_names),
            "converged": bool(model.converged_),
            "termination_reason": model.termination_reason_,
            "stationarity_residual": model.feature_residual_,
            "coefficient_rmse": coefficient_rmse,
            "fit_seconds": fit_seconds,
        },
        "synthetic_metrics": synthetic_metrics,
        "paper_target_gaps": target_gaps,
        "metric_semantics": {
            "distance_match_percent": (
                "most likely fitted path versus held-out path, distance weighted"
            ),
            "routes_at_least_90_percent": "held-out paths with at least 90 percent shared distance",
            "average_log_probability": "mean fitted log probability on the training paths",
        },
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax": jax.__version__,
        },
    }
    receipt["passed"] = bool(
        receipt["network"]["shape_checks"]["passed"]
        and receipt["fit"]["converged"]
        and receipt["fit"]["stationarity_residual"] <= 0.02
        and np.isfinite(list(synthetic_metrics.values())).all()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = SMOKE_SHAPE if args.smoke else PAPER_SHAPE
    receipt = run(config, output=args.output)
    if args.quiet:
        display_output = (
            args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output
        )
        print(f"wrote {display_output}")
        print(f"status: {'passed' if receipt['passed'] else 'failed'}")
    else:
        print(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
