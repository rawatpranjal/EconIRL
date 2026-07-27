#!/usr/bin/env python3
"""Paper-shaped synthetic validation for Ziebart et al. (2008) MCE-IRL.

This is not an original-data replication. It generates a deterministic road
network with the paper's reported lower-bound dimensions, four feature
families with 22 counts, destination-specific compact path sets, a shared
linear reward, and the reported train/test split.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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

FEATURE_NAMES = [
    *(f"road_type_{index}" for index in range(6)),
    *(f"speed_{index}" for index in range(6)),
    *(f"lanes_{index}" for index in range(5)),
    *(f"turn_{index}" for index in range(5)),
]


@dataclass(frozen=True)
class RoadStudyConfig:
    """Dimensions for one synthetic road-choice study."""

    num_states: int
    num_actions: int
    raw_trips: int
    discarded_trips: int
    train_trips: int
    test_trips: int
    num_tasks: int
    task_states: int
    horizon: int
    seed: int
    reward_scale: float = 1.0
    l2_regularization: float = 1e-4

    @property
    def retained_trips(self) -> int:
        return self.train_trips + self.test_trips


PAPER_SHAPE = RoadStudyConfig(
    num_states=300_001,
    num_actions=3,
    raw_trips=13_220,
    discarded_trips=3_966,
    train_trips=1_851,
    test_trips=7_403,
    num_tasks=25,
    task_states=48,
    horizon=47,
    seed=2008,
)

SMOKE_SHAPE = RoadStudyConfig(
    num_states=3_001,
    num_actions=3,
    raw_trips=160,
    discarded_trips=48,
    train_trips=32,
    test_trips=80,
    num_tasks=4,
    task_states=24,
    horizon=23,
    seed=2008,
)


def build_road_network(
    config: RoadStudyConfig,
) -> tuple[DeterministicTransitions, np.ndarray]:
    """Build deterministic successors and 22 paper-style feature counts."""
    states = np.arange(config.num_states, dtype=np.int32)[:, None]
    actions = np.arange(config.num_actions, dtype=np.int32)[None, :]
    next_state = (states + actions + 1) % config.num_states
    transitions = DeterministicTransitions(next_state=next_state)

    features = np.zeros(
        (config.num_states, config.num_actions, len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    road_length = 0.15 + ((states * 17 + actions * 13) % 85) / 100.0
    road_type = (states + 2 * actions) % 6
    speed = (states // 3 + actions) % 6
    lanes = (states // 7 + 3 * actions) % 5
    turn = (states + actions) % 5
    row = np.arange(config.num_states)[:, None]
    action_row = np.arange(config.num_actions)[None, :]
    features[row, action_row, road_type] = road_length
    features[row, action_row, 6 + speed] = road_length
    features[row, action_row, 12 + lanes] = road_length
    features[row, action_row, 17 + turn] = 1.0
    return transitions, features


def build_tasks(config: RoadStudyConfig) -> list[MCEIRLTask]:
    """Create destination-specific candidate-path subgraphs."""
    spacing = config.num_states // config.num_tasks
    tasks = []
    for task_index in range(config.num_tasks):
        start = task_index * spacing
        active = np.arange(start, start + config.task_states, dtype=np.int32)
        tasks.append(
            MCEIRLTask(
                task_id=f"od_{task_index:02d}",
                initial_state=int(active[0]),
                terminal_states=np.array([active[-1]], dtype=np.int32),
                active_states=active,
                horizon=config.horizon,
            )
        )
    return tasks


def true_parameters() -> np.ndarray:
    """Return a shared cost-based reward over the four feature families."""
    return -np.array(
        [
            0.35,
            0.42,
            0.51,
            0.60,
            0.72,
            0.84,
            0.82,
            0.68,
            0.55,
            0.44,
            0.36,
            0.30,
            0.62,
            0.50,
            0.40,
            0.34,
            0.30,
            0.08,
            0.20,
            0.24,
            0.30,
            0.75,
        ],
        dtype=np.float32,
    )


def _placeholder_panel(tasks: list[MCEIRLTask]) -> Panel:
    trajectories = []
    for task in tasks:
        start = int(task.initial_state)
        trajectories.append(
            Trajectory(
                states=jnp.array([start]),
                actions=jnp.array([0]),
                next_states=jnp.array([start + 1]),
                metadata={"task_id": task.task_id},
            )
        )
    return Panel(trajectories)


def generate_routes(
    config: RoadStudyConfig,
    transitions: DeterministicTransitions,
    features: np.ndarray,
    tasks: list[MCEIRLTask],
) -> tuple[Panel, Panel, np.ndarray]:
    """Generate retained trips from the shared-reward finite-horizon model."""
    compiled = compile_mce_irl_tasks(
        tasks,
        transitions,
        features,
        _placeholder_panel(tasks),
    )
    problem = DDCProblem(
        num_states=compiled.transitions.num_states,
        num_actions=config.num_actions,
        discount_factor=1.0,
        num_periods=config.horizon,
    )
    reward = ActionDependentReward(
        feature_matrix=compiled.feature_matrix,
        parameter_names=FEATURE_NAMES,
    )
    theta = true_parameters() * config.reward_scale
    operator = SoftBellmanOperator(
        problem,
        compiled.transitions,
        terminal_states=compiled.terminal_states,
    )
    solver = MCEIRLEstimator(MCEIRLConfig(compute_se=False))
    _, policy, _ = solver._soft_value_iteration(
        operator,
        reward.compute(jnp.asarray(theta)),
        num_periods=config.horizon,
    )

    rng = np.random.default_rng(config.seed)
    trajectories = []
    for trip_index in range(config.retained_trips):
        task = tasks[trip_index % len(tasks)]
        task_slice = compiled.task_slices[task.task_id]
        local_to_global = compiled.local_to_global[task.task_id]
        state = task_slice.start
        states = []
        actions = []
        next_states = []
        for period in range(config.horizon):
            if bool(compiled.terminal_states[state]):
                break
            action = int(rng.choice(config.num_actions, p=np.asarray(policy[period, state])))
            successor = int(compiled.transitions.next_state[state, action])
            states.append(int(local_to_global[state - task_slice.start]))
            actions.append(action)
            next_states.append(int(local_to_global[successor - task_slice.start]))
            state = successor
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states, dtype=jnp.int32),
                actions=jnp.asarray(actions, dtype=jnp.int32),
                next_states=jnp.asarray(next_states, dtype=jnp.int32),
                individual_id=trip_index,
                metadata={"task_id": task.task_id},
            )
        )

    order = rng.permutation(len(trajectories))
    train_indices = order[: config.train_trips]
    test_indices = order[config.train_trips :]
    return (
        Panel([trajectories[index] for index in train_indices]),
        Panel([trajectories[index] for index in test_indices]),
        theta,
    )


def evaluate_routes(model: MCEIRL, panel: Panel) -> dict[str, float]:
    """Compute the three route metrics reported in Ziebart et al. Table 1."""
    distance_matches = []
    log_probabilities = []
    for trajectory in panel.trajectories:
        task_id = trajectory.metadata["task_id"]
        actual_states = np.asarray(trajectory.states, dtype=int)
        actual_actions = np.asarray(trajectory.actions, dtype=int)
        if len(actual_states) == 0:
            continue

        log_probability = 0.0
        for period, (state, action) in enumerate(zip(actual_states, actual_actions)):
            probability = model.predict_proba(
                np.array([state]),
                task_id=task_id,
                period=period,
            )[0, action]
            log_probability += float(np.log(max(probability, 1e-300)))
        log_probabilities.append(log_probability)

        predicted_states = []
        state = int(actual_states[0])
        task_slice = model._compiled_tasks.task_slices[task_id]
        terminal = model._effective_terminal_states
        for period in range(model._problem.num_periods):
            compiled_state = (
                task_slice.start + model._compiled_tasks.global_to_local[task_id][state]
            )
            if terminal[compiled_state]:
                break
            predicted_states.append(state)
            probabilities = model.predict_proba(
                np.array([state]),
                task_id=task_id,
                period=period,
            )[0]
            action = int(np.argmax(probabilities))
            successor = int(model.transition_model_.next_state[compiled_state, action])
            state = int(
                model._compiled_tasks.local_to_global[task_id][successor - task_slice.start]
            )

        shared = len(set(actual_states).intersection(predicted_states))
        distance_matches.append(shared / len(actual_states))

    matches = np.asarray(distance_matches)
    return {
        "distance_match_percent": 100.0 * float(matches.mean()),
        "routes_at_least_90_percent": 100.0 * float(np.mean(matches >= 0.9)),
        "average_log_probability": float(np.mean(log_probabilities)),
    }


def shape_checks(
    config: RoadStudyConfig,
    transitions: DeterministicTransitions,
    features: np.ndarray,
) -> dict[str, Any]:
    """Return explicit paper-shape contract checks."""
    checks = {
        "states_over_300000": transitions.num_states > 300_000,
        "actions_over_900000": int(transitions.valid_action.sum()) > 900_000,
        "deterministic": True,
        "feature_families": 4,
        "feature_counts": features.shape[-1],
        "raw_trips_approximately_13000": 12_000 <= config.raw_trips <= 14_000,
        "discard_fraction": config.discarded_trips / config.raw_trips,
        "training_fraction_of_retained": config.train_trips / config.retained_trips,
        "test_trips": config.test_trips,
    }
    paper_scale = config == PAPER_SHAPE
    checks["paper_scale"] = paper_scale
    checks["passed"] = checks["feature_counts"] == 22 and (
        not paper_scale
        or (
            checks["states_over_300000"]
            and checks["actions_over_900000"]
            and abs(checks["discard_fraction"] - 0.30) < 1e-12
            and abs(checks["training_fraction_of_retained"] - 0.20) < 1e-3
            and checks["test_trips"] == 7_403
        )
    )
    return checks


def run(config: RoadStudyConfig, *, output: Path) -> dict[str, Any]:
    """Run the synthetic study and write a machine-readable receipt."""
    started = time.perf_counter()
    transitions, features = build_road_network(config)
    tasks = build_tasks(config)
    train, test, theta = generate_routes(config, transitions, features, tasks)
    model = MCEIRL(
        n_states=config.num_states,
        n_actions=config.num_actions,
        discount=1.0,
        horizon=config.horizon,
        feature_matrix=features,
        feature_names=FEATURE_NAMES,
        compute_se=False,
        l2_regularization=config.l2_regularization,
    )
    fit_started = time.perf_counter()
    model.fit(train, transitions=transitions, tasks=tasks)
    fit_seconds = time.perf_counter() - fit_started
    synthetic_metrics = evaluate_routes(model, test)
    coefficient_rmse = float(np.sqrt(np.mean((model.coef_ - theta) ** 2)))
    receipt = {
        "claim": (
            "Paper-shaped synthetic validation. This does not reproduce "
            "Ziebart et al. Table 1 from the withheld Pittsburgh taxi data."
        ),
        "paper": {
            "citation": "Ziebart, Maas, Bagnell, and Dey (2008)",
            "reported_table_1_targets": PAPER_TARGETS,
        },
        "config": asdict(config),
        "network": {
            "transition_orientation": "next_state[state, action]",
            "num_states": transitions.num_states,
            "num_action_slots": int(transitions.valid_action.sum()),
            "feature_names": FEATURE_NAMES,
            "shape_checks": shape_checks(config, transitions, features),
        },
        "data": {
            "train_trips": train.num_individuals,
            "test_trips": test.num_individuals,
            "train_observations": train.num_observations,
            "test_observations": test.num_observations,
        },
        "fit": {
            "converged": bool(model.converged_),
            "termination_reason": model.termination_reason_,
            "stationarity_residual": model.feature_residual_,
            "coefficient_rmse": coefficient_rmse,
            "fit_seconds": fit_seconds,
        },
        "synthetic_metrics": synthetic_metrics,
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
