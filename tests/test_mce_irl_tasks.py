"""General task compilation for MCE-IRL."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.tasks import MCEIRLTask, compile_mce_irl_tasks
from econirl.core.types import Panel, Trajectory
from econirl.estimators import MCEIRL
from econirl.transitions import DeterministicTransitions


def test_task_compiler_builds_compact_disjoint_mdp() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array(
            [[1, 2], [3, 2], [3, 1], [3, -1]],
            dtype=int,
        ),
        valid_action=np.array([[True, True], [True, True], [True, True], [True, False]]),
    )
    features = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    tasks = [
        MCEIRLTask(
            task_id="left",
            initial_state=0,
            terminal_states=np.array([3]),
            active_states=np.array([0, 1, 3]),
            horizon=4,
        ),
        MCEIRLTask(
            task_id="right",
            initial_state=0,
            terminal_states=np.array([3]),
            active_states=np.array([0, 2, 3]),
            horizon=4,
        ),
    ]
    panel = Panel(
        [
            Trajectory(
                states=jnp.array([0, 1]),
                actions=jnp.array([0, 0]),
                next_states=jnp.array([1, 3]),
                metadata={"task_id": "left"},
            ),
            Trajectory(
                states=jnp.array([0, 2]),
                actions=jnp.array([1, 0]),
                next_states=jnp.array([2, 3]),
                metadata={"task_id": "right"},
            ),
        ]
    )

    compiled = compile_mce_irl_tasks(tasks, transitions, features, panel)

    assert compiled.transitions.next_state.shape == (6, 2)
    assert compiled.feature_matrix.shape == (6, 2, 3)
    assert compiled.task_slices["left"] == slice(0, 3)
    assert compiled.task_slices["right"] == slice(3, 6)
    np.testing.assert_array_equal(
        np.asarray(compiled.panel.trajectories[1].states),
        np.array([3, 4]),
    )
    np.testing.assert_array_equal(
        np.asarray(compiled.panel.trajectories[1].next_states),
        np.array([4, 5]),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(np.asarray(compiled.terminal_states)),
        np.array([2, 5]),
    )
    np.testing.assert_allclose(compiled.initial_state_dist, [0.5, 0, 0, 0.5, 0, 0])


def test_task_compiler_rejects_trajectory_outside_subgraph() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1], [3], [3], [3]]),
    )
    task = MCEIRLTask(
        task_id=1,
        initial_state=0,
        terminal_states=np.array([3]),
        active_states=np.array([0, 1, 3]),
        horizon=3,
    )
    panel = Panel(
        [
            Trajectory(
                states=jnp.array([0, 2]),
                actions=jnp.array([0, 0]),
                next_states=jnp.array([2, 3]),
                metadata={"task_id": 1},
            )
        ]
    )

    try:
        compile_mce_irl_tasks(
            [task],
            transitions,
            np.ones((4, 1, 1)),
            panel,
        )
    except ValueError as exc:
        assert "leaves its active subgraph" in str(exc)
    else:
        raise AssertionError("out-of-subgraph trajectory was accepted")


def test_task_compiler_uses_task_start_distributions_and_demo_mix() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1], [1], [3], [3]]),
    )
    tasks = [
        MCEIRLTask(
            task_id="a",
            initial_state_dist=np.array([0.25, 0.75, 0.0, 0.0]),
            terminal_states=np.array([1]),
            active_states=np.array([0, 1]),
            horizon=2,
        ),
        MCEIRLTask(
            task_id="b",
            initial_state=2,
            terminal_states=np.array([3]),
            active_states=np.array([2, 3]),
            horizon=2,
        ),
    ]
    trajectories = [
        Trajectory(
            states=jnp.array([0]),
            actions=jnp.array([0]),
            next_states=jnp.array([1]),
            metadata={"task_id": "a"},
        )
    ]
    trajectories.extend(
        Trajectory(
            states=jnp.array([2]),
            actions=jnp.array([0]),
            next_states=jnp.array([3]),
            metadata={"task_id": "b"},
        )
        for _ in range(3)
    )

    compiled = compile_mce_irl_tasks(
        tasks,
        transitions,
        np.ones((4, 1, 1)),
        Panel(trajectories),
    )

    np.testing.assert_allclose(
        compiled.initial_state_dist,
        np.array([0.0625, 0.1875, 0.75, 0.0]),
    )
    np.testing.assert_allclose(np.asarray(compiled.feature_matrix)[[1, 3]], 0.0)


def test_task_compiler_rejects_transition_mismatch() -> None:
    transitions = DeterministicTransitions(next_state=np.array([[1], [1]]))
    task = MCEIRLTask(
        task_id="route",
        initial_state=0,
        terminal_states=np.array([1]),
        horizon=1,
    )
    panel = Panel(
        [
            Trajectory(
                states=jnp.array([0]),
                actions=jnp.array([0]),
                next_states=jnp.array([0]),
                metadata={"task_id": "route"},
            )
        ]
    )

    with pytest.raises(ValueError, match="disagrees with deterministic transitions"):
        compile_mce_irl_tasks(
            [task],
            transitions,
            np.ones((2, 1, 1)),
            panel,
        )


def test_public_wrapper_fits_shared_reward_across_tasks() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1, 1], [1, -1]]),
        valid_action=np.array([[True, True], [True, False]]),
    )
    features = np.zeros((2, 2, 1), dtype=float)
    features[0, 1, 0] = 1.0
    tasks = [
        MCEIRLTask(
            task_id="morning",
            initial_state=0,
            terminal_states=np.array([1]),
            horizon=1,
        ),
        MCEIRLTask(
            task_id="evening",
            initial_state=0,
            terminal_states=np.array([1]),
            horizon=1,
        ),
    ]
    rng = np.random.default_rng(41)
    trajectories = []
    for task_id in ("morning", "evening"):
        for individual in range(300):
            action = int(rng.random() < 0.7)
            trajectories.append(
                Trajectory(
                    states=jnp.array([0]),
                    actions=jnp.array([action]),
                    next_states=jnp.array([1]),
                    individual_id=f"{task_id}-{individual}",
                    metadata={"task_id": task_id},
                )
            )

    model = MCEIRL(
        n_states=2,
        n_actions=2,
        discount=1.0,
        horizon=1,
        feature_matrix=features,
        feature_names=["action_one"],
        compute_se=False,
    )
    model.fit(
        Panel(trajectories),
        transitions=transitions,
        tasks=tasks,
    )

    assert model.converged_
    assert set(model.task_policy_) == {"morning", "evening"}
    morning = model.predict_proba(
        np.array([0]),
        task_id="morning",
        period=0,
    )
    evening = model.predict_proba(
        np.array([0]),
        task_id="evening",
        period=0,
    )
    np.testing.assert_allclose(morning, evening)
    assert abs(float(morning[0, 1]) - 0.7) < 0.04

    simulated = model.simulate(8, task_id="morning", seed=9)
    assert simulated.num_individuals == 8
    assert all(len(trajectory) == 1 for trajectory in simulated.trajectories)
    assert all(int(trajectory.next_states[0]) == 1 for trajectory in simulated.trajectories)

    counterfactual = model.counterfactual(params={"action_one": model.params_["action_one"] + 1.0})
    assert counterfactual.params["action_one"] > model.params_["action_one"]
    assert float(np.abs(counterfactual.policy_change).max()) > 0.05
    assert counterfactual.metadata["reward_level_identified"] is False

    masked_transitions = DeterministicTransitions(
        next_state=np.array([[1, -1], [1, -1]]),
        valid_action=np.array([[True, False], [True, False]]),
    )
    transition_counterfactual = model.counterfactual(transitions=masked_transitions)
    np.testing.assert_allclose(transition_counterfactual.counterfactual_policy[:, 1], 0.0)

    with pytest.raises(ValueError, match="exactly one"):
        model.counterfactual()
