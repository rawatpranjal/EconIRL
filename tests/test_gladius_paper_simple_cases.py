"""Cheap known-truth gates for the literal GLADIUS paper training path."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.gladius import GLADIUSEstimator
from econirl.preferences.action_reward import ActionDependentReward
from validation.estimators.gladius.paper_table2_mape import paper_training_config


def _one_state_panel() -> tuple[Panel, np.ndarray]:
    """Exact action frequencies for rewards (-3, -4) in an absorbing state."""
    rewards = np.asarray([-3.0, -4.0], dtype=np.float32)
    probabilities = np.exp(rewards - rewards.max())
    probabilities /= probabilities.sum()
    n_observations = 32 * 20
    n_action_zero = int(round(n_observations * probabilities[0]))
    actions = np.asarray(
        [0] * n_action_zero + [1] * (n_observations - n_action_zero),
        dtype=np.int32,
    )
    np.random.default_rng(0).shuffle(actions)
    trajectories = []
    for individual in range(32):
        state = jnp.zeros(20, dtype=jnp.int32)
        start = individual * 20
        trajectories.append(
            Trajectory(
                states=state,
                actions=jnp.asarray(actions[start : start + 20]),
                next_states=state,
            )
        )
    return Panel(trajectories=trajectories), rewards


def test_paper_recipe_recovers_cardinal_reward_in_one_state_case() -> None:
    """The paper path must work before any Table 2 replication is attempted."""
    panel, true_reward = _one_state_panel()
    feature_matrix = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    utility = ActionDependentReward(feature_matrix, ["action_zero", "action_one"])
    problem = DDCProblem(
        num_states=1,
        num_actions=2,
        discount_factor=0.95,
        scale_parameter=1.0,
        state_dim=1,
        state_encoder=lambda states: jnp.zeros((len(states), 1)),
    )
    config = paper_training_config(
        anchor_rewards=(-4.0,),
        batch_size=2,
        max_epochs=100,
    )

    result = GLADIUSEstimator(config).estimate(
        panel=panel,
        utility=utility,
        problem=problem,
        transitions=jnp.ones((2, 1, 1)),
    )

    estimated_reward = np.asarray(result.metadata["reward_table"])[0]
    reward_mape = float(np.mean(np.abs((estimated_reward - true_reward) / true_reward)) * 100)
    true_policy = np.exp(true_reward - true_reward.max())
    true_policy /= true_policy.sum()
    policy_error = float(np.max(np.abs(np.asarray(result.policy)[0] - true_policy)))

    assert reward_mape <= 2.0, (
        f"paper_minimax failed cardinal reward recovery: MAPE={reward_mape:.3f}%"
    )
    assert policy_error <= 0.025, f"paper_minimax policy error={policy_error:.4f}"
    assert abs(float(estimated_reward[1] - true_reward[1])) <= 0.1


def _three_state_bus_case() -> tuple[Panel, np.ndarray, np.ndarray, np.ndarray]:
    rewards = np.asarray([[-1.0, -5.0], [-2.0, -5.0], [-3.0, -5.0]], dtype=np.float32)
    transitions = np.zeros((2, 3, 3), dtype=np.float32)
    for state in range(3):
        transitions[0, state, min(state + 1, 2)] = 1.0
        transitions[1, state, 0] = 1.0

    value = np.zeros(3, dtype=np.float64)
    for _ in range(10_000):
        q_values = rewards + 0.95 * np.stack(
            [value[[1, 2, 2]], value[[0, 0, 0]]],
            axis=1,
        )
        updated = np.logaddexp(q_values[:, 0], q_values[:, 1])
        if np.max(np.abs(updated - value)) < 1e-12:
            break
        value = updated
    policy = np.exp(q_values - q_values.max(axis=1, keepdims=True))
    policy /= policy.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(4)
    trajectories = []
    for _ in range(64):
        states = []
        actions = []
        next_states = []
        state = 0
        for _period in range(100):
            action = int(rng.choice(2, p=policy[state]))
            next_state = 0 if action == 1 else min(state + 1, 2)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            state = next_state
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
            )
        )
    return Panel(trajectories=trajectories), rewards, transitions, policy


def test_paper_recipe_recovers_small_bus_reward_policy_and_anchor() -> None:
    panel, true_reward, transitions, true_policy = _three_state_bus_case()
    features = np.zeros((3, 2, 2), dtype=np.float32)
    features[:, 0, 0] = -np.arange(1, 4)
    features[:, 1, 1] = -1.0
    utility = ActionDependentReward(jnp.asarray(features), ["mileage", "replacement"])
    problem = DDCProblem(
        num_states=3,
        num_actions=2,
        discount_factor=0.95,
        scale_parameter=1.0,
        state_dim=1,
        state_encoder=lambda states: (jnp.asarray(states) + 1.0)[:, None],
    )
    config = paper_training_config(
        anchor_rewards=(-5.0, -5.0, -5.0),
        batch_size=2,
        max_epochs=200,
    )

    result = GLADIUSEstimator(config).estimate(
        panel=panel,
        utility=utility,
        problem=problem,
        transitions=jnp.asarray(transitions),
    )

    estimated_reward = np.asarray(result.metadata["reward_table"])
    reward_mape = float(np.mean(np.abs((estimated_reward - true_reward) / true_reward)) * 100)
    policy_error = float(np.mean(np.abs(np.asarray(result.policy) - true_policy)))
    anchor_error = float(np.max(np.abs(estimated_reward[:, 1] - true_reward[:, 1])))

    assert reward_mape <= 10.0, f"small-bus reward MAPE={reward_mape:.3f}%"
    assert policy_error <= 0.05, f"small-bus mean policy error={policy_error:.4f}"
    assert anchor_error <= 0.25, f"small-bus anchor error={anchor_error:.4f}"
