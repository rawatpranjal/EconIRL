"""Deterministic transition and finite-horizon MCE-IRL contracts."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.transitions import DeterministicTransitions


def test_deterministic_transitions_validate_shape_and_bounds() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1, 2], [2, 0], [2, -1]]),
        valid_action=np.array([[True, True], [True, True], [True, False]]),
    )

    assert transitions.num_states == 3
    assert transitions.num_actions == 2
    np.testing.assert_array_equal(
        np.asarray(transitions.next_state),
        np.array([[1, 2], [2, 0], [2, 0]]),
    )

    with pytest.raises(ValueError, match="same shape"):
        DeterministicTransitions(
            next_state=np.zeros((3, 2), dtype=int),
            valid_action=np.ones((3, 1), dtype=bool),
        )

    with pytest.raises(ValueError, match="valid next states"):
        DeterministicTransitions(
            next_state=np.array([[1, 3], [0, 1], [1, 2]]),
            valid_action=np.ones((3, 2), dtype=bool),
        )


def test_dense_and_deterministic_bellman_backups_agree() -> None:
    next_state = np.array([[1, 2], [2, 0], [0, 1]], dtype=int)
    deterministic = DeterministicTransitions(next_state=next_state)
    dense = np.zeros((2, 3, 3), dtype=float)
    for state in range(3):
        for action in range(2):
            dense[action, state, next_state[state, action]] = 1.0

    problem = DDCProblem(num_states=3, num_actions=2, discount_factor=0.9)
    utility = jnp.array([[0.2, -0.1], [0.4, 0.0], [-0.2, 0.3]])
    value = jnp.array([0.5, -0.25, 0.75])

    dense_result = SoftBellmanOperator(problem, jnp.asarray(dense)).apply(utility, value)
    deterministic_result = SoftBellmanOperator(problem, deterministic).apply(utility, value)

    np.testing.assert_allclose(deterministic_result.Q, dense_result.Q)
    np.testing.assert_allclose(deterministic_result.V, dense_result.V)
    np.testing.assert_allclose(deterministic_result.policy, dense_result.policy)


def test_action_mask_and_terminal_state_are_enforced() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1, 2], [2, 0], [2, -1]]),
        valid_action=np.array([[True, False], [True, True], [True, False]]),
    )
    problem = DDCProblem(
        num_states=3,
        num_actions=2,
        discount_factor=1.0,
        num_periods=5,
    )
    operator = SoftBellmanOperator(
        problem,
        transitions,
        terminal_states=jnp.array([False, False, True]),
    )

    result = operator.apply(
        jnp.array([[-1.0, 100.0], [-1.0, -2.0], [50.0, 50.0]]),
        jnp.array([0.0, 0.0, 10.0]),
    )

    np.testing.assert_allclose(result.policy[0], np.array([1.0, 0.0]))
    np.testing.assert_allclose(result.policy[2], np.array([1.0, 0.0]))
    assert float(result.V[2]) == 0.0
    assert float(result.Q[2, 0]) == 0.0
    assert np.isneginf(float(result.Q[2, 1]))


def test_discount_one_requires_finite_horizon() -> None:
    DDCProblem(
        num_states=3,
        num_actions=2,
        discount_factor=1.0,
        num_periods=5,
    )

    with pytest.raises(ValueError, match="finite-horizon"):
        DDCProblem(
            num_states=3,
            num_actions=2,
            discount_factor=1.0,
        )


def test_finite_horizon_solver_retains_time_indexed_policy() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1, 2], [2, 2], [2, -1]]),
        valid_action=np.array([[True, True], [True, False], [True, False]]),
    )
    problem = DDCProblem(
        num_states=3,
        num_actions=2,
        discount_factor=1.0,
        num_periods=4,
    )
    operator = SoftBellmanOperator(
        problem,
        transitions,
        terminal_states=jnp.array([False, False, True]),
    )
    estimator = MCEIRLEstimator(MCEIRLConfig(compute_se=False))

    values, policy, converged = estimator._soft_value_iteration(
        operator,
        jnp.array([[-1.0, -2.0], [-1.0, 0.0], [0.0, 0.0]]),
        num_periods=4,
    )

    assert converged
    assert values.shape == (4, 3)
    assert policy.shape == (4, 3, 2)
    np.testing.assert_allclose(policy[:, 0].sum(axis=1), 1.0)
    np.testing.assert_allclose(policy[:, 2, 0], 1.0)
    np.testing.assert_allclose(policy[:, 2, 1], 0.0)


def test_finite_horizon_deterministic_fit_uses_masked_policy() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1, 2], [2, 3], [3, 3], [3, -1]]),
        valid_action=np.array([[True, True], [True, True], [True, False], [True, False]]),
    )
    terminal = jnp.array([False, False, False, True])
    horizon = 5
    problem = DDCProblem(
        num_states=4,
        num_actions=2,
        discount_factor=1.0,
        num_periods=horizon,
    )
    features = np.zeros((4, 2, 2), dtype=float)
    features[:3, :, 0] = -1.0
    features[:2, 1, 1] = 1.0
    reward = ActionDependentReward(
        feature_matrix=jnp.asarray(features),
        parameter_names=["step_cost", "skip_bonus"],
    )
    true_params = jnp.array([0.7, 0.4])
    operator = SoftBellmanOperator(problem, transitions, terminal_states=terminal)
    generator = MCEIRLEstimator(MCEIRLConfig(compute_se=False))
    _, expert_policy, _ = generator._soft_value_iteration(
        operator,
        reward.compute(true_params),
        num_periods=horizon,
    )

    rng = np.random.default_rng(812)
    trajectories = []
    for individual in range(400):
        states = []
        actions = []
        next_states = []
        state = 0
        for period in range(horizon):
            action = int(rng.choice(2, p=np.asarray(expert_policy[period, state])))
            successor = int(transitions.next_state[state, action])
            states.append(state)
            actions.append(action)
            next_states.append(successor)
            state = successor
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
                individual_id=individual,
            )
        )

    estimator = MCEIRLEstimator(
        MCEIRLConfig(
            compute_se=False,
            optimizer="L-BFGS-B",
            outer_max_iter=100,
            outer_tol=1e-7,
        )
    )
    result = estimator.estimate(
        Panel(trajectories),
        reward,
        problem,
        transitions,
        initial_params=jnp.zeros(2),
        terminal_states=terminal,
    )

    assert result.converged, (
        result.convergence_message,
        result.metadata["feature_difference"],
        np.asarray(result.parameters),
    )
    assert result.metadata["feature_difference"] < 0.03
    assert np.asarray(result.metadata["time_policy"]).shape == (horizon, 4, 2)
    np.testing.assert_allclose(
        np.asarray(result.metadata["time_policy"])[:, 2, 1],
        0.0,
    )


def test_short_terminal_routes_use_fixed_horizon_feature_scale() -> None:
    transitions = DeterministicTransitions(
        next_state=np.array([[1, 2], [2, 2], [2, -1]]),
        valid_action=np.array([[True, True], [True, False], [True, False]]),
    )
    terminal = jnp.array([False, False, True])
    problem = DDCProblem(
        num_states=3,
        num_actions=2,
        discount_factor=1.0,
        num_periods=2,
    )
    features = np.zeros((3, 2, 1), dtype=float)
    features[0, 1, 0] = 1.0
    reward = ActionDependentReward(
        feature_matrix=jnp.asarray(features),
        parameter_names=["shortcut"],
    )
    panel = Panel(
        [
            Trajectory(
                states=jnp.array([0]),
                actions=jnp.array([1]),
                next_states=jnp.array([2]),
            ),
            Trajectory(
                states=jnp.array([0, 1]),
                actions=jnp.array([0, 0]),
                next_states=jnp.array([1, 2]),
            ),
        ]
    )
    estimator = MCEIRLEstimator(
        MCEIRLConfig(
            compute_se=True,
            se_method="asymptotic",
            optimizer="L-BFGS-B",
        )
    )

    result = estimator.estimate(
        panel,
        reward,
        problem,
        transitions,
        terminal_states=terminal,
        initial_dist=jnp.array([1.0, 0.0, 0.0]),
    )

    assert result.converged
    assert np.isfinite(np.asarray(result.standard_errors)).all()
    assert np.asarray(result.standard_errors)[0] > 0
    assert result.metadata["feature_difference"] < 1e-6


def test_optimizer_success_does_not_override_feature_gate() -> None:
    transitions = DeterministicTransitions(next_state=np.array([[1, 1], [1, 1]]))
    problem = DDCProblem(
        num_states=2,
        num_actions=2,
        discount_factor=1.0,
        num_periods=1,
    )
    features = np.zeros((2, 2, 1), dtype=float)
    features[0, 1, 0] = 1.0
    reward = ActionDependentReward(
        feature_matrix=jnp.asarray(features),
        parameter_names=["action_one"],
    )
    panel = Panel(
        [
            Trajectory(
                states=jnp.array([0]),
                actions=jnp.array([individual % 2]),
                next_states=jnp.array([1]),
            )
            for individual in range(11)
        ]
    )
    estimator = MCEIRLEstimator(
        MCEIRLConfig(
            compute_se=False,
            optimizer="L-BFGS-B",
            feature_tol=1e-20,
        )
    )

    result = estimator.estimate(
        panel,
        reward,
        problem,
        transitions,
        terminal_states=jnp.array([False, True]),
    )

    assert result.metadata["optimizer_converged"]
    assert not result.converged
    assert result.metadata["termination_reason"] == "feature_residual"
