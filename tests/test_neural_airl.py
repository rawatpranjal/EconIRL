"""Load-bearing tests for the standalone NeuralAIRL capability."""

from __future__ import annotations

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory


def _nonlinear_case(seed: int = 912) -> tuple[np.ndarray, np.ndarray, Panel]:
    """Create demonstrations whose state reward is nonlinear in one input."""
    n_states = 9
    n_actions = 2
    x = np.linspace(-1.0, 1.0, n_states, dtype=np.float64)
    true_reward = 1.5 * np.cos(np.pi * x) - 0.35 * x

    transitions = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
    for state in range(n_states):
        transitions[0, state, (state + 1) % n_states] = 0.9
        transitions[0, state, state] = 0.1
        transitions[1, state, (state - 1) % n_states] = 0.9
        transitions[1, state, state] = 0.1

    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.9,
        scale_parameter=1.0,
    )
    truth = value_iteration(
        SoftBellmanOperator(problem, jnp.asarray(transitions)),
        jnp.repeat(jnp.asarray(true_reward)[:, None], n_actions, axis=1),
        tol=1e-10,
        max_iter=5_000,
    )

    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []
    for individual in range(180):
        state = int(rng.integers(n_states))
        states: list[int] = []
        actions: list[int] = []
        next_states: list[int] = []
        for _ in range(30):
            action = int(rng.choice(n_actions, p=np.asarray(truth.policy[state])))
            following = int(rng.choice(n_states, p=transitions[action, state]))
            states.append(state)
            actions.append(action)
            next_states.append(following)
            state = following
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
                individual_id=individual,
            )
        )
    return x[:, None], transitions, Panel(trajectories=trajectories)


@pytest.fixture(scope="module")
def nonlinear_fits():
    """Fit the frozen linear AIRL and standalone neural model once."""
    from econirl import AIRL, NeuralAIRL

    state_inputs, transitions, panel = _nonlinear_case()
    common = dict(
        n_states=state_inputs.shape[0],
        n_actions=2,
        discount=0.9,
        feature_matrix=state_inputs,
        max_rounds=180,
        min_rounds=100,
        discriminator_steps=3,
        policy_step_size=0.1,
        compute_se=False,
        seed=913,
    )
    linear = AIRL(**common).fit(panel, transitions=transitions)
    neural = NeuralAIRL(
        **common,
        reward_hidden_dim=32,
        reward_num_layers=2,
    ).fit(panel, transitions=transitions)
    return linear, neural


def test_neural_airl_is_independent_and_rejects_heterogeneity():
    """NeuralAIRL must be a real nonlinear estimator with a closed v1 scope."""
    from econirl import AIRL, NeuralAIRL

    assert NeuralAIRL is not AIRL, "NeuralAIRL still resolves to frozen tabular AIRL"
    model = NeuralAIRL(n_states=3, n_actions=2, feature_matrix=np.arange(3)[:, None])
    with pytest.raises(NotImplementedError, match="AIRL2"):
        model.fit(None, transitions=np.zeros((2, 3, 3)), context=np.zeros(1))


@pytest.mark.slow
def test_neural_airl_beats_linear_airl_on_nonlinear_behavior(nonlinear_fits):
    """The new capability must matter on behavior the linear basis cannot express."""
    linear, neural = nonlinear_fits
    _, transitions, _ = _nonlinear_case()
    state_inputs = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    true_reward = 1.5 * np.cos(np.pi * state_inputs) - 0.35 * state_inputs
    problem = DDCProblem(9, 2, 0.9, 1.0)
    truth = value_iteration(
        SoftBellmanOperator(problem, jnp.asarray(transitions)),
        jnp.repeat(jnp.asarray(true_reward)[:, None], 2, axis=1),
        tol=1e-10,
        max_iter=5_000,
    )

    linear_tv = 0.5 * np.mean(np.abs(linear.policy_ - truth.policy).sum(axis=1))
    neural_tv = 0.5 * np.mean(np.abs(neural.policy_ - truth.policy).sum(axis=1))

    assert neural_tv <= 0.75 * linear_tv, (
        f"NeuralAIRL did not materially improve held-out behavior: "
        f"neural_tv={neural_tv:.6f}, linear_tv={linear_tv:.6f}"
    )


def test_neural_airl_serialization_preserves_reward_and_policy(nonlinear_fits):
    """A fitted clean-wheel model must preserve its two decision surfaces."""
    _, neural = nonlinear_fits
    states = np.arange(9)
    reward_before = neural.predict_reward(states)
    policy_before = neural.predict_proba(states)

    restored = pickle.loads(pickle.dumps(neural))

    np.testing.assert_allclose(restored.predict_reward(states), reward_before, atol=1e-6)
    np.testing.assert_allclose(restored.predict_proba(states), policy_before, atol=1e-6)
