"""Tests for the Neural MPEC estimator.

Tests cover:
- NeuralMPECConfig defaults
- The estimator runs through the uniform .estimate interface and returns
  well-shaped objects
- The reference action's reward is anchored to exactly zero
- The known-P Bellman residual is small at convergence
- Reward / value recovery on an anchored linear cell (slow, @pytest.mark.slow)
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from econirl.core.types import DDCProblem
from econirl.environments import ArrayMDP
from econirl.estimation import (
    NeuralMPEC,
    NeuralMPECConfig,
    NeuralMPECEstimator,
)
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


def _anchored_linear_env(num_states: int = 12, seed: int = 0) -> ArrayMDP:
    """A small 3-action ergodic MDP with a zero-reward reference action (action 2).

    Action 0 drifts the state up, action 1 down, action 2 is a local walk with
    identically-zero reward (its features are zero), the location-normalization
    anchor. Reward is linear in the state feature.
    """
    S = num_states
    x = np.arange(S, dtype=np.float64) / (S - 1)
    phi = np.zeros((S, 3, 4), dtype=np.float64)
    phi[:, 0, 0] = 1.0
    phi[:, 0, 1] = x
    phi[:, 1, 2] = 1.0
    phi[:, 1, 3] = x
    theta = np.array([0.4, -0.6, 0.2, -0.4], dtype=np.float64)
    T = np.zeros((3, S, S), dtype=np.float64)
    for s in range(S):
        for off, p in [(0, 0.2), (1, 0.6), (2, 0.2)]:
            T[0, s, min(s + off, S - 1)] += p
        for off, p in [(0, 0.2), (1, 0.6), (2, 0.2)]:
            T[1, s, max(s - off, 0)] += p
        for off, p in [(-1, 0.25), (0, 0.5), (1, 0.25)]:
            T[2, s, min(max(s + off, 0), S - 1)] += p
    return ArrayMDP(
        T, phi, theta, discount_factor=0.95, scale_parameter=1.0,
        parameter_names=["a0_const", "a0_slope", "a1_const", "a1_slope"], seed=seed,
    )


class TestNeuralMPECConfig:
    def test_default_values(self):
        cfg = NeuralMPECConfig()
        assert cfg.width == 32
        assert cfg.depth == 2
        assert cfg.bellman_penalty_weight == 1.0
        assert cfg.learning_rate == 5e-3
        assert cfg.max_epochs == 4000
        assert cfg.reference_action == -1
        assert cfg.collocation == "all"

    def test_alias(self):
        assert NeuralMPEC is NeuralMPECEstimator

    def test_name(self):
        assert NeuralMPECEstimator().name == "Neural MPEC"


class TestNeuralMPECSmall:
    @pytest.fixture
    def setup(self):
        env = _anchored_linear_env(num_states=10, seed=1)
        panel = simulate_panel(env, n_individuals=60, n_periods=20, seed=7)
        util = LinearUtility(feature_matrix=env.feature_matrix,
                             parameter_names=env.parameter_names)
        return env, panel, util

    def test_runs_and_shapes(self, setup):
        env, panel, util = setup
        est = NeuralMPECEstimator(
            NeuralMPECConfig(width=16, depth=1, max_epochs=200, reference_action=2)
        )
        summ = est.estimate(panel, util, env.problem_spec, env.transition_matrices)
        S, A = env.num_states, env.num_actions
        assert np.asarray(summ.policy).shape == (S, A)
        assert np.asarray(summ.value_function).shape == (S,)
        assert np.isfinite(summ.log_likelihood)
        assert np.asarray(summ.parameters).shape == (util.num_parameters,)

    def test_policy_is_valid_distribution(self, setup):
        env, panel, util = setup
        est = NeuralMPECEstimator(
            NeuralMPECConfig(width=16, depth=1, max_epochs=200, reference_action=2)
        )
        summ = est.estimate(panel, util, env.problem_spec, env.transition_matrices)
        pol = np.asarray(summ.policy)
        assert np.allclose(pol.sum(axis=1), 1.0, atol=1e-4)
        assert (pol >= 0).all()

    def test_reference_action_anchored_to_zero(self, setup):
        env, panel, util = setup
        est = NeuralMPECEstimator(
            NeuralMPECConfig(width=16, depth=1, max_epochs=200, reference_action=2)
        )
        summ = est.estimate(panel, util, env.problem_spec, env.transition_matrices)
        reward = np.asarray(summ.metadata["reward_table"])
        # Action 2 is the anchor: its reward is exactly zero everywhere.
        assert np.abs(reward[:, 2]).max() == 0.0

    def test_bellman_residual_reported(self, setup):
        env, panel, util = setup
        est = NeuralMPECEstimator(
            NeuralMPECConfig(width=16, depth=1, max_epochs=500, reference_action=2,
                             bellman_penalty_weight=10.0)
        )
        summ = est.estimate(panel, util, env.problem_spec, env.transition_matrices)
        # The penalty keeps the value network near its own Bellman image.
        assert summ.metadata["max_bellman_residual"] < 0.5


@pytest.mark.slow
class TestNeuralMPECRecovery:
    def test_reward_and_value_recovery(self):
        """On a larger panel the neural reward and value track the truth."""
        from econirl.core.bellman import SoftBellmanOperator
        from econirl.core.solvers import value_iteration

        env = _anchored_linear_env(num_states=20, seed=0)
        panel = simulate_panel(env, n_individuals=400, n_periods=40, seed=12345)
        util = LinearUtility(feature_matrix=env.feature_matrix,
                             parameter_names=env.parameter_names)

        est = NeuralMPECEstimator(
            NeuralMPECConfig(width=32, depth=2, max_epochs=4000, reference_action=2)
        )
        summ = est.estimate(panel, util, env.problem_spec, env.transition_matrices)

        op = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
        oracle_V = np.asarray(value_iteration(op, env.compute_utility_matrix()).V)
        reward = np.asarray(summ.metadata["reward_table"])
        true_R = np.asarray(env.true_reward_matrix)

        def rmse(a, b):
            a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
            return float(np.sqrt(np.mean((a - b) ** 2)))

        # Reward RMSE over the estimated actions {0, 1} (reference excluded).
        reward_rmse = rmse(reward[:, :2], true_R[:, :2])
        value_rmse = rmse(summ.value_function, oracle_V)
        assert reward_rmse < 0.30, f"reward RMSE too high: {reward_rmse}"
        assert value_rmse < 0.60, f"value RMSE too high: {value_rmse}"
        assert summ.metadata["max_bellman_residual"] < 0.05
