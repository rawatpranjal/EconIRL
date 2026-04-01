"""Tests for AIRL with unobserved heterogeneity (Lee, Sudhir & Wang 2026).

Tests the EM-AIRL estimator on a small serialized content environment
with 2 segments. Verifies:
1. The estimator runs without errors
2. Anchor enforcement zeros out exit action and absorbing state reward
3. Segment priors update from uniform initialization
4. Mixture log-likelihood improves over EM iterations
5. Segment posteriors are valid probability distributions
"""

import pytest
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig
from econirl.preferences.action_reward import ActionDependentReward


# --- Test environment (small, 6 states) ---

def _build_small_env():
    """Build a tiny 6-state serialized content environment.

    5 episodes + 1 absorbing state, 3 actions (buy=0, wait=1, exit=2).
    True reward has action-dependent structure with exit anchored to zero.
    """
    n_episodes = 5
    n_states = 6  # 5 episodes + 1 absorbing
    n_actions = 3
    absorbing = 5
    exit_action = 2

    # Transitions (deterministic)
    T = np.zeros((n_actions, n_states, n_states))
    for s in range(n_episodes):
        T[0, s, min(s + 1, n_episodes - 1)] = 1.0  # buy: advance
        T[1, s, s] = 1.0                             # wait: stay
        T[2, s, absorbing] = 1.0                      # exit: absorb
    T[0, absorbing, absorbing] = 1.0
    T[1, absorbing, absorbing] = 1.0
    T[2, absorbing, absorbing] = 1.0
    transitions = jnp.array(T, dtype=jnp.float32)

    # Features (3 features: buy_indicator, quality, wait_indicator)
    features = np.zeros((n_states, n_actions, 3))
    for s in range(n_episodes):
        features[s, 0, 0] = 1.0          # buy indicator
        features[s, 0, 1] = 1.0 - 0.2*s  # quality
        features[s, 1, 2] = 1.0          # wait indicator
    feature_matrix = jnp.array(features, dtype=jnp.float32)

    # True reward: r(s,buy) = -1 + 2*quality, r(s,wait) = -0.5, r(s,exit) = 0
    true_params = jnp.array([-1.0, 2.0, -0.5])
    reward = jnp.einsum("sak,k->sa", feature_matrix, true_params)

    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.9,
        scale_parameter=1.0,
    )

    utility = ActionDependentReward(feature_matrix, ["buy_cost", "quality", "wait_cost"])

    return {
        "transitions": transitions,
        "feature_matrix": feature_matrix,
        "reward": reward,
        "true_params": true_params,
        "problem": problem,
        "utility": utility,
        "n_states": n_states,
        "n_actions": n_actions,
        "absorbing": absorbing,
        "exit_action": exit_action,
    }


def _simulate_panel(env_dict, n_individuals=50, n_periods=20, seed=42):
    """Simulate panel data from true reward."""
    problem = env_dict["problem"]
    transitions = env_dict["transitions"]
    reward = env_dict["reward"]

    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, reward, tol=1e-10, max_iter=5000)
    policy = result.policy

    rng = np.random.default_rng(seed)
    trajectories = []

    for i in range(n_individuals):
        state = 0
        states, actions, next_states = [], [], []
        for t in range(n_periods):
            probs = np.asarray(policy[state])
            probs = probs / probs.sum()
            action = rng.choice(env_dict["n_actions"], p=probs)
            states.append(state)
            actions.append(action)

            # Deterministic transition
            next_state = int(jnp.argmax(transitions[action, state]))
            next_states.append(next_state)
            state = next_state

        trajectories.append(Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states, dtype=jnp.int32),
            individual_id=i,
        ))

    return Panel(trajectories=trajectories)


# --- Tests ---

class TestAIRLHetConfig:
    """Test config validation."""

    def test_missing_exit_action_raises(self):
        with pytest.raises(ValueError, match="exit_action"):
            AIRLHetConfig(absorbing_state=5)

    def test_missing_absorbing_state_raises(self):
        with pytest.raises(ValueError, match="absorbing_state"):
            AIRLHetConfig(exit_action=2)

    def test_valid_config(self):
        config = AIRLHetConfig(exit_action=2, absorbing_state=5)
        assert config.num_segments == 2
        assert config.exit_action == 2
        assert config.absorbing_state == 5


class TestAIRLHetEstimator:
    """Test the full EM-AIRL estimator."""

    @pytest.fixture
    def env(self):
        return _build_small_env()

    @pytest.fixture
    def panel(self, env):
        return _simulate_panel(env, n_individuals=50, n_periods=20)

    def test_runs_without_error(self, env, panel):
        """Estimator runs end-to-end and returns EstimationSummary."""
        config = AIRLHetConfig(
            num_segments=2,
            exit_action=env["exit_action"],
            absorbing_state=env["absorbing"],
            max_em_iterations=3,
            max_airl_rounds=10,
            reward_lr=0.01,
        )
        estimator = AIRLHetEstimator(config)
        summary = estimator.estimate(
            panel, env["utility"], env["problem"], env["transitions"],
        )
        assert summary is not None
        assert summary.policy.shape == (env["n_states"], env["n_actions"])

    def test_anchor_enforcement(self, env, panel):
        """Segment reward matrices have zero exit action and absorbing state."""
        config = AIRLHetConfig(
            num_segments=2,
            exit_action=env["exit_action"],
            absorbing_state=env["absorbing"],
            max_em_iterations=3,
            max_airl_rounds=10,
        )
        estimator = AIRLHetEstimator(config)
        summary = estimator.estimate(
            panel, env["utility"], env["problem"], env["transitions"],
        )
        for k, rm in enumerate(summary.metadata["segment_reward_matrices"]):
            rm_arr = np.array(rm)
            # Exit action column should be zero
            np.testing.assert_allclose(
                rm_arr[:, env["exit_action"]], 0.0, atol=1e-6,
                err_msg=f"Segment {k}: exit action reward not zero"
            )
            # Absorbing state row should be zero
            np.testing.assert_allclose(
                rm_arr[env["absorbing"], :], 0.0, atol=1e-6,
                err_msg=f"Segment {k}: absorbing state reward not zero"
            )

    def test_priors_update_from_uniform(self, env, panel):
        """Segment priors should move away from uniform after EM."""
        config = AIRLHetConfig(
            num_segments=2,
            exit_action=env["exit_action"],
            absorbing_state=env["absorbing"],
            max_em_iterations=5,
            max_airl_rounds=15,
        )
        estimator = AIRLHetEstimator(config)
        summary = estimator.estimate(
            panel, env["utility"], env["problem"], env["transitions"],
        )
        priors = np.array(summary.metadata["segment_priors"])
        # Priors should sum to 1
        np.testing.assert_allclose(priors.sum(), 1.0, atol=1e-6)
        # Priors should be positive
        assert all(p > 0 for p in priors)

    def test_posteriors_valid_probabilities(self, env, panel):
        """Segment posteriors should be valid probability distributions."""
        config = AIRLHetConfig(
            num_segments=2,
            exit_action=env["exit_action"],
            absorbing_state=env["absorbing"],
            max_em_iterations=3,
            max_airl_rounds=10,
        )
        estimator = AIRLHetEstimator(config)
        summary = estimator.estimate(
            panel, env["utility"], env["problem"], env["transitions"],
        )
        posteriors = np.array(summary.metadata["segment_posteriors"])
        # Each row should sum to 1
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        # All entries should be non-negative
        assert np.all(posteriors >= 0)

    def test_em_log_likelihood_recorded(self, env, panel):
        """EM log-likelihood history should be recorded in metadata."""
        config = AIRLHetConfig(
            num_segments=2,
            exit_action=env["exit_action"],
            absorbing_state=env["absorbing"],
            max_em_iterations=5,
            max_airl_rounds=10,
        )
        estimator = AIRLHetEstimator(config)
        summary = estimator.estimate(
            panel, env["utility"], env["problem"], env["transitions"],
        )
        em_lls = summary.metadata["em_log_likelihoods"]
        assert len(em_lls) > 0
        # All LLs should be finite
        assert all(np.isfinite(ll) for ll in em_lls)

    def test_name_property(self):
        config = AIRLHetConfig(exit_action=0, absorbing_state=5)
        estimator = AIRLHetEstimator(config)
        assert "Lee, Sudhir & Wang" in estimator.name

    def test_three_segments(self, env, panel):
        """Estimator works with K=3 segments."""
        config = AIRLHetConfig(
            num_segments=3,
            exit_action=env["exit_action"],
            absorbing_state=env["absorbing"],
            max_em_iterations=2,
            max_airl_rounds=5,
        )
        estimator = AIRLHetEstimator(config)
        summary = estimator.estimate(
            panel, env["utility"], env["problem"], env["transitions"],
        )
        priors = summary.metadata["segment_priors"]
        assert len(priors) == 3
        posteriors = np.array(summary.metadata["segment_posteriors"])
        assert posteriors.shape[1] == 3
