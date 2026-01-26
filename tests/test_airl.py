"""Tests for AIRL estimator."""

import pytest
import torch

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
from econirl.preferences.action_reward import ActionDependentReward


@pytest.fixture
def simple_problem():
    """Create a simple 3-state, 2-action problem."""
    return DDCProblem(
        num_states=3,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
    )


@pytest.fixture
def simple_transitions(simple_problem):
    """Create simple deterministic transitions."""
    n_states = simple_problem.num_states
    n_actions = simple_problem.num_actions
    transitions = torch.zeros(n_actions, n_states, n_states)

    for s in range(n_states):
        transitions[0, s, s] = 1.0
    for s in range(n_states):
        next_s = (s + 1) % n_states
        transitions[1, s, next_s] = 1.0

    return transitions


@pytest.fixture
def simple_reward_fn(simple_problem):
    """Create simple action-dependent reward function."""
    n_states = simple_problem.num_states
    n_actions = simple_problem.num_actions
    features = torch.zeros(n_states, n_actions, 2)
    features[:, 0, 0] = 1.0
    features[:, 1, 1] = 1.0
    return ActionDependentReward(
        feature_matrix=features,
        parameter_names=["action_0_reward", "action_1_reward"],
    )


@pytest.fixture
def expert_panel():
    """Create expert demonstrations favoring action 0."""
    trajectories = []
    for i in range(20):
        states = torch.tensor([0, 0, 0, 0, 0])
        actions = torch.tensor([0, 0, 0, 0, 0])
        next_states = torch.tensor([0, 0, 0, 0, 0])
        trajectories.append(Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=i,
        ))
    return Panel(trajectories=trajectories)


class TestAIRLEstimator:
    """Tests for AIRL estimator."""

    def test_airl_init(self):
        """AIRL should initialize with default config."""
        estimator = AIRLEstimator()
        assert estimator.name == "AIRL (Fu et al. 2018)"

    def test_airl_init_with_config(self):
        """AIRL should accept custom config."""
        config = AIRLConfig(max_rounds=50, reward_lr=0.05)
        estimator = AIRLEstimator(config=config)
        assert estimator.config.max_rounds == 50
        assert estimator.config.reward_lr == 0.05

    def test_airl_estimate_returns_result(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """AIRL estimate should return EstimationSummary."""
        config = AIRLConfig(max_rounds=10, verbose=False)
        estimator = AIRLEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert result.parameters is not None
        assert result.policy is not None
        assert result.policy.shape == (3, 2)

    def test_airl_policy_is_valid_distribution(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """AIRL policy should sum to 1 for each state."""
        config = AIRLConfig(max_rounds=10, verbose=False)
        estimator = AIRLEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        policy_sum = result.policy.sum(dim=1)
        assert torch.allclose(policy_sum, torch.ones(3), atol=1e-5)

    def test_airl_recovers_reward_structure(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """AIRL should recover relative reward structure."""
        config = AIRLConfig(max_rounds=50, verbose=False)
        estimator = AIRLEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Expert prefers action 0, so learned policy should prefer action 0
        assert result.policy[0, 0] > result.policy[0, 1]
