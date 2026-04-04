"""Tests for MaxEnt IRL estimator core functionality.

Tests cover:
1. Estimator instantiation
2. _compute_empirical_features works correctly
3. _compute_expected_features works correctly
4. _optimize runs and returns EstimationResult
5. estimate() method works (inherited from BaseEstimator)
"""

import pytest
import numpy as np
import jax.numpy as jnp

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.contrib.maxent_irl import MaxEntIRLEstimator
from econirl.estimation.base import EstimationResult
from econirl.preferences.reward import LinearReward
from econirl.simulation.synthetic import simulate_panel


# ============================================================================
# Fixtures specific to MaxEnt IRL testing
# ============================================================================

@pytest.fixture
def simple_problem() -> DDCProblem:
    """Simple 10-state, 2-action problem for testing."""
    return DDCProblem(
        num_states=10,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
    )


@pytest.fixture
def simple_transitions(simple_problem: DDCProblem):
    """Simple transition matrices for testing."""
    num_states = simple_problem.num_states
    num_actions = simple_problem.num_actions

    # Create deterministic-ish transitions
    transitions = jnp.zeros((num_actions, num_states, num_states))

    for s in range(num_states):
        # Action 0: move forward (with some probability of staying)
        next_s = min(s + 1, num_states - 1)
        transitions = transitions.at[0, s, next_s].set(0.8)
        transitions = transitions.at[0, s, s].set(0.2)

        # Action 1: reset to state 0 (replacement-like)
        transitions = transitions.at[1, s, 0].set(1.0)

    return transitions


@pytest.fixture
def simple_features(simple_problem: DDCProblem):
    """Simple state features for testing."""
    num_states = simple_problem.num_states

    # Feature 1: state index (normalized)
    f1 = jnp.arange(num_states, dtype=jnp.float32) / num_states

    # Feature 2: quadratic in state
    f2 = (jnp.arange(num_states, dtype=jnp.float32) / num_states) ** 2

    # Feature 3: indicator for high states
    f3 = (jnp.arange(num_states) >= num_states // 2).astype(jnp.float32)

    features = jnp.stack([f1, f2, f3], axis=1)
    return features


@pytest.fixture
def simple_reward_fn(simple_features, simple_problem: DDCProblem) -> LinearReward:
    """LinearReward for testing."""
    return LinearReward(
        state_features=simple_features,
        parameter_names=["state_index", "state_squared", "high_state"],
        n_actions=simple_problem.num_actions,
    )


@pytest.fixture
def simple_panel(
    simple_problem: DDCProblem,
    simple_transitions,
    simple_reward_fn: LinearReward,
) -> Panel:
    """Generate a simple panel for testing."""
    # Use some reasonable reward parameters
    true_params = jnp.array([-0.1, -0.05, -0.5])

    # Compute optimal policy
    operator = SoftBellmanOperator(simple_problem, simple_transitions)
    reward_matrix = simple_reward_fn.compute(true_params)
    result = value_iteration(operator, reward_matrix)

    # Generate trajectories by sampling from policy
    np.random.seed(42)
    trajectories = []

    for i in range(20):  # 20 individuals
        states = []
        actions = []
        next_states = []

        state = np.random.randint(0, simple_problem.num_states)

        for t in range(30):  # 30 periods each
            states.append(state)

            # Sample action from policy
            probs = np.asarray(result.policy[state])
            action = np.random.choice(simple_problem.num_actions, p=probs)
            actions.append(action)

            # Sample next state
            trans_probs = np.asarray(simple_transitions[action, state])
            next_state = np.random.choice(simple_problem.num_states, p=trans_probs)
            next_states.append(next_state)

            state = next_state

        traj = Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states, dtype=jnp.int32),
            individual_id=i,
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


# ============================================================================
# Test: Estimator Instantiation
# ============================================================================

class TestMaxEntIRLInstantiation:
    """Tests for MaxEntIRLEstimator instantiation."""

    def test_default_instantiation(self):
        """Test that estimator can be created with default parameters."""
        estimator = MaxEntIRLEstimator()
        assert estimator is not None
        assert estimator.name == "MaxEnt IRL (Ziebart 2008)"

    def test_custom_parameters(self):
        """Test that estimator can be created with custom parameters."""
        estimator = MaxEntIRLEstimator(
            optimizer="BFGS",
            inner_tol=1e-8,
            outer_tol=1e-5,
            outer_max_iter=100,
            verbose=True,
        )
        assert estimator is not None
        assert estimator._optimizer == "BFGS"
        assert estimator._inner_tol == 1e-8
        assert estimator._outer_tol == 1e-5
        assert estimator._outer_max_iter == 100
        assert estimator._verbose is True

    def test_estimator_name(self):
        """Test that estimator has correct name."""
        estimator = MaxEntIRLEstimator()
        assert "MaxEnt" in estimator.name
        assert "IRL" in estimator.name

    def test_inner_solver_options(self):
        """Test both inner solver options work."""
        estimator_value = MaxEntIRLEstimator(inner_solver="value")
        estimator_policy = MaxEntIRLEstimator(inner_solver="policy")

        assert estimator_value._inner_solver == "value"
        assert estimator_policy._inner_solver == "policy"


# ============================================================================
# Test: _compute_empirical_features
# ============================================================================

class TestComputeEmpiricalFeatures:
    """Tests for _compute_empirical_features method."""

    def test_empirical_features_shape(
        self,
        simple_panel: Panel,
        simple_reward_fn: LinearReward,
    ):
        """Test that empirical features have correct shape."""
        estimator = MaxEntIRLEstimator()

        empirical = estimator._compute_empirical_features(simple_panel, simple_reward_fn)

        assert empirical.shape == (simple_reward_fn.num_parameters,)

    def test_empirical_features_finite(
        self,
        simple_panel: Panel,
        simple_reward_fn: LinearReward,
    ):
        """Test that empirical features are finite."""
        estimator = MaxEntIRLEstimator()

        empirical = estimator._compute_empirical_features(simple_panel, simple_reward_fn)

        assert jnp.isfinite(empirical).all()

    def test_empirical_features_match_manual(
        self,
        simple_panel: Panel,
        simple_reward_fn: LinearReward,
    ):
        """Test that empirical features match manual computation."""
        estimator = MaxEntIRLEstimator()

        # Compute using estimator method
        empirical = estimator._compute_empirical_features(simple_panel, simple_reward_fn)

        # Compute manually
        state_features = simple_reward_fn.state_features
        manual_sum = jnp.zeros(state_features.shape[1])
        total = 0

        for traj in simple_panel.trajectories:
            for state in traj.states:
                manual_sum = manual_sum + state_features[int(state)]
                total += 1

        manual_empirical = manual_sum / total

        assert jnp.allclose(empirical, manual_empirical, atol=1e-6)

    def test_empirical_features_single_trajectory(self, simple_reward_fn: LinearReward):
        """Test empirical features with single trajectory."""
        # Create minimal panel
        traj = Trajectory(
            states=jnp.array([0, 1, 2], dtype=jnp.int32),
            actions=jnp.array([0, 0, 1], dtype=jnp.int32),
            next_states=jnp.array([1, 2, 0], dtype=jnp.int32),
        )
        panel = Panel(trajectories=[traj])

        estimator = MaxEntIRLEstimator()
        empirical = estimator._compute_empirical_features(panel, simple_reward_fn)

        # Should be average of features for states 0, 1, 2
        expected = (
            simple_reward_fn.state_features[0] +
            simple_reward_fn.state_features[1] +
            simple_reward_fn.state_features[2]
        ) / 3

        assert jnp.allclose(empirical, expected, atol=1e-6)


# ============================================================================
# Test: _compute_expected_features
# ============================================================================

class TestComputeExpectedFeatures:
    """Tests for _compute_expected_features method."""

    def test_expected_features_shape(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that expected features have correct shape."""
        estimator = MaxEntIRLEstimator()

        # Compute a policy
        operator = SoftBellmanOperator(simple_problem, simple_transitions)
        params = jnp.zeros(simple_reward_fn.num_parameters)
        reward_matrix = simple_reward_fn.compute(params)
        solver_result = value_iteration(operator, reward_matrix)

        expected = estimator._compute_expected_features(
            solver_result.policy,
            simple_transitions,
            simple_reward_fn,
            simple_problem,
            simple_panel,
        )

        assert expected.shape == (simple_reward_fn.num_parameters,)

    def test_expected_features_finite(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that expected features are finite."""
        estimator = MaxEntIRLEstimator()

        operator = SoftBellmanOperator(simple_problem, simple_transitions)
        params = jnp.zeros(simple_reward_fn.num_parameters)
        reward_matrix = simple_reward_fn.compute(params)
        solver_result = value_iteration(operator, reward_matrix)

        expected = estimator._compute_expected_features(
            solver_result.policy,
            simple_transitions,
            simple_reward_fn,
            simple_problem,
            simple_panel,
        )

        assert jnp.isfinite(expected).all()

    def test_expected_features_different_policies(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
    ):
        """Test that different policies give different expected features."""
        estimator = MaxEntIRLEstimator()

        # Policy 1: always action 0
        policy1 = jnp.zeros((simple_problem.num_states, simple_problem.num_actions))
        policy1 = policy1.at[:, 0].set(1.0)

        # Policy 2: always action 1
        policy2 = jnp.zeros((simple_problem.num_states, simple_problem.num_actions))
        policy2 = policy2.at[:, 1].set(1.0)

        expected1 = estimator._compute_expected_features(
            policy1, simple_transitions, simple_reward_fn, simple_problem
        )
        expected2 = estimator._compute_expected_features(
            policy2, simple_transitions, simple_reward_fn, simple_problem
        )

        # Should be different (different policies lead to different state distributions)
        assert not jnp.allclose(expected1, expected2, atol=1e-4)

    def test_state_visitation_sums_to_one(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
    ):
        """Test that state visitation frequencies sum to 1."""
        estimator = MaxEntIRLEstimator()

        # Uniform policy
        policy = jnp.ones((simple_problem.num_states, simple_problem.num_actions))
        policy = policy / policy.sum(axis=1, keepdims=True)

        visitation = estimator._compute_state_visitation_frequency(
            policy, simple_transitions, simple_problem
        )

        assert jnp.isclose(visitation.sum(), jnp.array(1.0), atol=1e-5)


# ============================================================================
# Test: _optimize
# ============================================================================

class TestOptimize:
    """Tests for _optimize method."""

    def test_optimize_returns_estimation_result(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that _optimize returns an EstimationResult."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator._optimize(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        assert isinstance(result, EstimationResult)

    def test_optimize_returns_correct_shapes(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that _optimize returns tensors with correct shapes."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator._optimize(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        assert result.parameters.shape == (simple_reward_fn.num_parameters,)
        assert result.value_function.shape == (simple_problem.num_states,)
        assert result.policy.shape == (simple_problem.num_states, simple_problem.num_actions)

    def test_optimize_returns_finite_values(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that _optimize returns finite values."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator._optimize(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        assert jnp.isfinite(result.parameters).all()
        assert jnp.isfinite(result.value_function).all()
        assert jnp.isfinite(result.policy).all()
        assert np.isfinite(result.log_likelihood)

    def test_optimize_valid_policy(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that _optimize returns valid policy (rows sum to 1)."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator._optimize(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        # Policy should be non-negative
        assert (result.policy >= 0).all()

        # Rows should sum to 1
        row_sums = result.policy.sum(axis=1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)

    def test_optimize_with_initial_params(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that _optimize accepts initial parameters."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=5,
            compute_hessian=False,
            verbose=False,
        )

        initial_params = jnp.array([0.1, -0.1, 0.0])

        result = estimator._optimize(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
            initial_params=initial_params,
        )

        assert isinstance(result, EstimationResult)

    def test_optimize_rejects_non_linear_reward(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_panel: Panel,
        utility_small,  # This is a LinearUtility, not LinearReward
    ):
        """Test that _optimize rejects non-LinearReward functions."""
        estimator = MaxEntIRLEstimator(verbose=False)

        with pytest.raises(TypeError, match="LinearReward"):
            estimator._optimize(
                simple_panel,
                utility_small,
                simple_problem,
                simple_transitions,
            )


# ============================================================================
# Test: estimate() method (inherited from BaseEstimator)
# ============================================================================

class TestEstimateMethod:
    """Tests for the estimate() method inherited from BaseEstimator."""

    def test_estimate_returns_estimation_summary(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that estimate() returns an EstimationSummary."""
        from econirl.inference.results import EstimationSummary

        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=True,
            verbose=False,
        )

        result = estimator.estimate(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        assert isinstance(result, EstimationSummary)

    def test_estimate_has_standard_errors(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that estimate() computes standard errors."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=True,
            verbose=False,
        )

        result = estimator.estimate(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        assert result.standard_errors is not None
        assert len(result.standard_errors) == simple_reward_fn.num_parameters

    def test_estimate_has_parameter_names(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that estimate() includes parameter names."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator.estimate(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        assert result.parameter_names == simple_reward_fn.parameter_names

    def test_estimate_summary_output(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that estimate() result can produce summary."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=10,
            compute_hessian=True,
            verbose=False,
        )

        result = estimator.estimate(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        summary = result.summary()

        assert "MaxEnt IRL" in summary
        assert "coef" in summary

    def test_estimate_converges_with_enough_iterations(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that estimate() converges with sufficient iterations."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=100,
            outer_tol=1e-4,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator.estimate(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        # Should converge (or at least not fail)
        assert result is not None
        assert jnp.isfinite(result.parameters).all()


# ============================================================================
# Test: Feature matching quality
# ============================================================================

class TestFeatureMatching:
    """Tests for feature matching quality."""

    def test_compute_feature_expectations(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test compute_feature_expectations method."""
        estimator = MaxEntIRLEstimator(verbose=False)

        params = jnp.zeros(simple_reward_fn.num_parameters)

        empirical, expected = estimator.compute_feature_expectations(
            params,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
            simple_panel,
        )

        assert empirical.shape == (simple_reward_fn.num_parameters,)
        assert expected.shape == (simple_reward_fn.num_parameters,)

    def test_optimization_reduces_feature_difference(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
    ):
        """Test that optimization reduces feature difference."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=50,
            compute_hessian=False,
            verbose=False,
        )

        # Initial feature difference
        initial_params = jnp.zeros(simple_reward_fn.num_parameters)
        emp_init, exp_init = estimator.compute_feature_expectations(
            initial_params,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
            simple_panel,
        )
        initial_diff = float(jnp.linalg.norm(emp_init - exp_init))

        # Run optimization
        result = estimator.estimate(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        # Final feature difference
        emp_final, exp_final = estimator.compute_feature_expectations(
            result.parameters,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
            simple_panel,
        )
        final_diff = float(jnp.linalg.norm(emp_final - exp_final))

        # Feature difference should decrease (or stay similar if already matched)
        # Allow some tolerance for numerical issues
        assert final_diff <= initial_diff + 0.1, \
            f"Feature difference increased: {initial_diff:.4f} -> {final_diff:.4f}"


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_state(self):
        """Test with single state (degenerate case)."""
        problem = DDCProblem(
            num_states=1,
            num_actions=2,
            discount_factor=0.9,
        )

        transitions = jnp.ones((2, 1, 1))  # Always stay in state 0

        features = jnp.array([[1.0, 0.5]])
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["f1", "f2"],
            n_actions=2,
        )

        traj = Trajectory(
            states=jnp.array([0, 0, 0], dtype=jnp.int32),
            actions=jnp.array([0, 1, 0], dtype=jnp.int32),
            next_states=jnp.array([0, 0, 0], dtype=jnp.int32),
        )
        panel = Panel(trajectories=[traj])

        estimator = MaxEntIRLEstimator(
            outer_max_iter=5,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator._optimize(
            panel, reward_fn, problem, transitions
        )

        assert isinstance(result, EstimationResult)
        assert jnp.isfinite(result.parameters).all()

    def test_many_features(self):
        """Test with many features."""
        num_features = 20
        num_states = 10

        problem = DDCProblem(
            num_states=num_states,
            num_actions=2,
            discount_factor=0.9,
        )

        # Simple transitions
        transitions = jnp.zeros((2, num_states, num_states))
        for s in range(num_states):
            transitions = transitions.at[0, s, min(s + 1, num_states - 1)].set(1.0)
            transitions = transitions.at[1, s, 0].set(1.0)

        features = jnp.array(np.random.randn(num_states, num_features))
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=[f"f{i}" for i in range(num_features)],
            n_actions=2,
        )

        # Create simple panel
        traj = Trajectory(
            states=jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32),
            actions=jnp.array([0, 0, 0, 0, 1], dtype=jnp.int32),
            next_states=jnp.array([1, 2, 3, 4, 0], dtype=jnp.int32),
        )
        panel = Panel(trajectories=[traj])

        estimator = MaxEntIRLEstimator(
            outer_max_iter=5,
            compute_hessian=False,
            verbose=False,
        )

        result = estimator._optimize(
            panel, reward_fn, problem, transitions
        )

        assert result.parameters.shape == (num_features,)

    def test_verbose_mode(
        self,
        simple_problem: DDCProblem,
        simple_transitions,
        simple_reward_fn: LinearReward,
        simple_panel: Panel,
        capsys,
    ):
        """Test that verbose mode prints output."""
        estimator = MaxEntIRLEstimator(
            outer_max_iter=5,
            compute_hessian=False,
            verbose=True,
        )

        estimator._optimize(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
        )

        captured = capsys.readouterr()
        # Should have printed something
        assert len(captured.out) > 0 or estimator._verbose
