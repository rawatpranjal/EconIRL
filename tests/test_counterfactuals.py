"""Tests for the counterfactual analysis module.

Tests cover:
- CounterfactualType enum membership and values
- CounterfactualResult default type
- state_extrapolation with identity and shift mappings
- discount_factor_change direction
- welfare_decomposition additivity
- Unified dispatcher routing for all type combinations
- Invalid argument detection in the dispatcher
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from econirl.simulation.counterfactual import (
    CounterfactualType,
    CounterfactualResult,
    state_extrapolation,
    counterfactual_policy,
    counterfactual_transitions,
    discount_factor_change,
    welfare_decomposition,
    counterfactual,
    compute_stationary_distribution,
)
from econirl.core.types import DDCProblem
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.inference.results import EstimationSummary
from econirl.preferences.linear import LinearUtility


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_problem():
    """A 10-state, 2-action DDC problem with deterministic transitions.

    Action 0 (keep) moves to the next state. Action 1 (replace) resets
    to state 0. Discount factor is 0.95 for fast convergence.
    """
    n_states = 10
    n_actions = 2
    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.95,
        scale_parameter=1.0,
    )

    transitions = jnp.zeros((n_actions, n_states, n_states))
    for s in range(n_states):
        # Keep: deterministic move to next state (or stay at last)
        transitions = transitions.at[0, s, min(s + 1, n_states - 1)].set(1.0)
        # Replace: deterministic reset to state 0
        transitions = transitions.at[1, s, 0].set(1.0)

    return problem, transitions


@pytest.fixture
def small_utility(small_problem):
    """A simple linear utility for the 10-state problem.

    Operating cost increases linearly with state. Replacement incurs a
    fixed cost of 2.0.
    """
    problem, _ = small_problem
    n_states = problem.num_states
    n_actions = problem.num_actions

    # Feature matrix: (n_states, n_actions, n_features)
    # Feature 0: operating cost (state * 0.01 for keep, 0 for replace)
    # Feature 1: replacement indicator (0 for keep, 1 for replace)
    features = jnp.zeros((n_states, n_actions, 2))
    for s in range(n_states):
        features = features.at[s, 0, 0].set(-s * 0.1)  # keep cost
        features = features.at[s, 1, 1].set(-2.0)       # replace cost

    utility = LinearUtility(
        feature_matrix=features,
        parameter_names=["operating_cost", "replacement_cost"],
    )
    return utility


@pytest.fixture
def small_solution(small_problem, small_utility):
    """Solve the small problem and return a mock EstimationSummary."""
    problem, transitions = small_problem
    utility = small_utility

    true_params = jnp.array([1.0, 1.0])
    reward = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)
    sol = value_iteration(operator, reward)

    result = EstimationSummary(
        parameters=true_params,
        parameter_names=["operating_cost", "replacement_cost"],
        standard_errors=jnp.array([0.01, 0.01]),
        method="test",
        value_function=sol.V,
        policy=sol.policy,
    )
    return result


# ============================================================================
# Enum Tests
# ============================================================================


class TestCounterfactualTypeEnum:
    """Tests for CounterfactualType enum membership and values."""

    def test_has_four_members(self):
        """The enum should have exactly four members."""
        assert len(CounterfactualType) == 4

    def test_integer_values_1_through_4(self):
        """Each member should have an integer value from 1 to 4."""
        assert CounterfactualType.STATE_EXTRAPOLATION == 1
        assert CounterfactualType.ENVIRONMENT_CHANGE == 2
        assert CounterfactualType.REWARD_CHANGE == 3
        assert CounterfactualType.WELFARE_DECOMPOSITION == 4

    def test_is_intenum(self):
        """Members should be usable as integers."""
        assert CounterfactualType.STATE_EXTRAPOLATION + 1 == 2


class TestCounterfactualResultDefault:
    """Tests for CounterfactualResult defaults."""

    def test_default_type_is_reward_change(self):
        """The default counterfactual_type should be REWARD_CHANGE."""
        dummy = jnp.zeros((2, 2))
        dummy_v = jnp.zeros(2)
        result = CounterfactualResult(
            baseline_policy=dummy,
            counterfactual_policy=dummy,
            baseline_value=dummy_v,
            counterfactual_value=dummy_v,
            policy_change=dummy,
            value_change=dummy_v,
            welfare_change=0.0,
        )
        assert result.counterfactual_type == CounterfactualType.REWARD_CHANGE


# ============================================================================
# State Extrapolation (Type 1)
# ============================================================================


class TestStateExtrapolation:
    """Tests for the state_extrapolation function."""

    def test_identity_mapping_returns_zero_change(self, small_problem, small_solution):
        """An identity state mapping should produce zero policy change."""
        problem, transitions = small_problem
        identity_map = {s: s for s in range(problem.num_states)}

        cf = state_extrapolation(small_solution, identity_map, problem, transitions)

        np.testing.assert_allclose(
            np.asarray(cf.policy_change),
            np.zeros_like(np.asarray(cf.policy_change)),
            atol=1e-10,
            err_msg="Identity mapping should produce zero policy change",
        )
        assert cf.counterfactual_type == CounterfactualType.STATE_EXTRAPOLATION
        assert abs(cf.welfare_change) < 1e-10

    def test_shift_mapping_moves_policy(self, small_problem, small_solution):
        """Shifting states down by 5 should map state 7 policy to state 2."""
        problem, transitions = small_problem
        shift = 5
        mapping = {s: max(0, s - shift) for s in range(problem.num_states)}

        cf = state_extrapolation(small_solution, mapping, problem, transitions)

        # State 7 in the counterfactual should have the policy of state 2
        baseline_policy_at_2 = small_solution.policy[2]
        cf_policy_at_7 = cf.counterfactual_policy[7]

        np.testing.assert_allclose(
            np.asarray(cf_policy_at_7),
            np.asarray(baseline_policy_at_2),
            atol=1e-10,
            err_msg="State 7 counterfactual policy should equal baseline state 2 policy",
        )

    def test_array_mapping(self, small_problem, small_solution):
        """A numpy array mapping should work identically to a dict mapping."""
        problem, transitions = small_problem
        mapping_dict = {s: max(0, s - 3) for s in range(problem.num_states)}
        mapping_arr = jnp.array(
            [max(0, s - 3) for s in range(problem.num_states)], dtype=jnp.int32
        )

        cf_dict = state_extrapolation(small_solution, mapping_dict, problem, transitions)
        cf_arr = state_extrapolation(small_solution, mapping_arr, problem, transitions)

        np.testing.assert_allclose(
            np.asarray(cf_dict.counterfactual_policy),
            np.asarray(cf_arr.counterfactual_policy),
            atol=1e-10,
        )


# ============================================================================
# Discount Factor Change (Type 3)
# ============================================================================


class TestDiscountFactorChange:
    """Tests for the discount_factor_change function."""

    def test_lower_beta_changes_replacement_probability(
        self, small_problem, small_solution, small_utility
    ):
        """A lower discount factor should change the replacement probability.

        More myopic agents care less about the future cost of high
        mileage, so they should replace less often at intermediate
        states compared to patient agents.
        """
        problem, transitions = small_problem

        cf = discount_factor_change(
            result=small_solution,
            new_discount=0.5,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
        )

        # The policy should change somewhere
        max_abs_change = float(jnp.abs(cf.policy_change).max())
        assert max_abs_change > 0.01, (
            f"Expected noticeable policy change from beta=0.95 to beta=0.5, "
            f"but max absolute change is {max_abs_change}"
        )
        assert cf.counterfactual_type == CounterfactualType.REWARD_CHANGE


# ============================================================================
# Welfare Decomposition (Type 4)
# ============================================================================


class TestWelfareDecomposition:
    """Tests for the welfare_decomposition function."""

    def test_additivity(self, small_problem, small_solution, small_utility):
        """reward_channel + transition_channel + interaction should equal total."""
        problem, transitions = small_problem

        # Create a slightly different transition matrix
        new_transitions = transitions.copy()
        # Make action 0 slightly stochastic: 90% move forward, 10% stay
        n_states = problem.num_states
        for s in range(n_states - 1):
            new_transitions = new_transitions.at[0, s, min(s + 1, n_states - 1)].set(
                0.9
            )
            new_transitions = new_transitions.at[0, s, s].set(0.1)

        new_params = small_solution.parameters * 1.5

        decomp = welfare_decomposition(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            baseline_transitions=transitions,
            new_parameters=new_params,
            new_transitions=new_transitions,
        )

        total = decomp["total_welfare_change"]
        channels_sum = (
            decomp["reward_channel"]
            + decomp["transition_channel"]
            + decomp["interaction_effect"]
        )

        np.testing.assert_allclose(
            total,
            channels_sum,
            atol=1e-6,
            err_msg=(
                "Reward channel, transition channel, and interaction "
                "should sum to total welfare change"
            ),
        )

    def test_requires_at_least_one_change(
        self, small_problem, small_solution, small_utility
    ):
        """Calling with neither new_parameters nor new_transitions should raise."""
        problem, transitions = small_problem

        with pytest.raises(ValueError, match="At least one"):
            welfare_decomposition(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                baseline_transitions=transitions,
                new_parameters=None,
                new_transitions=None,
            )

    def test_params_only_has_zero_transition_channel(
        self, small_problem, small_solution, small_utility
    ):
        """When only parameters change, the transition channel should be zero."""
        problem, transitions = small_problem

        new_params = small_solution.parameters * 1.5

        decomp = welfare_decomposition(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            baseline_transitions=transitions,
            new_parameters=new_params,
            new_transitions=None,
        )

        np.testing.assert_allclose(
            decomp["transition_channel"],
            0.0,
            atol=1e-6,
            err_msg="Transition channel should be zero when only parameters change",
        )


# ============================================================================
# Unified Dispatcher
# ============================================================================


class TestDispatcher:
    """Tests for the counterfactual() unified dispatcher."""

    def test_type1_dispatch(self, small_problem, small_solution, small_utility):
        """Providing only state_mapping should dispatch to Type 1."""
        problem, transitions = small_problem
        mapping = {s: max(0, s - 5) for s in range(problem.num_states)}

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            state_mapping=mapping,
        )
        assert cf.counterfactual_type == CounterfactualType.STATE_EXTRAPOLATION

    def test_type2_dispatch(self, small_problem, small_solution, small_utility):
        """Providing only new_transitions should dispatch to Type 2."""
        problem, transitions = small_problem

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_transitions=transitions,
        )
        assert cf.counterfactual_type == CounterfactualType.ENVIRONMENT_CHANGE

    def test_type3_params_dispatch(
        self, small_problem, small_solution, small_utility
    ):
        """Providing only new_parameters should dispatch to Type 3."""
        problem, transitions = small_problem
        new_params = small_solution.parameters * 2.0

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_parameters=new_params,
        )
        assert cf.counterfactual_type == CounterfactualType.REWARD_CHANGE

    def test_type3_discount_dispatch(
        self, small_problem, small_solution, small_utility
    ):
        """Providing only new_discount should dispatch to Type 3."""
        problem, transitions = small_problem

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_discount=0.8,
        )
        assert cf.counterfactual_type == CounterfactualType.REWARD_CHANGE

    def test_invalid_combo_mapping_and_params(
        self, small_problem, small_solution, small_utility
    ):
        """state_mapping combined with new_parameters should raise ValueError."""
        problem, transitions = small_problem
        mapping = {0: 1}
        new_params = small_solution.parameters * 2.0

        with pytest.raises(ValueError, match="cannot be combined"):
            counterfactual(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                transitions=transitions,
                state_mapping=mapping,
                new_parameters=new_params,
            )

    def test_no_args_raises(self, small_problem, small_solution, small_utility):
        """Providing no counterfactual change should raise ValueError."""
        problem, transitions = small_problem

        with pytest.raises(ValueError, match="No counterfactual change specified"):
            counterfactual(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                transitions=transitions,
            )

    def test_combined_type2_and_type3(
        self, small_problem, small_solution, small_utility
    ):
        """Providing both new_parameters and new_transitions should work."""
        problem, transitions = small_problem
        new_params = small_solution.parameters * 1.5

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_parameters=new_params,
            new_transitions=transitions,
        )
        # Combined dispatch returns ENVIRONMENT_CHANGE type
        assert cf.counterfactual_type == CounterfactualType.ENVIRONMENT_CHANGE

    def test_discount_with_transitions_raises(
        self, small_problem, small_solution, small_utility
    ):
        """new_discount combined with new_transitions should raise ValueError."""
        problem, transitions = small_problem

        with pytest.raises(ValueError, match="cannot be combined"):
            counterfactual(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                transitions=transitions,
                new_discount=0.8,
                new_transitions=transitions,
            )


# ============================================================================
# Stationary Distribution
# ============================================================================


class TestStationaryDistribution:
    """Tests for the compute_stationary_distribution helper."""

    def test_sums_to_one(self, small_problem, small_solution):
        """The stationary distribution should sum to 1."""
        _, transitions = small_problem

        mu = compute_stationary_distribution(small_solution.policy, transitions)

        np.testing.assert_allclose(
            float(mu.sum()), 1.0, atol=1e-8,
            err_msg="Stationary distribution should sum to 1",
        )

    def test_all_nonnegative(self, small_problem, small_solution):
        """All entries in the stationary distribution should be nonnegative."""
        _, transitions = small_problem

        mu = compute_stationary_distribution(small_solution.policy, transitions)

        assert bool(jnp.all(mu >= 0)), "Stationary distribution has negative entries"

    def test_is_fixed_point(self, small_problem, small_solution):
        """The distribution should be a fixed point of the transition operator."""
        _, transitions = small_problem
        policy = small_solution.policy

        mu = compute_stationary_distribution(policy, transitions)

        # Policy-weighted transition: P^pi(s,s') = sum_a pi(a|s) P(s'|s,a)
        P_pi = jnp.einsum("sa,ast->st", policy, transitions)
        mu_next = P_pi.T @ mu
        mu_next = mu_next / mu_next.sum()

        np.testing.assert_allclose(
            np.asarray(mu),
            np.asarray(mu_next),
            atol=1e-8,
            err_msg="Stationary distribution should be a fixed point",
        )
