"""Pytest fixtures for econirl tests.

This module provides reusable fixtures for testing:
- Rust bus environment with known parameters
- Simulated panel data
- Utility specifications
- Pre-computed solutions
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def rust_env() -> RustBusEnvironment:
    """Standard Rust bus environment with default parameters."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=90,
        discount_factor=0.9999,
        scale_parameter=1.0,
        seed=42,
    )


@pytest.fixture
def rust_env_small() -> RustBusEnvironment:
    """Small Rust bus environment for faster tests."""
    return RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        scale_parameter=1.0,
        seed=42,
    )


@pytest.fixture
def problem_spec(rust_env: RustBusEnvironment) -> DDCProblem:
    """DDCProblem from standard environment."""
    return rust_env.problem_spec


@pytest.fixture
def problem_spec_small(rust_env_small: RustBusEnvironment) -> DDCProblem:
    """DDCProblem from small environment."""
    return rust_env_small.problem_spec


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def small_panel(rust_env_small: RustBusEnvironment) -> Panel:
    """Small panel for quick tests (50 individuals, 50 periods)."""
    return _simulate_panel_jax(rust_env_small, n_individuals=50, n_periods=50, seed=123)


@pytest.fixture
def medium_panel(rust_env: RustBusEnvironment) -> Panel:
    """Medium panel for estimation tests (100 individuals, 100 periods)."""
    return _simulate_panel_jax(rust_env, n_individuals=100, n_periods=100, seed=456)


@pytest.fixture
def large_panel(rust_env: RustBusEnvironment) -> Panel:
    """Large panel for accuracy tests (500 individuals, 100 periods)."""
    return _simulate_panel_jax(rust_env, n_individuals=500, n_periods=100, seed=789)


@pytest.fixture
def single_trajectory() -> Trajectory:
    """Single trajectory for unit tests."""
    return Trajectory(
        states=jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32),
        actions=jnp.array([0, 0, 0, 0, 1, 0], dtype=jnp.int32),
        next_states=jnp.array([1, 2, 3, 4, 0, 1], dtype=jnp.int32),
        individual_id=0,
    )


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def utility(rust_env: RustBusEnvironment) -> LinearUtility:
    """LinearUtility from standard environment."""
    return LinearUtility.from_environment(rust_env)


@pytest.fixture
def utility_small(rust_env_small: RustBusEnvironment) -> LinearUtility:
    """LinearUtility from small environment."""
    return LinearUtility.from_environment(rust_env_small)


# ============================================================================
# Transition Fixtures
# ============================================================================

@pytest.fixture
def transitions(rust_env: RustBusEnvironment) -> jnp.ndarray:
    """Transition matrices from standard environment."""
    return rust_env.transition_matrices


@pytest.fixture
def transitions_small(rust_env_small: RustBusEnvironment) -> jnp.ndarray:
    """Transition matrices from small environment."""
    return rust_env_small.transition_matrices


# ============================================================================
# Solution Fixtures
# ============================================================================

@pytest.fixture
def bellman_operator(
    problem_spec: DDCProblem, transitions: jnp.ndarray
) -> SoftBellmanOperator:
    """Bellman operator for standard environment."""
    return SoftBellmanOperator(problem_spec, transitions)


@pytest.fixture
def optimal_policy(
    rust_env: RustBusEnvironment,
    bellman_operator: SoftBellmanOperator,
) -> jnp.ndarray:
    """Optimal policy from true parameters."""
    utility_matrix = rust_env.compute_utility_matrix()
    result = hybrid_iteration(bellman_operator, utility_matrix,
                              tol=1e-10, max_iter=200000, switch_tol=1e-3)
    return result.policy


@pytest.fixture
def optimal_value(
    rust_env: RustBusEnvironment,
    bellman_operator: SoftBellmanOperator,
) -> jnp.ndarray:
    """Optimal value function from true parameters."""
    utility_matrix = rust_env.compute_utility_matrix()
    result = hybrid_iteration(bellman_operator, utility_matrix,
                              tol=1e-10, max_iter=200000, switch_tol=1e-3)
    return result.V


# ============================================================================
# Parameter Fixtures
# ============================================================================

@pytest.fixture
def true_params(rust_env: RustBusEnvironment) -> jnp.ndarray:
    """True parameters as array."""
    return rust_env.get_true_parameter_vector()


@pytest.fixture
def true_params_small(rust_env_small: RustBusEnvironment) -> jnp.ndarray:
    """True parameters from small environment."""
    return rust_env_small.get_true_parameter_vector()


@pytest.fixture
def perturbed_params(true_params: jnp.ndarray) -> jnp.ndarray:
    """Parameters perturbed from true values (for testing)."""
    return true_params * 1.5


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def seed() -> int:
    """Standard seed for reproducibility."""
    return 42


@pytest.fixture
def tolerance() -> float:
    """Standard numerical tolerance."""
    return 1e-6


@pytest.fixture
def estimation_tolerance() -> float:
    """Tolerance for estimation accuracy (looser)."""
    return 0.5


# ============================================================================
# Helper Functions as Fixtures
# ============================================================================

@pytest.fixture
def assert_valid_policy():
    """Fixture providing a policy validation function."""
    def _assert_valid_policy(policy: jnp.ndarray):
        """Assert that policy is valid (rows sum to 1, all non-negative)."""
        assert bool(jnp.all(policy >= 0)), "Policy has negative probabilities"
        row_sums = policy.sum(axis=1)
        np.testing.assert_allclose(
            np.asarray(row_sums), np.ones(row_sums.shape[0]), atol=1e-6,
            err_msg="Policy rows don't sum to 1"
        )
    return _assert_valid_policy


@pytest.fixture
def assert_valid_value_function():
    """Fixture providing a value function validation function."""
    def _assert_valid_value(V: jnp.ndarray, problem: DDCProblem):
        """Assert that value function has correct shape and is finite."""
        assert V.shape == (problem.num_states,), \
            f"Value function has wrong shape: {V.shape}"
        assert bool(jnp.all(jnp.isfinite(V))), "Value function has non-finite values"
    return _assert_valid_value


# ============================================================================
# MCE IRL Test Fixtures
# ============================================================================

@pytest.fixture
def mce_irl_seed():
    """Fixture for reproducible random state in MCE IRL tests."""
    old_state = np.random.get_state()
    np.random.seed(42)
    yield 42
    np.random.set_state(old_state)


@pytest.fixture
def simple_problem():
    """Create a simple 10-state MDP with known structure."""
    n_states = 10
    problem = DDCProblem(
        num_states=n_states,
        num_actions=2,
        discount_factor=0.95,
    )

    # Deterministic transitions: keep -> next state, replace -> state 0
    transitions = jnp.zeros((2, n_states, n_states))
    for s in range(n_states):
        transitions = transitions.at[0, s, min(s + 1, n_states - 1)].set(1.0)
        transitions = transitions.at[1, s, 0].set(1.0)

    return problem, transitions


@pytest.fixture
def synthetic_panel(simple_problem, mce_irl_seed):
    """Generate synthetic data from a known policy."""
    problem, transitions = simple_problem
    n_states = problem.num_states

    trajectories = []
    for i in range(20):
        states, actions, next_states = [], [], []
        s = 0
        for t in range(50):
            states.append(s)
            p_replace = 0.05 + 0.15 * s / n_states
            a = 1 if np.random.random() < p_replace else 0
            actions.append(a)
            next_s = 0 if a == 1 else min(s + 1, n_states - 1)
            next_states.append(next_s)
            s = next_s

        traj = Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states, dtype=jnp.int32),
            individual_id=i,
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


# ============================================================================
# Helper: simulate panel from environment using JAX arrays
# ============================================================================

def _simulate_panel_jax(env, n_individuals, n_periods, seed):
    """Simulate panel data from environment."""
    problem = env.problem_spec
    T = env.transition_matrices
    operator = SoftBellmanOperator(problem, T)
    utility_matrix = env.compute_utility_matrix()
    result = hybrid_iteration(operator, utility_matrix,
                              tol=1e-10, max_iter=200000, switch_tol=1e-3)
    policy = result.policy

    rng = np.random.RandomState(seed)
    trajectories = []
    for i in range(n_individuals):
        state = 0
        states, actions, next_states_list = [], [], []
        for _ in range(n_periods):
            action = rng.choice(problem.num_actions, p=np.asarray(policy[state]))
            next_state = rng.choice(problem.num_states, p=np.asarray(T[action, state]))
            states.append(state)
            actions.append(action)
            next_states_list.append(next_state)
            state = next_state
        trajectories.append(Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states_list, dtype=jnp.int32),
            individual_id=i,
        ))

    return Panel(trajectories=trajectories)
