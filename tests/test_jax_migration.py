"""Rigorous tests for the JAX migration.

Tests numerical correctness, gradient accuracy, solver consistency,
implicit differentiation, and cross-estimator agreement. These tests
go beyond smoke testing to verify that the JAX implementation produces
machine-precision results matching the mathematical specifications.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from econirl.core.bellman import (
    SoftBellmanOperator, bellman_operator_fn, compute_flow_utility,
)
from econirl.core.occupancy import (
    compute_state_visitation, compute_state_action_visitation,
)
from econirl.core.solvers import (
    hybrid_iteration, optimistix_solve, policy_iteration, value_iteration,
)
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.inference.standard_errors import (
    compute_analytical_hessian, compute_numerical_hessian,
)
from econirl.preferences.linear import LinearUtility


# ============================================================================
# Bellman operator correctness
# ============================================================================


class TestBellmanOperator:
    """Test the soft Bellman operator against analytical solutions."""

    def test_two_state_analytical(self):
        """Verify Bellman step on 2-state, 2-action MDP with known answer."""
        problem = DDCProblem(num_states=2, num_actions=2, discount_factor=0.5, scale_parameter=1.0)
        T = jnp.array([
            [[0.8, 0.2], [0.3, 0.7]],  # action 0
            [[1.0, 0.0], [1.0, 0.0]],  # action 1 (reset)
        ])
        U = jnp.array([[1.0, 0.0], [0.5, -0.5]])  # (S, A)
        V = jnp.zeros(2)

        op = SoftBellmanOperator(problem, T)
        result = op.apply(U, V)

        # Manual: Q[s,a] = U[s,a] + 0.5 * sum_s' T[a,s,s'] * V[s']
        # With V=0, Q = U. V_new = logsumexp(Q, axis=1)
        Q_expected = U
        V_expected = jax.scipy.special.logsumexp(Q_expected, axis=1)
        npt.assert_allclose(np.asarray(result.Q), np.asarray(Q_expected), atol=1e-14)
        npt.assert_allclose(np.asarray(result.V), np.asarray(V_expected), atol=1e-14)

    def test_policy_is_softmax_of_Q(self):
        """Verify pi(a|s) = softmax(Q(s,a)/sigma)."""
        problem = DDCProblem(num_states=5, num_actions=3, discount_factor=0.9, scale_parameter=2.0)
        T = jnp.ones((3, 5, 5)) / 5  # uniform transitions
        U = jax.random.normal(jax.random.key(0), (5, 3))
        V = jax.random.normal(jax.random.key(1), (5,))

        op = SoftBellmanOperator(problem, T)
        result = op.apply(U, V)

        policy_expected = jax.nn.softmax(result.Q / 2.0, axis=1)
        npt.assert_allclose(np.asarray(result.policy), np.asarray(policy_expected), atol=1e-14)

    def test_contraction_property(self):
        """Verify T is a contraction: ||T(V1) - T(V2)|| <= beta * ||V1 - V2||."""
        problem = DDCProblem(num_states=10, num_actions=2, discount_factor=0.95)
        T = jnp.ones((2, 10, 10)) / 10
        U = jnp.zeros((10, 2))

        op = SoftBellmanOperator(problem, T)
        V1 = jax.random.normal(jax.random.key(0), (10,))
        V2 = jax.random.normal(jax.random.key(1), (10,))

        TV1 = op.apply(U, V1).V
        TV2 = op.apply(U, V2).V

        contraction = float(jnp.max(jnp.abs(TV1 - TV2)))
        original = float(jnp.max(jnp.abs(V1 - V2)))
        assert contraction <= 0.95 * original + 1e-10

    def test_transition_row_sums(self):
        """Verify environment transition matrices are valid (rows sum to 1)."""
        env = RustBusEnvironment()
        T = env.transition_matrices
        row_sums = T.sum(axis=2)
        npt.assert_allclose(np.asarray(row_sums), np.ones(row_sums.shape), atol=1e-10)


# ============================================================================
# Solver correctness and cross-solver consistency
# ============================================================================


class TestSolverConsistency:
    """Verify all solvers converge to the same fixed point."""

    @pytest.fixture
    def rust_setup(self):
        env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.99,
                                 operating_cost=0.01, replacement_cost=2.0)
        T = env.transition_matrices
        problem = env.problem_spec
        features = env.feature_matrix
        params = jnp.array([0.01, 2.0])
        utility = compute_flow_utility(params, features)
        operator = SoftBellmanOperator(problem, T)
        return operator, utility, problem

    def test_vi_hybrid_agree(self, rust_setup):
        """Value iteration and hybrid iteration converge to same V."""
        op, U, problem = rust_setup
        vi = value_iteration(op, U, tol=1e-10, max_iter=100000)
        hy = hybrid_iteration(op, U, tol=1e-10, max_iter=100000, switch_tol=1e-3)
        npt.assert_allclose(np.asarray(vi.V), np.asarray(hy.V), atol=1e-6)

    def test_pi_hybrid_agree(self, rust_setup):
        """Policy iteration and hybrid iteration converge to same V."""
        op, U, problem = rust_setup
        pi = policy_iteration(op, U, tol=1e-10, max_iter=100)
        hy = hybrid_iteration(op, U, tol=1e-10, max_iter=100000, switch_tol=1e-3)
        npt.assert_allclose(np.asarray(pi.V), np.asarray(hy.V), atol=1e-6)

    def test_optimistix_hybrid_agree(self, rust_setup):
        """Optimistix fixed-point and hybrid iteration converge to same V."""
        op, U, problem = rust_setup
        hy = hybrid_iteration(op, U, tol=1e-10, max_iter=100000, switch_tol=1e-3)
        V_optx = optimistix_solve(problem, op.transitions, U, tol=1e-10, max_steps=100000)
        npt.assert_allclose(np.asarray(V_optx), np.asarray(hy.V), atol=1e-4)

    def test_fixed_point_residual(self, rust_setup):
        """Converged V satisfies V = T(V) to tolerance."""
        op, U, _ = rust_setup
        result = hybrid_iteration(op, U, tol=1e-10, max_iter=200000, switch_tol=1e-3)
        TV = op.apply(U, result.V).V
        residual = float(jnp.max(jnp.abs(TV - result.V)))
        assert residual < 1e-8, f"Fixed-point residual {residual} too large"

    def test_policy_sums_to_one(self, rust_setup):
        """Policy rows sum to 1."""
        op, U, _ = rust_setup
        result = hybrid_iteration(op, U, tol=1e-10, max_iter=200000, switch_tol=1e-3)
        row_sums = result.policy.sum(axis=1)
        npt.assert_allclose(np.asarray(row_sums), np.ones(row_sums.shape[0]), atol=1e-10)


# ============================================================================
# Implicit differentiation through the fixed point
# ============================================================================


class TestImplicitDifferentiation:
    """Verify gradients through the Bellman fixed point via optimistix."""

    @pytest.fixture
    def setup(self):
        env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.95,
                                 operating_cost=0.01, replacement_cost=2.0)
        T = env.transition_matrices
        features = env.feature_matrix
        problem = env.problem_spec
        params = jnp.array([0.01, 2.0])
        return T, features, problem, params

    def test_grad_matches_finite_diff(self, setup):
        """Implicit diff gradient matches finite differences to 4 digits."""
        T, features, problem, params = setup

        def total_V(theta):
            u = compute_flow_utility(theta, features)
            V = optimistix_solve(problem, T, u, tol=1e-10, max_steps=100000)
            return V.sum()

        # Analytical gradient via implicit diff
        grad_ad = jax.grad(total_V)(params)

        # Finite differences
        eps = 1e-5
        grad_fd = np.zeros(2)
        for i in range(2):
            e = np.zeros(2)
            e[i] = eps
            grad_fd[i] = (float(total_V(params + e)) - float(total_V(params - e))) / (2 * eps)

        npt.assert_allclose(np.asarray(grad_ad), grad_fd, rtol=1e-3,
                            err_msg="Implicit diff gradient does not match finite differences")

    def test_grad_log_likelihood_matches_fd(self, setup):
        """Gradient of log-likelihood matches finite differences."""
        T, features, problem, params = setup

        # Simulate small panel
        operator = SoftBellmanOperator(problem, T)
        u = compute_flow_utility(params, features)
        sol = hybrid_iteration(operator, u, tol=1e-10, max_iter=100000, switch_tol=1e-3)

        rng = np.random.RandomState(42)
        obs_s, obs_a = [], []
        for _ in range(200):
            state = 0
            for _ in range(10):
                action = rng.choice(2, p=np.asarray(sol.policy[state]))
                obs_s.append(state)
                obs_a.append(action)
                state = rng.choice(problem.num_states, p=np.asarray(T[action, state]))
        obs_s = jnp.array(obs_s, dtype=jnp.int32)
        obs_a = jnp.array(obs_a, dtype=jnp.int32)

        def log_likelihood(theta):
            u = compute_flow_utility(theta, features)
            V = optimistix_solve(problem, T, u, tol=1e-10, max_steps=100000)
            EV = jnp.einsum("ast,t->as", T, V)
            Q = u + problem.discount_factor * EV.T
            lp = jax.nn.log_softmax(Q / problem.scale_parameter, axis=1)
            return lp[obs_s, obs_a].sum()

        grad_ad = jax.grad(log_likelihood)(params)

        eps = 1e-5
        grad_fd = np.zeros(2)
        for i in range(2):
            e = np.zeros(2)
            e[i] = eps
            grad_fd[i] = (float(log_likelihood(params + e)) - float(log_likelihood(params - e))) / (2 * eps)

        npt.assert_allclose(np.asarray(grad_ad), grad_fd, rtol=1e-2,
                            err_msg="LL gradient does not match finite differences")

    def test_hessian_is_symmetric(self, setup):
        """Hessian of log-likelihood is symmetric."""
        T, features, problem, params = setup

        rng = np.random.RandomState(0)
        obs_s = jnp.array(rng.randint(0, problem.num_states, 500), dtype=jnp.int32)
        obs_a = jnp.array(rng.randint(0, 2, 500), dtype=jnp.int32)

        def log_likelihood(theta):
            u = compute_flow_utility(theta, features)
            V = optimistix_solve(problem, T, u, tol=1e-10, max_steps=100000)
            EV = jnp.einsum("ast,t->as", T, V)
            Q = u + problem.discount_factor * EV.T
            lp = jax.nn.log_softmax(Q / problem.scale_parameter, axis=1)
            return lp[obs_s, obs_a].sum()

        H = jax.hessian(log_likelihood)(params)
        npt.assert_allclose(np.asarray(H), np.asarray(H.T), atol=1e-8,
                            err_msg="Hessian is not symmetric")

    def test_hessian_analytical_vs_numerical(self, setup):
        """Analytical Hessian (jax.hessian) matches numerical Hessian."""
        T, features, problem, params = setup

        rng = np.random.RandomState(0)
        obs_s = jnp.array(rng.randint(0, problem.num_states, 200), dtype=jnp.int32)
        obs_a = jnp.array(rng.randint(0, 2, 200), dtype=jnp.int32)

        def log_likelihood(theta):
            u = compute_flow_utility(theta, features)
            V = optimistix_solve(problem, T, u, tol=1e-10, max_steps=100000)
            EV = jnp.einsum("ast,t->as", T, V)
            Q = u + problem.discount_factor * EV.T
            lp = jax.nn.log_softmax(Q / problem.scale_parameter, axis=1)
            return lp[obs_s, obs_a].sum()

        H_analytical = compute_analytical_hessian(params, log_likelihood)
        H_numerical = compute_numerical_hessian(params, log_likelihood, eps=1e-4)

        npt.assert_allclose(np.asarray(H_analytical), np.asarray(H_numerical), rtol=0.05,
                            err_msg="Analytical Hessian diverges from numerical")


# ============================================================================
# Occupancy measures
# ============================================================================


class TestOccupancyMeasures:
    """Verify state visitation frequencies satisfy theoretical properties."""

    def test_visitation_sums_to_one(self):
        """Discounted state visitation sums to 1/(1-gamma) after normalization."""
        env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.95)
        T = env.transition_matrices
        problem = env.problem_spec
        U = env.compute_utility_matrix()
        op = SoftBellmanOperator(problem, T)
        result = hybrid_iteration(op, U, tol=1e-10, max_iter=100000, switch_tol=1e-3)

        D = compute_state_visitation(result.policy, T, problem)
        npt.assert_allclose(float(D.sum()), 1.0, atol=1e-6)

    def test_visitation_nonnegative(self):
        """State visitation is non-negative everywhere."""
        env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.95)
        T = env.transition_matrices
        problem = env.problem_spec
        U = env.compute_utility_matrix()
        op = SoftBellmanOperator(problem, T)
        result = hybrid_iteration(op, U, tol=1e-10, max_iter=100000, switch_tol=1e-3)

        D = compute_state_visitation(result.policy, T, problem)
        assert bool(jnp.all(D >= -1e-10)), f"Negative visitation: min={float(D.min())}"

    def test_state_action_consistent(self):
        """D_sa[s,a] = D[s] * pi(a|s)."""
        env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.95)
        T = env.transition_matrices
        problem = env.problem_spec
        U = env.compute_utility_matrix()
        op = SoftBellmanOperator(problem, T)
        result = hybrid_iteration(op, U, tol=1e-10, max_iter=100000, switch_tol=1e-3)

        D = compute_state_visitation(result.policy, T, problem)
        D_sa = compute_state_action_visitation(result.policy, T, problem)

        D_sa_expected = D[:, None] * result.policy
        npt.assert_allclose(np.asarray(D_sa), np.asarray(D_sa_expected), atol=1e-8)


# ============================================================================
# NFXP estimator
# ============================================================================


class TestNFXPEstimator:
    """Test NFXP estimation pipeline end-to-end."""

    def test_analytical_score_gradient_direction(self):
        """Score at non-optimal params points toward true params."""
        env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.99,
                                 operating_cost=0.01, replacement_cost=2.0)
        T = env.transition_matrices
        problem = env.problem_spec
        utility = LinearUtility.from_environment(env)
        operator = SoftBellmanOperator(problem, T)

        # Solve at true params
        true_params = jnp.array([0.01, 2.0])
        true_U = jnp.array(utility.compute(true_params), dtype=jnp.float64)
        true_sol = hybrid_iteration(operator, true_U, tol=1e-10, max_iter=100000, switch_tol=1e-3)

        # Simulate data at true params
        rng = np.random.RandomState(42)
        trajs = []
        for i in range(200):
            state = 0
            s, a, ns = [], [], []
            for _ in range(50):
                act = rng.choice(2, p=np.asarray(true_sol.policy[state]))
                nxt = rng.choice(problem.num_states, p=np.asarray(T[act, state]))
                s.append(state); a.append(act); ns.append(nxt); state = nxt
            trajs.append(Trajectory(
                states=jnp.array(s, dtype=jnp.int32),
                actions=jnp.array(a, dtype=jnp.int32),
                next_states=jnp.array(ns, dtype=jnp.int32), individual_id=i))
        panel = Panel(trajectories=trajs)

        from econirl.estimation.nfxp import NFXPEstimator
        est = NFXPEstimator(inner_solver="hybrid")

        # Score at true params should be near zero (MLE condition)
        scores_true, ll_true = est._compute_analytical_score(
            true_params, panel, utility, operator, true_sol.V, true_sol.policy
        )
        grad_true = np.asarray(scores_true.sum(axis=0))
        grad_norm_true = np.linalg.norm(grad_true)

        # Score at wrong params should be larger
        wrong_params = jnp.array([0.005, 1.0])
        wrong_U = jnp.array(utility.compute(wrong_params), dtype=jnp.float64)
        wrong_sol = hybrid_iteration(operator, wrong_U, tol=1e-10, max_iter=100000, switch_tol=1e-3)
        scores_wrong, ll_wrong = est._compute_analytical_score(
            wrong_params, panel, utility, operator, wrong_sol.V, wrong_sol.policy
        )
        grad_wrong = np.asarray(scores_wrong.sum(axis=0))
        grad_norm_wrong = np.linalg.norm(grad_wrong)

        assert grad_norm_wrong > grad_norm_true, \
            f"Gradient at wrong params ({grad_norm_wrong:.2f}) should be larger than at true ({grad_norm_true:.2f})"
        assert ll_true > ll_wrong, \
            f"LL at true params ({ll_true:.2f}) should exceed LL at wrong params ({ll_wrong:.2f})"


# ============================================================================
# Data types and panel operations
# ============================================================================


class TestDataTypes:
    """Verify core data types work correctly with JAX arrays."""

    def test_trajectory_creation(self):
        """Trajectory created from JAX arrays."""
        t = Trajectory(
            states=jnp.array([0, 1, 2], dtype=jnp.int32),
            actions=jnp.array([0, 1, 0], dtype=jnp.int32),
            next_states=jnp.array([1, 2, 0], dtype=jnp.int32),
        )
        assert len(t) == 3
        assert t.states.dtype == jnp.int32

    def test_panel_from_numpy(self):
        """Panel.from_numpy groups by individual correctly."""
        states = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        actions = np.array([0, 0, 1, 0, 1], dtype=np.int64)
        next_states = np.array([1, 2, 0, 1, 2], dtype=np.int64)
        ids = np.array([0, 0, 0, 1, 1], dtype=np.int64)

        panel = Panel.from_numpy(states, actions, next_states, ids)
        assert panel.num_individuals == 2
        assert panel.num_observations == 5
        assert len(panel[0]) == 3
        assert len(panel[1]) == 2

    def test_sufficient_stats_consistency(self):
        """Sufficient stats are internally consistent."""
        env = RustBusEnvironment(num_mileage_bins=10, discount_factor=0.95)
        T = env.transition_matrices
        problem = env.problem_spec
        operator = SoftBellmanOperator(problem, T)
        U = env.compute_utility_matrix()
        sol = hybrid_iteration(operator, U, tol=1e-10, max_iter=100000, switch_tol=1e-3)

        rng = np.random.RandomState(0)
        trajs = []
        for i in range(50):
            state = 0; s, a, ns = [], [], []
            for _ in range(30):
                act = rng.choice(2, p=np.asarray(sol.policy[state]))
                nxt = rng.choice(10, p=np.asarray(T[act, state]))
                s.append(state); a.append(act); ns.append(nxt); state = nxt
            trajs.append(Trajectory(
                states=jnp.array(s, dtype=jnp.int32),
                actions=jnp.array(a, dtype=jnp.int32),
                next_states=jnp.array(ns, dtype=jnp.int32), individual_id=i))
        panel = Panel(trajectories=trajs)

        ss = panel.sufficient_stats(10, 2)
        # CCPs sum to 1 across actions
        npt.assert_allclose(np.asarray(ss.empirical_ccps.sum(axis=1)),
                            np.ones(10), atol=1e-6)
        # Transitions sum to 1 across next states
        row_sums = np.asarray(ss.transitions.sum(axis=2))
        npt.assert_allclose(row_sums, np.ones(row_sums.shape), atol=1e-6)
        # Initial dist sums to 1
        npt.assert_allclose(float(ss.initial_distribution.sum()), 1.0, atol=1e-6)


# ============================================================================
# RewardSpec and LinearUtility
# ============================================================================


class TestRewardSpec:
    """Test reward specification correctness."""

    def test_linear_utility_einsum(self):
        """U(s,a) = theta . phi(s,a) via einsum."""
        features = jnp.array([
            [[1.0, 0.0], [0.0, -1.0]],
            [[2.0, 0.0], [0.0, -1.0]],
        ])
        utility = LinearUtility(features, parameter_names=["a", "b"])
        theta = jnp.array([0.5, 3.0])
        U = utility.compute(theta)

        # U[0,0] = 0.5*1 + 3*0 = 0.5
        # U[0,1] = 0.5*0 + 3*(-1) = -3.0
        # U[1,0] = 0.5*2 + 3*0 = 1.0
        # U[1,1] = 0.5*0 + 3*(-1) = -3.0
        expected = jnp.array([[0.5, -3.0], [1.0, -3.0]])
        npt.assert_allclose(np.asarray(U), np.asarray(expected), atol=1e-14)

    def test_gradient_is_feature_matrix(self):
        """For linear utility, dU/dtheta = phi (constant)."""
        features = jax.random.normal(jax.random.key(0), (5, 3, 4))
        utility = LinearUtility(features, parameter_names=["a", "b", "c", "d"])
        theta = jax.random.normal(jax.random.key(1), (4,))
        grad = utility.compute_gradient(theta)
        npt.assert_allclose(np.asarray(grad), np.asarray(features), atol=1e-14)


# ============================================================================
# Environment correctness
# ============================================================================


class TestEnvironments:
    """Test environment implementations."""

    def test_rust_bus_replace_resets_to_zero(self):
        """Replace action transitions to low mileage states."""
        env = RustBusEnvironment(num_mileage_bins=20)
        T = env.transition_matrices
        # Replace action (1): from any state, should go near state 0
        for s in range(20):
            assert float(T[1, s, :3].sum()) > 0.99, \
                f"Replace from state {s} doesn't reset: P(s'<3) = {float(T[1,s,:3].sum())}"

    def test_rust_bus_keep_advances_mileage(self):
        """Keep action advances mileage by 0, 1, or 2."""
        env = RustBusEnvironment(num_mileage_bins=20)
        T = env.transition_matrices
        # Keep action (0) from state 5: should only go to 5, 6, or 7
        for s in range(15):
            valid = float(T[0, s, s:s+3].sum())
            assert valid > 0.99, \
                f"Keep from state {s}: P(valid) = {valid}"

    def test_feature_matrix_shape(self):
        """Feature matrix has correct shape (S, A, K)."""
        env = RustBusEnvironment(num_mileage_bins=90)
        assert env.feature_matrix.shape == (90, 2, 2)


# ============================================================================
# Float64 precision
# ============================================================================


class TestPrecision:
    """Verify float64 is used throughout for structural estimation accuracy."""

    def test_bellman_float64(self):
        """Bellman operator outputs float64."""
        problem = DDCProblem(num_states=5, num_actions=2, discount_factor=0.99)
        T = jnp.ones((2, 5, 5), dtype=jnp.float64) / 5
        U = jnp.zeros((5, 2), dtype=jnp.float64)
        op = SoftBellmanOperator(problem, T)
        result = op.apply(U, jnp.zeros(5, dtype=jnp.float64))
        assert result.V.dtype == jnp.float64

    def test_solver_float64(self):
        """Solvers produce float64 value functions."""
        problem = DDCProblem(num_states=5, num_actions=2, discount_factor=0.99)
        T = jnp.ones((2, 5, 5), dtype=jnp.float64) / 5
        U = jnp.zeros((5, 2), dtype=jnp.float64)
        op = SoftBellmanOperator(problem, T)
        result = value_iteration(op, U, tol=1e-10, max_iter=10000)
        assert result.V.dtype == jnp.float64

    def test_high_beta_convergence(self):
        """Hybrid solver converges at beta=0.9999 (requires float64)."""
        env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
        T = env.transition_matrices
        problem = env.problem_spec
        U = env.compute_utility_matrix()
        op = SoftBellmanOperator(problem, T)
        result = hybrid_iteration(op, U, tol=1e-10, max_iter=200000, switch_tol=1e-3)
        assert result.converged, f"Failed to converge at beta=0.9999: error={result.final_error}"
        assert result.final_error < 1e-8
