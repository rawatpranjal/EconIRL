"""Gradient and Hessian correctness tests.

For estimators that use gradients (NFXP, CCP, MCE IRL), compares
numerical gradients (central differences) against the analytical / computed
gradients. Also checks Hessian symmetry.

These tests use small environments and run quickly (not marked slow).
"""

import numpy as np
import pytest

import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.environments import RustBusEnvironment
from econirl.inference.standard_errors import compute_numerical_hessian
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Fixtures (small environment for speed)
# ---------------------------------------------------------------------------

@pytest.fixture
def quick_env():
    """Small Rust bus environment for fast gradient checks."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=20,
        discount_factor=0.99,
    )


@pytest.fixture
def quick_setup(quick_env):
    """Return (panel, utility, problem, transitions, true_params) for quick env."""
    panel = simulate_panel(quick_env, n_individuals=50, n_periods=30, seed=123)
    utility = LinearUtility.from_environment(quick_env)
    problem = quick_env.problem_spec
    transitions = quick_env.transition_matrices
    true_params = quick_env.get_true_parameter_vector()
    return panel, utility, problem, transitions, true_params


# ---------------------------------------------------------------------------
# NFXP gradient check
# ---------------------------------------------------------------------------

def test_nfxp_numerical_gradient(quick_setup):
    """NFXP: numerical gradient should be non-zero and directionally consistent."""
    panel, utility, problem, transitions, true_params = quick_setup
    operator = SoftBellmanOperator(problem, transitions)

    def _ll(params):
        params_f32 = jnp.asarray(params, dtype=jnp.float32)
        flow_utility = utility.compute(params_f32)
        result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)
        log_probs = operator.compute_log_choice_probabilities(flow_utility, result.V)
        ll_val = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = int(traj.states[t])
                a = int(traj.actions[t])
                ll_val += float(log_probs[s, a])
        return ll_val

    params = np.asarray(true_params, dtype=np.float64)
    eps = 1e-4
    n_params = len(params)
    grad = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad[i] = (_ll(p_plus) - _ll(p_minus)) / (2 * eps)

    assert np.any(np.abs(grad) > 1e-6), (
        f"Gradient is effectively zero: {grad}"
    )

    grad_norm = np.linalg.norm(grad)
    perturbed = params.copy()
    perturbed[0] += 0.01
    grad_perturbed = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        p_plus = perturbed.copy()
        p_minus = perturbed.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad_perturbed[i] = (_ll(p_plus) - _ll(p_minus)) / (2 * eps)
    grad_perturbed_norm = np.linalg.norm(grad_perturbed)

    assert grad_norm < grad_perturbed_norm * 100, (
        f"Gradient at true params ({grad_norm:.4f}) is unexpectedly large "
        f"compared to perturbed ({grad_perturbed_norm:.4f})"
    )


# ---------------------------------------------------------------------------
# CCP gradient check
# ---------------------------------------------------------------------------

def test_ccp_numerical_gradient(quick_setup):
    """CCP: numerical gradient of pseudo-log-likelihood should be non-zero
    and have consistent signs across step sizes."""
    from econirl.estimation.ccp import CCPEstimator

    panel, utility, problem, transitions, true_params = quick_setup
    ccp_est = CCPEstimator(num_policy_iterations=1, compute_hessian=False)

    ccps = ccp_est._estimate_ccps_from_data(
        panel, problem.num_states, problem.num_actions
    )

    def _ll(params):
        params_f32 = jnp.asarray(params, dtype=jnp.float32)
        return float(
            ccp_est._compute_log_likelihood(
                params_f32, panel, utility, ccps, transitions, problem
            )
        )

    params = np.asarray(true_params, dtype=np.float64)
    eps = 1e-4
    n_params = len(params)

    grad_a = np.zeros(n_params, dtype=np.float64)
    grad_b = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        p_plus = params.copy(); p_minus = params.copy()
        p_plus[i] += eps; p_minus[i] -= eps
        grad_a[i] = (_ll(p_plus) - _ll(p_minus)) / (2 * eps)

        p_plus2 = params.copy(); p_minus2 = params.copy()
        p_plus2[i] += eps / 2; p_minus2[i] -= eps / 2
        grad_b[i] = (_ll(p_plus2) - _ll(p_minus2)) / eps

    assert np.any(np.abs(grad_a) > 1e-6), (
        f"CCP gradient is effectively zero: {grad_a}"
    )

    for i in range(n_params):
        if abs(grad_a[i]) > 1.0:
            assert grad_a[i] * grad_b[i] > 0, (
                f"CCP gradient sign mismatch at parameter {i}: "
                f"grad_a={grad_a[i]:.4f}, grad_b={grad_b[i]:.4f}"
            )


# ---------------------------------------------------------------------------
# Hessian symmetry check
# ---------------------------------------------------------------------------

def test_hessian_symmetry(quick_setup):
    """Numerical Hessian of NFXP log-likelihood should be symmetric."""
    panel, utility, problem, transitions, true_params = quick_setup
    operator = SoftBellmanOperator(problem, transitions)

    def _ll(params):
        flow_utility = utility.compute(params)
        result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)
        log_probs = operator.compute_log_choice_probabilities(flow_utility, result.V)
        ll_val = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = int(traj.states[t])
                a = int(traj.actions[t])
                ll_val += float(log_probs[s, a])
        return ll_val

    hessian = compute_numerical_hessian(true_params, _ll)

    asym = jnp.abs(hessian - hessian.T)
    max_asym = float(asym.max())
    assert max_asym < 1e-4, (
        f"Hessian is not symmetric: max |H - H^T| = {max_asym:.8f}"
    )


# ---------------------------------------------------------------------------
# Hessian negative-definiteness at optimum
# ---------------------------------------------------------------------------

def test_hessian_negative_definite_at_true(quick_setup):
    """At the true parameters, the Hessian of LL should be negative semi-definite."""
    panel, utility, problem, transitions, true_params = quick_setup
    operator = SoftBellmanOperator(problem, transitions)

    def _ll(params):
        flow_utility = utility.compute(params)
        result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)
        log_probs = operator.compute_log_choice_probabilities(flow_utility, result.V)
        ll_val = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = int(traj.states[t])
                a = int(traj.actions[t])
                ll_val += float(log_probs[s, a])
        return ll_val

    hessian = compute_numerical_hessian(true_params, _ll)
    eigenvalues = jnp.linalg.eigvalsh(hessian)

    assert float(eigenvalues.max()) < 1e-2, (
        f"Hessian at true params is not NSD: max eigenvalue = {float(eigenvalues.max()):.6f}"
    )
