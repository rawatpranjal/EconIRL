"""Component tests for SEES against exact known-truth DGP objects."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from econirl.estimation.sees import SEESEstimator
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from validation.known_truth import (
    KnownTruthDGP,
    KnownTruthDGPConfig,
    build_known_truth_dgp,
    get_cell,
    solve_known_truth,
)


def _low_dim_dgp() -> KnownTruthDGP:
    return build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            num_regular_states=8,
            transition_noise=0.02,
            seed=801,
        )
    )


def _project_value(basis: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    alpha, *_ = np.linalg.lstsq(
        np.asarray(basis, dtype=np.float64),
        np.asarray(value, dtype=np.float64),
        rcond=None,
    )
    return jnp.asarray(alpha, dtype=jnp.float64)


def _project_state_action(basis: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    basis_np = np.asarray(basis, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    coeffs = []
    for action in range(target_np.shape[1]):
        alpha, *_ = np.linalg.lstsq(basis_np, target_np[:, action], rcond=None)
        coeffs.append(alpha)
    return jnp.asarray(np.stack(coeffs, axis=0), dtype=jnp.float64)


def _state_action_from_alpha(basis: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("sk,ak->sa", basis, alpha)


def _expected_value(transitions: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("ast,t->as", transitions.astype(jnp.float64), value).T


def _rust_bus_truth(num_states: int = 12):
    env = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=num_states,
        discount_factor=0.95,
    )
    utility = LinearUtility.from_environment(env)
    flow_u = jnp.einsum(
        "sak,k->sa",
        utility.feature_matrix,
        env.get_true_parameter_vector(),
    )
    solution = value_iteration(
        SoftBellmanOperator(env.problem_spec, env.transition_matrices),
        flow_u,
        tol=1e-11,
        max_iter=20_000,
    )
    assert solution.converged
    return env, solution


def _rust_oracle_alpha(
    solution_type: str,
    basis: jnp.ndarray,
    env: RustBusEnvironment,
    solution,
) -> tuple[jnp.ndarray, float]:
    if solution_type in {"value", "collocation"}:
        alpha = _project_value(basis, solution.V)
        fitted = basis @ alpha
        rmse = float(jnp.sqrt(jnp.mean((fitted - solution.V) ** 2)))
        return alpha, rmse

    if solution_type == "q":
        alpha = _project_state_action(basis, solution.Q)
        fitted = _state_action_from_alpha(basis, alpha)
        rmse = float(jnp.sqrt(jnp.mean((fitted - solution.Q) ** 2)))
        return alpha.reshape(-1), rmse

    if solution_type == "ev":
        continuation = _expected_value(env.transition_matrices, solution.V)
        alpha = _project_state_action(basis, continuation)
        fitted = _state_action_from_alpha(basis, alpha)
        rmse = float(jnp.sqrt(jnp.mean((fitted - continuation) ** 2)))
        return alpha.reshape(-1), rmse

    logits = jnp.log(jnp.clip(solution.policy, 1e-12, 1.0))
    logits = logits - logits.mean(axis=1, keepdims=True)
    alpha = _project_state_action(basis, logits)
    fitted = _state_action_from_alpha(basis, alpha)
    fitted = fitted - fitted.mean(axis=1, keepdims=True)
    rmse = float(jnp.sqrt(jnp.mean((fitted - logits) ** 2)))
    return alpha.reshape(-1), rmse


def _choice_values_from_basis(
    dgp: KnownTruthDGP,
    basis: jnp.ndarray,
    alpha: jnp.ndarray,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    expected_basis = jnp.einsum(
        "ast,tk->ask",
        dgp.transitions.astype(jnp.float64),
        basis.astype(jnp.float64),
    )
    flow_u = jnp.einsum(
        "sak,k->sa",
        dgp.feature_matrix.astype(jnp.float64),
        theta.astype(jnp.float64),
    )
    continuation = dgp.problem.discount_factor * jnp.einsum(
        "ask,k->sa",
        expected_basis,
        alpha.astype(jnp.float64),
    )
    return flow_u + continuation


def test_sees_low_dimensional_basis_represents_exact_value():
    dgp = _low_dim_dgp()
    solution = solve_known_truth(dgp)
    estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(dgp.problem.num_states, dgp.problem)
    alpha = _project_value(basis, solution.V)
    projected_value = basis @ alpha

    assert estimator._last_basis_metadata["basis_source"] == "state_index"
    assert float(jnp.sqrt(jnp.mean((projected_value - solution.V) ** 2))) < 1e-8


def test_sees_high_dimensional_basis_represents_exact_value():
    dgp = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)
    solution = solve_known_truth(dgp)
    estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(dgp.problem.num_states, dgp.problem)
    alpha = _project_value(basis, solution.V)
    projected_value = basis @ alpha

    assert estimator._last_basis_metadata["basis_source"] == "encoded_state"
    assert estimator._last_basis_metadata["state_feature_dim"] == 16
    assert float(jnp.sqrt(jnp.mean((projected_value - solution.V) ** 2))) < 1e-8


def test_sees_basis_choice_values_reproduce_known_truth_q_policy_and_bellman():
    dgp = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)
    solution = solve_known_truth(dgp)
    estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(dgp.problem.num_states, dgp.problem)
    alpha = _project_value(basis, solution.V)
    q_vals = _choice_values_from_basis(
        dgp,
        basis,
        alpha,
        dgp.homogeneous_parameters,
    )
    policy = jax.nn.softmax(q_vals / dgp.problem.scale_parameter, axis=1)
    bellman_value = dgp.problem.scale_parameter * jax.scipy.special.logsumexp(
        q_vals / dgp.problem.scale_parameter,
        axis=1,
    )
    projected_value = basis @ alpha

    assert float(jnp.max(jnp.abs(q_vals - solution.Q))) < 1e-6
    assert float(jnp.max(jnp.abs(policy - solution.policy))) < 1e-6
    assert float(jnp.max(jnp.abs(projected_value - bellman_value))) < 1e-6


def test_sees_auto_basis_uses_state_encoder_for_canonical_high_action():
    dgp = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)

    encoded_estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )
    encoded_basis = encoded_estimator._build_basis(dgp.problem.num_states, dgp.problem)

    index_estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        state_basis_mode="index",
        compute_se=False,
    )
    index_basis = index_estimator._build_basis(dgp.problem.num_states, dgp.problem)

    encoded_states = dgp.problem.state_encoder(jnp.arange(dgp.problem.num_states))
    assert encoded_states.shape == (dgp.problem.num_states, 16)
    assert encoded_estimator._last_basis_metadata["basis_source"] == "encoded_state"
    assert index_estimator._last_basis_metadata["basis_source"] == "state_index"
    assert not jnp.allclose(encoded_basis, index_basis)


def test_value_sees_full_rust_bus_basis_represents_oracle_value():
    env, solution = _rust_bus_truth()
    estimator = SEESEstimator(
        solution="value",
        basis_type="bspline",
        basis_dim=env.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(env.num_states, env.problem_spec)
    alpha = _project_value(basis, solution.V)
    value = basis @ alpha

    assert float(jnp.max(jnp.abs(value - solution.V))) < 1e-8


def test_q_sees_full_rust_bus_basis_represents_oracle_q():
    env, solution = _rust_bus_truth()
    estimator = SEESEstimator(
        solution="q",
        basis_type="bspline",
        basis_dim=env.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(env.num_states, env.problem_spec)
    alpha = _project_state_action(basis, solution.Q)
    q_vals = _state_action_from_alpha(basis, alpha)

    assert float(jnp.max(jnp.abs(q_vals - solution.Q))) < 1e-8


def test_ev_sees_full_rust_bus_basis_represents_oracle_continuation_values():
    env, solution = _rust_bus_truth()
    estimator = SEESEstimator(
        solution="ev",
        basis_type="bspline",
        basis_dim=env.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(env.num_states, env.problem_spec)
    continuation = _expected_value(env.transition_matrices, solution.V)
    alpha = _project_state_action(basis, continuation)
    fitted = _state_action_from_alpha(basis, alpha)

    assert float(jnp.max(jnp.abs(fitted - continuation))) < 1e-8


def test_policy_sees_full_rust_bus_basis_represents_centered_oracle_logits():
    env, solution = _rust_bus_truth()
    estimator = SEESEstimator(
        solution="policy",
        basis_type="bspline",
        basis_dim=env.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(env.num_states, env.problem_spec)
    logits = jnp.log(jnp.clip(solution.policy, 1e-12, 1.0))
    logits = logits - logits.mean(axis=1, keepdims=True)
    alpha = _project_state_action(basis, logits)
    fitted = _state_action_from_alpha(basis, alpha)
    fitted = fitted - fitted.mean(axis=1, keepdims=True)

    assert float(jnp.max(jnp.abs(fitted - logits))) < 1e-8


def test_collocation_sees_oracle_rust_bus_value_has_zero_collocation_residual():
    env, solution = _rust_bus_truth()
    estimator = SEESEstimator(
        solution="collocation",
        basis_type="bspline",
        basis_dim=env.num_states,
        compute_se=False,
    )

    states = jnp.arange(env.num_states, dtype=jnp.int32)
    indices = estimator._collocation_state_indices(env.num_states, states)
    residual = solution.V - (
        env.problem_spec.scale_parameter
        * jax.scipy.special.logsumexp(
            solution.Q / env.problem_spec.scale_parameter,
            axis=1,
        )
    )

    assert int(indices.shape[0]) == env.num_states
    assert float(jnp.max(jnp.abs(residual[indices]))) < 1e-8


@pytest.mark.parametrize(
    "solution_type",
    ["value", "q", "ev", "policy", "collocation"],
)
def test_sees_solution_outputs_exactly_recover_rust_bus_oracle(solution_type):
    env, solution = _rust_bus_truth()
    utility = LinearUtility.from_environment(env)
    estimator = SEESEstimator(
        solution=solution_type,
        basis_type="bspline",
        basis_dim=env.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(env.num_states, env.problem_spec)
    alpha, projection_rmse = _rust_oracle_alpha(
        solution_type,
        basis,
        env,
        solution,
    )
    _, value, policy, residual, q_vals = estimator._evaluate_solution_outputs(
        env.get_true_parameter_vector(),
        alpha,
        basis=basis,
        feature_matrix=utility.feature_matrix,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    assert projection_rmse < 1e-10
    assert float(jnp.max(jnp.abs(residual))) < 1e-8
    assert float(jnp.max(jnp.abs(policy - solution.policy))) < 1e-8
    assert float(jnp.max(jnp.abs(value - solution.V))) < 1e-8
    assert float(jnp.max(jnp.abs(q_vals - solution.Q))) < 1e-8
