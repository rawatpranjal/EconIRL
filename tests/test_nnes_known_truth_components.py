"""Component tests for NNES against the exact known-truth DGP."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from econirl.estimation.nnes import NNESEstimator
from experiments.known_truth import (
    KnownTruthDGP,
    KnownTruthDGPConfig,
    build_known_truth_dgp,
    get_cell,
    run_pre_estimation_diagnostics,
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
            seed=701,
        )
    )


def _component_inputs(dgp: KnownTruthDGP):
    solution = solve_known_truth(dgp)
    estimator = NNESEstimator(
        hidden_dim=8,
        num_layers=1,
        v_epochs=1,
        n_outer_iterations=1,
        compute_se=False,
    )
    return estimator, solution


def _policy_evaluation_system(dgp: KnownTruthDGP, policy: jnp.ndarray):
    policy64 = policy.astype(jnp.float64)
    transitions64 = dgp.transitions.astype(jnp.float64)
    features64 = dgp.feature_matrix.astype(jnp.float64)
    sigma = dgp.problem.scale_parameter
    beta = dgp.problem.discount_factor

    transition_pi = jnp.einsum("sa,ast->st", policy64, transitions64)
    expected_features = jnp.einsum("sa,sak->sk", policy64, features64)
    safe_policy = jnp.maximum(policy64, 1e-12)
    expected_entropy = -sigma * jnp.einsum(
        "sa,sa->s",
        policy64,
        jnp.log(safe_policy),
    )
    lhs = jnp.eye(dgp.problem.num_states, dtype=jnp.float64) - beta * transition_pi
    return lhs, expected_features, expected_entropy


def test_nnes_policy_evaluation_components_solve_linear_system():
    dgp = _low_dim_dgp()
    estimator, solution = _component_inputs(dgp)

    W_z, W_e = estimator._compute_npl_components(
        solution.policy,
        dgp.transitions,
        dgp.feature_matrix,
        dgp.problem.scale_parameter,
        dgp.problem.discount_factor,
        dgp.problem.num_states,
    )
    lhs, expected_features, expected_entropy = _policy_evaluation_system(
        dgp,
        solution.policy,
    )

    feature_residual = lhs @ W_z - expected_features
    entropy_residual = lhs @ W_e - expected_entropy

    assert float(jnp.max(jnp.abs(feature_residual))) < 1e-10
    assert float(jnp.max(jnp.abs(entropy_residual))) < 1e-10


def test_nnes_profiled_value_matches_known_truth_value_at_fixed_point():
    dgp = _low_dim_dgp()
    estimator, solution = _component_inputs(dgp)

    W_z, W_e = estimator._compute_npl_components(
        solution.policy,
        dgp.transitions,
        dgp.feature_matrix,
        dgp.problem.scale_parameter,
        dgp.problem.discount_factor,
        dgp.problem.num_states,
    )
    profiled_value = W_z @ dgp.homogeneous_parameters.astype(jnp.float64) + W_e

    value_error = profiled_value - solution.V.astype(jnp.float64)

    assert solution.converged
    assert float(jnp.max(jnp.abs(value_error))) < 1e-5
    assert float(jnp.sqrt(jnp.mean(value_error**2))) < 1e-6


def test_nnes_profiled_choice_values_reproduce_known_truth_q_and_policy():
    dgp = _low_dim_dgp()
    estimator, solution = _component_inputs(dgp)

    profiled_q, _, _, _, _ = estimator._profiled_choice_values(
        solution.policy,
        dgp.transitions,
        dgp.feature_matrix,
        dgp.homogeneous_parameters,
        dgp.problem.scale_parameter,
        dgp.problem.discount_factor,
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )
    profiled_policy = jax.nn.softmax(profiled_q / dgp.problem.scale_parameter, axis=1)

    q_error = profiled_q - solution.Q.astype(jnp.float64)
    policy_error = profiled_policy - solution.policy.astype(jnp.float64)

    assert float(jnp.max(jnp.abs(q_error))) < 1e-5
    assert float(jnp.sqrt(jnp.mean(q_error**2))) < 1e-6
    assert float(jnp.max(jnp.abs(policy_error))) < 1e-6


def test_nnes_profiled_q_moves_continuation_value_with_theta():
    dgp = _low_dim_dgp()
    estimator, solution = _component_inputs(dgp)
    theta = dgp.homogeneous_parameters

    profiled_q, _, _, _, _ = estimator._profiled_choice_values(
        solution.policy,
        dgp.transitions,
        dgp.feature_matrix,
        theta,
        dgp.problem.scale_parameter,
        dgp.problem.discount_factor,
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )
    delta = jnp.linspace(-0.04, 0.05, theta.shape[0], dtype=theta.dtype)
    perturbed_q, _, _, _, _ = estimator._profiled_choice_values(
        solution.policy,
        dgp.transitions,
        dgp.feature_matrix,
        theta + delta,
        dgp.problem.scale_parameter,
        dgp.problem.discount_factor,
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )

    immediate_only_delta = jnp.einsum(
        "sak,k->sa",
        dgp.feature_matrix.astype(jnp.float64),
        delta.astype(jnp.float64),
    )
    profiled_delta = perturbed_q - profiled_q

    assert float(jnp.max(jnp.abs(profiled_delta - immediate_only_delta))) > 1e-3


def test_nnes_anchor_normalization_only_removes_value_level():
    dgp = _low_dim_dgp()
    estimator, solution = _component_inputs(dgp)

    target = estimator._compute_npl_target(
        solution.policy,
        dgp.transitions,
        dgp.feature_matrix,
        dgp.homogeneous_parameters,
        dgp.problem.scale_parameter,
        dgp.problem.discount_factor,
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )
    anchored_target = target - target[0]

    assert float(anchored_target[0]) == 0.0
    assert jnp.allclose(
        anchored_target[1:] - anchored_target[:-1],
        target[1:] - target[:-1],
        atol=1e-7,
    )
    assert not jnp.allclose(target, anchored_target)


def test_nnes_high_dimensional_truth_is_a_profiled_fixed_point():
    dgp = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)
    diagnostics = run_pre_estimation_diagnostics(dgp)
    estimator, solution = _component_inputs(dgp)

    profiled_q, W_z, W_e, _, _ = estimator._profiled_choice_values(
        solution.policy,
        dgp.transitions,
        dgp.feature_matrix,
        dgp.homogeneous_parameters,
        dgp.problem.scale_parameter,
        dgp.problem.discount_factor,
        dgp.problem.num_states,
        dgp.problem.num_actions,
    )
    profiled_value = W_z @ dgp.homogeneous_parameters.astype(jnp.float64) + W_e
    profiled_policy = jax.nn.softmax(profiled_q / dgp.problem.scale_parameter, axis=1)

    assert diagnostics.passed
    assert diagnostics.feature_rank == diagnostics.num_features
    assert diagnostics.condition_number < 10.0
    assert float(jnp.max(jnp.abs(profiled_value - solution.V))) < 5e-5
    assert float(jnp.max(jnp.abs(profiled_q - solution.Q))) < 5e-5
    assert float(jnp.max(jnp.abs(profiled_policy - solution.policy))) < 5e-5
