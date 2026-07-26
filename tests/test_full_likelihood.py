"""Tests for the shared Rust full-likelihood score."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration
from econirl.core.types import Panel, Trajectory
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.inference.full_likelihood import (
    build_rust_transition_tensor,
    compute_rust_full_likelihood_bhhh_score,
)
from econirl.preferences.linear import LinearUtility

jax.config.update("jax_enable_x64", True)


def _score_inputs():
    probabilities = np.array([0.4, 0.5, 0.1], dtype=np.float64)
    environment = RustBusEnvironment(
        operating_cost=0.03,
        replacement_cost=2.0,
        num_mileage_bins=5,
        mileage_transition_probs=probabilities,
        discount_factor=0.9,
    )
    utility = LinearUtility.from_environment(environment)
    parameters = jnp.array([0.03, 2.0], dtype=jnp.float64)
    reward = utility.compute(parameters).astype(jnp.float64)
    transitions = build_rust_transition_tensor(5, probabilities)
    operator = SoftBellmanOperator(environment.problem_spec, transitions)
    solution = policy_iteration(operator, reward, tol=1e-13, max_iter=1000)
    assert solution.converged
    panel = Panel(
        trajectories=[
            Trajectory(
                states=jnp.array([0, 1, 3, 4], dtype=jnp.int32),
                actions=jnp.array([0, 0, 1, 0], dtype=jnp.int32),
                next_states=jnp.array([1, 3, 0, 4], dtype=jnp.int32),
                individual_id=1,
            ),
            Trajectory(
                states=jnp.array([2, 4, 1], dtype=jnp.int32),
                actions=jnp.array([0, 1, 0], dtype=jnp.int32),
                next_states=jnp.array([4, 1, 1], dtype=jnp.int32),
                individual_id=2,
            ),
        ]
    )
    increments = np.array([1, 2, 0, 0, 2, 1, 0], dtype=np.int32)
    return environment, utility, parameters, transitions, solution, panel, probabilities, increments


def _per_observation_full_log_likelihood(
    joint_parameters: np.ndarray,
    *,
    environment: RustBusEnvironment,
    utility: LinearUtility,
    panel: Panel,
    increments: np.ndarray,
) -> np.ndarray:
    structural = jnp.asarray(joint_parameters[:2], dtype=jnp.float64)
    probabilities = np.array(
        [
            joint_parameters[2],
            joint_parameters[3],
            1.0 - joint_parameters[2] - joint_parameters[3],
        ],
        dtype=np.float64,
    )
    transitions = build_rust_transition_tensor(5, probabilities)
    operator = SoftBellmanOperator(environment.problem_spec, transitions)
    solution = policy_iteration(
        operator,
        utility.compute(structural).astype(jnp.float64),
        tol=1e-13,
        max_iter=1000,
    )
    assert solution.converged
    log_probabilities = operator.compute_log_choice_probabilities(
        utility.compute(structural).astype(jnp.float64),
        solution.V,
    )
    states = np.asarray(panel.get_all_states(), dtype=int)
    actions = np.asarray(panel.get_all_actions(), dtype=int)
    return np.asarray(log_probabilities[states, actions], dtype=np.float64) + np.log(
        probabilities[increments]
    )


def test_shared_full_likelihood_score_matches_finite_differences():
    (
        environment,
        utility,
        parameters,
        transitions,
        solution,
        panel,
        probabilities,
        increments,
    ) = _score_inputs()
    scores, metadata = compute_rust_full_likelihood_bhhh_score(
        panel=panel,
        utility=utility,
        problem=environment.problem_spec,
        transitions=transitions,
        value_function=solution.V,
        policy=solution.policy,
        transition_probabilities=probabilities,
        transition_increments=increments,
    )

    joint = np.array(
        [parameters[0], parameters[1], probabilities[0], probabilities[1]],
        dtype=np.float64,
    )
    numerical = np.empty_like(np.asarray(scores))
    for column in range(joint.size):
        step = 1e-5 if column < 2 else 1e-6
        plus = joint.copy()
        minus = joint.copy()
        plus[column] += step
        minus[column] -= step
        numerical[:, column] = (
            _per_observation_full_log_likelihood(
                plus,
                environment=environment,
                utility=utility,
                panel=panel,
                increments=increments,
            )
            - _per_observation_full_log_likelihood(
                minus,
                environment=environment,
                utility=utility,
                panel=panel,
                increments=increments,
            )
        ) / (2.0 * step)

    np.testing.assert_allclose(np.asarray(scores), numerical, rtol=2e-5, atol=2e-5)
    assert metadata["joint_parameter_names"] == [
        "operating_cost",
        "replacement_cost",
        "transition_p0",
        "transition_p1",
    ]
    assert metadata["transition_orientation"] == "(n_actions, n_states, n_states)"


def test_nfxp_delegates_to_the_shared_full_likelihood_score():
    environment, utility, _, transitions, solution, panel, probabilities, increments = (
        _score_inputs()
    )
    shared_scores, shared_metadata = compute_rust_full_likelihood_bhhh_score(
        panel=panel,
        utility=utility,
        problem=environment.problem_spec,
        transitions=transitions,
        value_function=solution.V,
        policy=solution.policy,
        transition_probabilities=probabilities,
        transition_increments=increments,
    )
    estimator_scores, estimator_metadata = NFXPEstimator()._compute_full_likelihood_bhhh_score(
        panel,
        utility,
        SoftBellmanOperator(environment.problem_spec, transitions),
        solution.V,
        solution.policy,
        jnp.asarray(probabilities),
        jnp.asarray(increments),
    )

    np.testing.assert_array_equal(np.asarray(estimator_scores), np.asarray(shared_scores))
    assert estimator_metadata == shared_metadata


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("transition_shape", "orientation"),
        ("transition_structure", "do not match"),
        ("increment_length", "one entry per panel observation"),
        ("increment_category", "residual transition category"),
        ("noninteger_increment", "integer category labels"),
    ],
)
def test_shared_full_likelihood_score_rejects_invalid_rust_inputs(mutation, message):
    environment, utility, _, transitions, solution, panel, probabilities, increments = (
        _score_inputs()
    )
    if mutation == "transition_shape":
        transitions = transitions[:, :, :-1]
    elif mutation == "transition_structure":
        transitions = transitions.at[0, 0, 0].add(0.01)
    elif mutation == "increment_length":
        increments = increments[:-1]
    elif mutation == "increment_category":
        increments = increments.copy()
        increments[0] = 3
    elif mutation == "noninteger_increment":
        increments = increments.astype(float)
        increments[0] = 0.5

    with pytest.raises(ValueError, match=message):
        compute_rust_full_likelihood_bhhh_score(
            panel=panel,
            utility=utility,
            problem=environment.problem_spec,
            transitions=transitions,
            value_function=solution.V,
            policy=solution.policy,
            transition_probabilities=probabilities,
            transition_increments=increments,
        )


def test_shared_full_likelihood_score_allows_missing_replacement_increments():
    environment, utility, _, transitions, solution, panel, probabilities, increments = (
        _score_inputs()
    )
    replacement_rows = np.asarray(panel.get_all_actions()) == 1
    increments = increments.copy()
    increments[replacement_rows] = -1

    scores, metadata = compute_rust_full_likelihood_bhhh_score(
        panel=panel,
        utility=utility,
        problem=environment.problem_spec,
        transitions=transitions,
        value_function=solution.V,
        policy=solution.policy,
        transition_probabilities=probabilities,
        transition_increments=increments,
    )

    assert np.all(np.isfinite(np.asarray(scores)))
    assert metadata["transition_counts"] == {0: 2, 1: 1, 2: 2}


def test_shared_full_likelihood_score_rejects_missing_keep_increment():
    environment, utility, _, transitions, solution, panel, probabilities, increments = (
        _score_inputs()
    )
    increments = increments.copy()
    increments[0] = -1

    with pytest.raises(ValueError, match="only replacement actions"):
        compute_rust_full_likelihood_bhhh_score(
            panel=panel,
            utility=utility,
            problem=environment.problem_spec,
            transitions=transitions,
            value_function=solution.V,
            policy=solution.policy,
            transition_probabilities=probabilities,
            transition_increments=increments,
        )
