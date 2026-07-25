"""Release-contract tests for the public NFXP implementation."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.types import DDCProblem
from econirl.estimators.nfxp import NFXP, CounterfactualResult
from econirl.inference.results import EstimationSummary
from econirl.preferences.linear import LinearUtility


def _action_specific_tensor(n_states: int = 3) -> np.ndarray:
    transitions = np.zeros((2, n_states, n_states), dtype=np.float64)
    transitions[0, :, 0] = 1.0
    transitions[1, :, n_states - 1] = 1.0
    return transitions


def _solved_model() -> NFXP:
    n_states = 5
    features = np.zeros((n_states, 2, 1), dtype=np.float32)
    features[:, 1, 0] = np.linspace(-1.0, 1.0, n_states)
    utility = LinearUtility(
        feature_matrix=jnp.asarray(features),
        parameter_names=["action_one_value"],
    )
    problem = DDCProblem(
        num_states=n_states,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
    )
    transitions = np.zeros((2, n_states, n_states), dtype=np.float32)
    for state in range(n_states):
        transitions[0, state, min(state + 1, n_states - 1)] = 1.0
        transitions[1, state, 0] = 1.0

    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration

    parameters = jnp.array([0.5])
    solution = value_iteration(
        SoftBellmanOperator(problem, jnp.asarray(transitions)),
        utility.compute(parameters),
    )
    summary = EstimationSummary(
        parameters=parameters,
        parameter_names=["action_one_value"],
        standard_errors=jnp.array([0.1]),
        method="NFXP (Nested Fixed Point)",
        value_function=solution.V,
        policy=solution.policy,
    )

    model = NFXP(n_states=n_states, discount=0.9)
    model.params_ = {"action_one_value": 0.5}
    model._utility_fn = utility
    model._problem = problem
    model._result = summary
    model.transitions_ = transitions
    model.transition_tensor_ = transitions
    return model


def test_summary_reports_convergence_and_supplied_transition_source() -> None:
    summary = EstimationSummary(
        parameters=jnp.array([1.0]),
        parameter_names=["theta"],
        standard_errors=jnp.array([0.2]),
        method="NFXP (Nested Fixed Point)",
        converged=False,
        num_iterations=17,
        convergence_message="maximum iterations reached",
        estimation_time=1.25,
        transition_source="supplied action-specific tensor",
    )

    report = summary.summary()

    assert "Converged:   no" in report
    assert "Iterations:  17" in report
    assert "Estimation time: 1.25 seconds" in report
    assert "maximum iterations reached" in report
    assert "Transition source: supplied action-specific tensor" in report


def test_nfxp_summary_forwards_confidence_level() -> None:
    model = NFXP(n_states=3)
    model._result = SimpleNamespace(summary=lambda alpha: f"alpha={alpha}")

    assert model.summary(alpha=0.10) == "alpha=0.1"


def test_build_transition_tensor_rejects_invalid_rows() -> None:
    model = NFXP(n_states=3, n_actions=2)
    invalid = _action_specific_tensor()
    invalid[1, 0] = 0.0

    with pytest.raises(ValueError, match="sum to 1"):
        model._build_transition_tensor(invalid)


def test_two_dimensional_transition_input_is_binary_only() -> None:
    model = NFXP(n_states=3, n_actions=3)

    with pytest.raises(ValueError, match="n_actions=2"):
        model._build_transition_tensor(np.eye(3))


def test_simulate_uses_action_specific_transition_tensor() -> None:
    model = NFXP(n_states=3, n_actions=2)
    model._result = SimpleNamespace(
        policy=np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
    )
    model.transitions_ = np.eye(3)
    model.transition_tensor_ = _action_specific_tensor()

    simulated = model.simulate(n_agents=1, n_periods=3, seed=7)

    assert simulated["state"].tolist() == [0, 2, 2]
    assert simulated["action"].tolist() == [1, 1, 1]


def test_reward_counterfactual_has_rich_summary_and_compatibility_aliases() -> None:
    model = _solved_model()

    counterfactual = model.counterfactual(action_one_value=1.5)
    report = counterfactual.summary()

    assert isinstance(counterfactual, CounterfactualResult)
    assert counterfactual.params["action_one_value"] == 1.5
    np.testing.assert_allclose(counterfactual.policy, counterfactual.counterfactual_policy)
    np.testing.assert_allclose(
        counterfactual.value_function,
        counterfactual.counterfactual_value,
    )
    assert "Counterfactual Summary" in report
    assert "reward parameter change" in report
    assert np.max(np.abs(np.asarray(counterfactual.policy_change))) > 1e-3


def test_transition_counterfactual_uses_new_kernel() -> None:
    model = _solved_model()
    new_transitions = np.asarray(model.transition_tensor_).copy()
    new_transitions[0] = 0.0
    new_transitions[0, :, 0] = 1.0

    counterfactual = model.counterfactual(transitions=new_transitions)

    assert counterfactual.counterfactual_transitions is not None
    np.testing.assert_allclose(counterfactual.counterfactual_transitions, new_transitions)
    assert "environment change" in counterfactual.summary()
