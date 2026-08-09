"""EconIRL 0.1.0 public-contract tests for CCP."""

from __future__ import annotations

import inspect
import pickle
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from econirl import CCP
from econirl.datasets import rust_bus_reward_spec


def _panel_frame() -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    n_states = 8
    for individual in range(8):
        state = individual % 4
        for period in range(8):
            action = int(state >= 5 or (individual + period) % 6 == 0)
            increment = (individual + 2 * period) % 3
            rows.append(
                {
                    "bus_id": individual,
                    "period": period,
                    "state": state,
                    "action": action,
                }
            )
            state = 0 if action else min(state + increment, n_states - 1)
    return pd.DataFrame(rows)


def _keep_transitions(n_states: int = 8) -> np.ndarray:
    transitions = np.zeros((n_states, n_states), dtype=float)
    for state in range(n_states):
        for increment, probability in ((0, 0.35), (1, 0.55), (2, 0.10)):
            transitions[state, min(state + increment, n_states - 1)] += probability
    return transitions


@pytest.fixture(scope="module")
def fitted_ccp() -> CCP:
    model = CCP(
        n_states=8,
        discount=0.9,
        utility=rust_bus_reward_spec(8),
        se_method="robust",
    )
    model.fit(
        _panel_frame(),
        state="state",
        action="action",
        id="bus_id",
        transitions=_keep_transitions(),
    )
    return model


def test_fit_uses_keyword_only_workflow_arguments() -> None:
    signature = inspect.signature(CCP.fit)

    assert signature.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("state", "action", "id", "transitions", "features", "context", "reward"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_unfitted_state_is_explicit() -> None:
    model = CCP()

    assert model.is_fitted_ is False
    assert model.result_ is None
    assert model.npl_converged_ is None
    assert "Not fitted" in model.summary()


def test_fitted_state_and_diagnostics_follow_contract(fitted_ccp: CCP) -> None:
    model = fitted_ccp

    assert model.is_fitted_ is True
    assert model.result_ is model._result
    assert model.termination_reason_ == "one_step_complete"
    assert model.failure_reason_ is None
    assert model.n_iter_ == 1
    assert model.fit_time_ > 0
    assert model.n_observations_ == len(_panel_frame())
    assert set(model.diagnostics_) == {
        "data",
        "identification",
        "transitions",
        "optimization",
    }
    assert model.diagnostics_["transitions"]["orientation"] == ("(n_actions, n_states, n_states)")
    assert model.diagnostics_["identification"]["contrast_rank"] == 2
    assert model.diagnostics_["optimization"]["npl_converged"] is False


def test_capabilities_are_complete_and_read_only(fitted_ccp: CCP) -> None:
    capabilities = fitted_ccp.capabilities_

    assert isinstance(capabilities, MappingProxyType)
    assert set(capabilities) == {
        "inference",
        "prediction",
        "simulation",
        "counterfactual",
        "serialization",
    }
    assert all(item["status"] == "supported" for item in capabilities.values())
    with pytest.raises(TypeError):
        capabilities["inference"] = {"status": "unsupported"}


def test_summary_has_manager_facing_sections_in_order(fitted_ccp: CCP) -> None:
    report = fitted_ccp.summary()
    headings = [
        "Estimator",
        "Data",
        "Model",
        "Pre-estimation checks",
        "Fit",
        "Outcome",
        "Uncertainty",
        "Limitations",
    ]

    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Converged: yes" in report
    assert "NPL fixed point: no" in report
    assert "Observations: 64" in report


def test_pickle_round_trip_preserves_supported_results(fitted_ccp: CCP) -> None:
    states = np.array([0, 3, 7])
    before_summary = fitted_ccp.summary()
    before_predictions = fitted_ccp.predict_proba(states)

    restored = pickle.loads(pickle.dumps(fitted_ccp))

    assert restored.econirl_version_ == fitted_ccp.econirl_version_
    assert restored.summary() == before_summary
    np.testing.assert_allclose(restored.predict_proba(states), before_predictions)
    before_cf = fitted_ccp.counterfactual(replacement_cost=3.0)
    after_cf = restored.counterfactual(replacement_cost=3.0)
    np.testing.assert_allclose(after_cf.counterfactual_policy, before_cf.counterfactual_policy)


def test_refit_resets_ccp_specific_state(fitted_ccp: CCP) -> None:
    fitted_ccp.npl_parameter_residual_ = 99.0
    fitted_ccp.fit(
        _panel_frame(),
        state="state",
        action="action",
        id="bus_id",
        transitions=_keep_transitions(),
    )

    assert fitted_ccp.npl_parameter_residual_ != 99.0
    assert fitted_ccp.termination_reason_ == "one_step_complete"


def test_full_likelihood_profile_rejects_supplied_transitions() -> None:
    model = CCP(
        n_states=8,
        discount=0.9,
        utility=rust_bus_reward_spec(8),
        se_method="full_likelihood_bhhh",
        num_policy_iterations=-1,
    )

    with pytest.raises(ValueError, match="requires transitions to be estimated"):
        model.fit(
            _panel_frame(),
            state="state",
            action="action",
            id="bus_id",
            transitions=_keep_transitions(),
        )

    assert model.is_fitted_ is False


def test_bootstrap_result_exposes_draws_and_intervals() -> None:
    model = CCP(
        n_states=8,
        discount=0.9,
        utility=rust_bus_reward_spec(8),
        se_method="bootstrap",
        n_bootstrap=3,
        se_seed=19,
    )
    model.fit(
        _panel_frame(),
        state="state",
        action="action",
        id="bus_id",
        transitions=_keep_transitions(),
    )

    assert model.bootstrap_ is not None
    assert model.bootstrap_.method == "pairs_cluster"
    assert model.bootstrap_.unit == "individual_trajectory"
    assert model.bootstrap_.n_requested == 3
    assert model.bootstrap_.n_successful >= 2
    assert model.bootstrap_.estimates.shape == (
        model.bootstrap_.n_successful,
        len(model.params_),
    )
    assert model.bootstrap_.intervals.shape == (len(model.params_), 2)
    assert np.isfinite(model.bootstrap_.estimates).all()
    assert np.isfinite(model.bootstrap_.intervals).all()
    assert "Bootstrap successful draws:" in model.summary()
