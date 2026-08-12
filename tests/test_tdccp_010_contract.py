"""EconIRL 0.1.0 public-contract tests for TD-CCP."""

from __future__ import annotations

import inspect
import pickle
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from econirl import TDCCP
from econirl.datasets import rust_bus_reward_spec


def _panel_frame() -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for individual in range(12):
        state = individual % 4
        for period in range(10):
            action = int(state >= 6 or (individual + period) % 7 == 0)
            rows.append(
                {
                    "bus_id": individual,
                    "period": period,
                    "state": state,
                    "action": action,
                }
            )
            increment = (individual + 2 * period) % 3
            state = 0 if action else min(state + increment, 7)
    return pd.DataFrame(rows)


def _transition_tensor(n_states: int = 8) -> np.ndarray:
    keep = np.zeros((n_states, n_states), dtype=float)
    for state in range(n_states):
        for increment, probability in ((0, 0.35), (1, 0.55), (2, 0.10)):
            keep[state, min(state + increment, n_states - 1)] += probability
    tensor = np.empty((2, n_states, n_states), dtype=float)
    tensor[0] = keep
    tensor[1] = keep[0]
    return tensor


@pytest.fixture(scope="module")
def fitted_tdccp() -> TDCCP:
    model = TDCCP(
        n_states=8,
        discount=0.9,
        utility=rust_bus_reward_spec(8),
        se_method="robust",
        cross_fitting=True,
        robust_se=True,
        seed=19,
    )
    model.fit(
        _panel_frame(),
        state="state",
        action="action",
        id="bus_id",
        transitions=_transition_tensor(),
    )
    return model


def test_fit_uses_keyword_only_workflow_arguments() -> None:
    signature = inspect.signature(TDCCP.fit)
    assert signature.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("state", "action", "id", "transitions", "features", "context", "reward"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_unfitted_state_is_explicit() -> None:
    model = TDCCP()
    assert model.is_fitted_ is False
    assert model.result_ is None
    assert model.termination_reason_ is None
    assert "Not fitted" in model.summary()


def test_fitted_state_and_diagnostics_follow_contract(fitted_tdccp: TDCCP) -> None:
    model = fitted_tdccp
    assert model.is_fitted_ is True
    assert model.result_ is model._result
    assert model.failure_reason_ is None
    assert model.termination_reason_ in {"converged", "optimizer_complete"}
    assert model.n_iter_ >= 0
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
    assert model.transition_tensor_.shape == (2, 8, 8)


def test_capabilities_are_complete_and_read_only(fitted_tdccp: TDCCP) -> None:
    capabilities = fitted_tdccp.capabilities_
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


def test_configuration_reaches_lower_level_result(fitted_tdccp: TDCCP) -> None:
    metadata = fitted_tdccp.result_.metadata
    assert metadata["method"] == "semigradient"
    assert metadata["cross_fitting"] is True
    assert metadata["robust_se"] is True
    assert metadata["se_method_detail"] == "tdccp_algorithm2_locally_robust"


def test_summary_has_manager_facing_sections_in_order(fitted_tdccp: TDCCP) -> None:
    report = fitted_tdccp.summary()
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
    assert "Algorithm 2 locally robust" in report
    assert "Observations: 120" in report


def test_refit_resets_tdccp_specific_state(fitted_tdccp: TDCCP) -> None:
    fitted_tdccp.ev_features_ = np.full((8, 2), 99.0)
    fitted_tdccp.fit(
        _panel_frame(),
        state="state",
        action="action",
        id="bus_id",
        transitions=_transition_tensor(),
    )
    assert fitted_tdccp.ev_features_ is not None
    assert not np.all(fitted_tdccp.ev_features_ == 99.0)
    assert fitted_tdccp.failure_reason_ is None


@pytest.mark.parametrize("argument", ["features", "context"])
def test_unused_common_workflow_arguments_fail_explicitly(argument: str) -> None:
    model = TDCCP(n_states=8, discount=0.9, utility=rust_bus_reward_spec(8))
    kwargs = {argument: np.ones((8, 2))}
    with pytest.raises(ValueError, match=argument):
        model.fit(
            _panel_frame(),
            state="state",
            action="action",
            id="bus_id",
            transitions=_transition_tensor(),
            **kwargs,
        )


def test_supplied_transitions_require_canonical_tensor() -> None:
    model = TDCCP(n_states=8, discount=0.9, utility=rust_bus_reward_spec(8))
    with pytest.raises(ValueError, match="3D action-specific tensor"):
        model.fit(
            _panel_frame(),
            state="state",
            action="action",
            id="bus_id",
            transitions=_transition_tensor()[0],
        )


def test_pickle_round_trip_preserves_supported_results(fitted_tdccp: TDCCP) -> None:
    states = np.array([0, 3, 7])
    restored = pickle.loads(pickle.dumps(fitted_tdccp))
    assert restored.econirl_version_ == fitted_tdccp.econirl_version_
    assert restored.summary() == fitted_tdccp.summary()
    np.testing.assert_allclose(restored.predict_proba(states), fitted_tdccp.predict_proba(states))
    before = fitted_tdccp.counterfactual(replacement_cost=3.0)
    after = restored.counterfactual(replacement_cost=3.0)
    np.testing.assert_allclose(after.counterfactual_policy, before.counterfactual_policy)


def test_encoded_state_features_are_publicly_configurable() -> None:
    encoded = np.column_stack([np.linspace(0.0, 1.0, 8), np.arange(8) % 2, (np.arange(8) // 2) % 2])
    model = TDCCP(
        n_states=8,
        discount=0.9,
        utility=rust_bus_reward_spec(8),
        basis_type="encoded",
        basis_dim=2,
        state_features=encoded,
        ccp_method="logit",
        ccp_use_encoder=True,
        seed=23,
    )
    model.fit(
        _panel_frame(),
        state="state",
        action="action",
        id="bus_id",
        transitions=_transition_tensor(),
    )
    transformed = np.asarray(model._problem.state_encoder(np.array([1, 4])))
    np.testing.assert_allclose(transformed, encoded[[1, 4]])
    assert model.result_.metadata["basis_type"] == "encoded"
    assert model.result_.metadata["ccp_poly_degree"] == model.ccp_poly_degree


def test_simulation_and_both_counterfactual_paths(fitted_tdccp: TDCCP) -> None:
    simulated = fitted_tdccp.simulate(n_agents=4, n_periods=5, seed=29)
    assert len(simulated) == 20
    reward_cf = fitted_tdccp.counterfactual(replacement_cost=3.0)
    assert reward_cf.counterfactual_policy.shape == (8, 2)
    changed = _transition_tensor()
    changed[0] = np.eye(8)
    transition_cf = fitted_tdccp.counterfactual(transitions=changed)
    assert transition_cf.counterfactual_policy.shape == (8, 2)


def test_bootstrap_result_exposes_draws_and_intervals() -> None:
    model = TDCCP(
        n_states=8,
        discount=0.9,
        utility=rust_bus_reward_spec(8),
        se_method="bootstrap",
        n_bootstrap=3,
        se_seed=31,
        cross_fitting=False,
        robust_se=False,
    )
    model.fit(
        _panel_frame(),
        state="state",
        action="action",
        id="bus_id",
        transitions=_transition_tensor(),
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
