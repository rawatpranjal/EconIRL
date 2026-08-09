"""EconIRL 0.1.0 public-contract tests for NFXP."""

from __future__ import annotations

import inspect
import pickle
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from econirl import NFXP
from econirl.core.reward_spec import RewardSpec
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
def fitted_nfxp() -> NFXP:
    model = NFXP(
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
    signature = inspect.signature(NFXP.fit)

    assert signature.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("state", "action", "id", "transitions", "features", "context", "reward"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_states": 0}, "n_states"),
        ({"n_actions": 1}, "n_actions"),
        ({"discount": 1.0}, "discount"),
        ({"n_bootstrap": -1}, "n_bootstrap"),
        ({"se_method": "unknown"}, "se_method"),
        ({"se_method": "bootstrap", "n_bootstrap": 1}, "n_bootstrap"),
    ],
)
def test_constructor_rejects_invalid_shared_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NFXP(**kwargs)


def test_unfitted_state_is_explicit() -> None:
    model = NFXP()

    assert model.is_fitted_ is False
    assert model.result_ is None
    assert "Not fitted" in model.summary()


def test_fitted_state_and_diagnostics_follow_contract(fitted_nfxp: NFXP) -> None:
    model = fitted_nfxp

    assert model.is_fitted_ is True
    assert model.result_ is model._result
    assert model.termination_reason_
    assert model.failure_reason_ is None
    assert model.n_iter_ > 0
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
    assert model.diagnostics_["identification"]["verdict"] == "identified"


def test_capabilities_are_complete_and_read_only(fitted_nfxp: NFXP) -> None:
    capabilities = fitted_nfxp.capabilities_

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


def test_summary_has_manager_facing_sections_in_order(fitted_nfxp: NFXP) -> None:
    report = fitted_nfxp.summary()
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
    assert "Observations: 64" in report


@pytest.mark.parametrize(
    "states",
    [
        np.array([-1]),
        np.array([8]),
        np.array([1.5]),
        np.array([[1, 2]]),
    ],
)
def test_predict_proba_rejects_invalid_state_codes(fitted_nfxp: NFXP, states: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fitted_nfxp.predict_proba(states)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_conf_int_rejects_invalid_alpha(fitted_nfxp: NFXP, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        fitted_nfxp.conf_int(alpha=alpha)


def test_rank_deficient_reward_stops_before_optimization() -> None:
    n_states = 4
    features = np.zeros((n_states, 2, 2), dtype=float)
    features[:, 1, 0] = 1.0
    features[:, :, 1] = np.arange(n_states)[:, None]
    reward = RewardSpec(features, names=["action", "state_only"])
    frame = _panel_frame().query("state < 4").copy()

    model = NFXP(n_states=n_states, discount=0.9, utility=reward)
    with pytest.raises(ValueError, match="action-contrast rank"):
        model.fit(
            frame,
            state="state",
            action="action",
            id="bus_id",
            transitions=_keep_transitions(n_states),
        )
    assert model.is_fitted_ is False


@pytest.mark.parametrize("column, value", [("state", 1.5), ("bus_id", np.nan)])
def test_dataframe_rejects_invalid_state_or_identifier(column: str, value: float) -> None:
    frame = _panel_frame()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    model = NFXP(n_states=8, discount=0.9, utility=rust_bus_reward_spec(8))

    with pytest.raises(ValueError, match=column):
        model.fit(
            frame,
            state="state",
            action="action",
            id="bus_id",
            transitions=_keep_transitions(),
        )

    assert model.is_fitted_ is False


def test_pickle_round_trip_preserves_supported_results(fitted_nfxp: NFXP) -> None:
    states = np.array([0, 3, 7])
    before_summary = fitted_nfxp.summary()
    before_predictions = fitted_nfxp.predict_proba(states)

    restored = pickle.loads(pickle.dumps(fitted_nfxp))

    assert restored.econirl_version_ == fitted_nfxp.econirl_version_
    assert restored.summary() == before_summary
    np.testing.assert_allclose(restored.predict_proba(states), before_predictions)
    before_cf = fitted_nfxp.counterfactual(replacement_cost=3.0)
    after_cf = restored.counterfactual(replacement_cost=3.0)
    np.testing.assert_allclose(after_cf.counterfactual_policy, before_cf.counterfactual_policy)


def test_bootstrap_result_exposes_draws_and_intervals() -> None:
    model = NFXP(
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
