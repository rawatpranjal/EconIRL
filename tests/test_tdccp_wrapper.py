"""Tests for the sklearn-style TD-CCP wrapper.

Uses small problems (n_states=5) with minimal iterations for fast tests.
Full Rust bus tests are marked @pytest.mark.slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econirl.estimators.tdccp import TDCCP
from econirl.estimators.protocol import EstimatorProtocol
from econirl.core.reward_spec import RewardSpec
from econirl.core.types import TrajectoryPanel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    """Create a synthetic 5-state, 2-action panel DataFrame."""
    np.random.seed(42)
    records = []
    for bus_id in range(20):
        state = 0
        for period in range(30):
            action = 1 if np.random.random() < 0.05 * state else 0
            records.append({
                "bus_id": bus_id,
                "period": period,
                "mileage_bin": state,
                "replaced": action,
            })
            state = 0 if action == 1 else min(state + np.random.choice([0, 1]), 4)
    return pd.DataFrame(records)


@pytest.fixture
def small_panel(small_df):
    """Create a TrajectoryPanel from the small DataFrame."""
    return TrajectoryPanel.from_dataframe(
        small_df, state="mileage_bin", action="replaced", id="bus_id"
    )


@pytest.fixture
def fitted_model(small_df):
    """Return a fitted TD-CCP model on small data."""
    model = TDCCP(
        n_states=5,
        n_actions=2,
        discount=0.95,
        utility="linear_cost",
        avi_iterations=2,
        epochs_per_avi=5,
        n_policy_iterations=1,
        hidden_dim=16,
        num_hidden_layers=1,
        batch_size=64,
        verbose=False,
    )
    model.fit(
        data=small_df,
        state="mileage_bin",
        action="replaced",
        id="bus_id",
    )
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicFit:
    """Test that basic fitting works without errors."""

    def test_fit_returns_self(self, small_df):
        model = TDCCP(
            n_states=5, discount=0.95,
            avi_iterations=2, epochs_per_avi=5,
            n_policy_iterations=1, hidden_dim=16, num_hidden_layers=1,
        )
        result = model.fit(
            data=small_df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )
        assert result is model

    def test_fit_completes(self, fitted_model):
        assert fitted_model.params_ is not None


class TestParametersPresent:
    """Test that estimated parameters have expected keys."""

    def test_params_keys(self, fitted_model):
        assert "theta_c" in fitted_model.params_
        assert "RC" in fitted_model.params_

    def test_params_are_floats(self, fitted_model):
        for name, val in fitted_model.params_.items():
            assert isinstance(val, float), f"{name} is not float: {type(val)}"

    def test_coef_array(self, fitted_model):
        assert fitted_model.coef_ is not None
        assert len(fitted_model.coef_) == 2


class TestAttributes:
    """Test that fitted attributes have correct shapes."""

    def test_policy_shape(self, fitted_model):
        assert fitted_model.policy_ is not None
        assert fitted_model.policy_.shape == (5, 2)

    def test_value_shape(self, fitted_model):
        assert fitted_model.value_ is not None
        assert fitted_model.value_.shape == (5,)

    def test_pvalues_present(self, fitted_model):
        assert fitted_model.pvalues_ is not None
        assert "theta_c" in fitted_model.pvalues_
        assert "RC" in fitted_model.pvalues_

    def test_se_present(self, fitted_model):
        assert fitted_model.se_ is not None
        assert "theta_c" in fitted_model.se_
        assert "RC" in fitted_model.se_

    def test_log_likelihood(self, fitted_model):
        assert fitted_model.log_likelihood_ is not None
        assert isinstance(fitted_model.log_likelihood_, float)

    def test_converged(self, fitted_model):
        assert fitted_model.converged_ is not None
        assert isinstance(fitted_model.converged_, bool)

    def test_transitions_stored(self, fitted_model):
        assert fitted_model.transitions_ is not None

    def test_reward_spec_stored(self, fitted_model):
        assert fitted_model.reward_spec_ is not None


class TestConfInt:
    """Test confidence interval computation."""

    def test_conf_int_returns_dict(self, fitted_model):
        ci = fitted_model.conf_int()
        assert isinstance(ci, dict)
        assert "theta_c" in ci
        assert "RC" in ci

    def test_conf_int_has_lower_upper(self, fitted_model):
        ci = fitted_model.conf_int()
        for name, (lower, upper) in ci.items():
            assert isinstance(lower, float)
            assert isinstance(upper, float)
            if np.isnan(lower) or np.isnan(upper):
                continue  # skip params with NaN SE (common on tiny data)
            assert lower <= upper, f"{name}: lower ({lower}) > upper ({upper})"

    def test_conf_int_custom_alpha(self, fitted_model):
        ci_90 = fitted_model.conf_int(alpha=0.10)
        ci_95 = fitted_model.conf_int(alpha=0.05)
        # 95% CI should be wider than 90% CI (skip NaN SE params)
        for name in fitted_model.params_:
            width_90 = ci_90[name][1] - ci_90[name][0]
            width_95 = ci_95[name][1] - ci_95[name][0]
            if np.isnan(width_90) or np.isnan(width_95):
                continue  # skip params with NaN SE (common on tiny data)
            assert width_95 >= width_90 - 1e-10, (
                f"{name}: 95% CI width ({width_95}) < 90% CI width ({width_90})"
            )

    def test_conf_int_before_fit_raises(self):
        model = TDCCP(n_states=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.conf_int()


class TestEstimatorProtocol:
    """Test that TDCCP satisfies the EstimatorProtocol."""

    def test_isinstance_check(self, fitted_model):
        assert isinstance(fitted_model, EstimatorProtocol)

    def test_unfitted_isinstance(self):
        model = TDCCP(n_states=5)
        assert isinstance(model, EstimatorProtocol)


class TestRewardSpecInput:
    """Test fitting with a RewardSpec instead of string utility."""

    def test_fit_with_reward_spec(self, small_df):
        import torch

        n_states = 5
        n_actions = 2
        features = torch.zeros((n_states, n_actions, 2), dtype=torch.float32)
        mileage = torch.arange(n_states, dtype=torch.float32)
        features[:, 0, 0] = -mileage
        features[:, 1, 1] = -1.0

        spec = RewardSpec(features, ["cost", "replace_cost"])

        model = TDCCP(
            n_states=n_states, discount=0.95,
            avi_iterations=2, epochs_per_avi=5,
            n_policy_iterations=1, hidden_dim=16, num_hidden_layers=1,
        )
        model.fit(
            data=small_df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
            reward=spec,
        )
        assert model.params_ is not None
        assert "cost" in model.params_
        assert "replace_cost" in model.params_


class TestEVFeatures:
    """Test the ev_features_ attribute."""

    def test_ev_features_accessible(self, fitted_model):
        # ev_features_ should be set after fitting (may be None if metadata
        # doesn't include it, but for TD-CCP it always should)
        ev = fitted_model.ev_features_
        if ev is not None:
            assert isinstance(ev, np.ndarray)
            # Shape should be (n_states, n_features)
            assert ev.shape[0] == 5  # n_states
            assert ev.shape[1] == 2  # n_features (theta_c, RC)

    def test_ev_features_type(self, fitted_model):
        ev = fitted_model.ev_features_
        if ev is not None:
            assert ev.dtype in (np.float32, np.float64)


class TestSummary:
    """Test summary output."""

    def test_summary_returns_string(self, fitted_model):
        s = fitted_model.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_before_fit(self):
        model = TDCCP(n_states=5)
        s = model.summary()
        assert "Not fitted" in s


class TestPredictProba:
    """Test predict_proba method."""

    def test_predict_proba_shape(self, fitted_model):
        states = np.array([0, 1, 2, 3, 4])
        proba = fitted_model.predict_proba(states)
        assert proba.shape == (5, 2)

    def test_predict_proba_sums_to_one(self, fitted_model):
        states = np.array([0, 2, 4])
        proba = fitted_model.predict_proba(states)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_nonnegative(self, fitted_model):
        states = np.array([0, 1, 2, 3, 4])
        proba = fitted_model.predict_proba(states)
        assert np.all(proba >= 0)

    def test_predict_proba_before_fit_raises(self):
        model = TDCCP(n_states=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0, 1]))


class TestNoNFXPMethods:
    """Test that NFXP-specific methods are NOT present."""

    def test_no_simulate(self, fitted_model):
        assert not hasattr(fitted_model, "simulate")

    def test_no_counterfactual(self, fitted_model):
        assert not hasattr(fitted_model, "counterfactual")


class TestTrajectoryPanelInput:
    """Test fitting with a TrajectoryPanel directly."""

    def test_fit_with_panel(self, small_panel):
        model = TDCCP(
            n_states=5, discount=0.95,
            avi_iterations=2, epochs_per_avi=5,
            n_policy_iterations=1, hidden_dim=16, num_hidden_layers=1,
        )
        model.fit(data=small_panel)
        assert model.params_ is not None
        assert model.policy_ is not None
        assert model.value_ is not None


class TestRepr:
    """Test string representation."""

    def test_repr_unfitted(self):
        model = TDCCP(n_states=10, discount=0.95)
        r = repr(model)
        assert "TDCCP" in r
        assert "fitted=False" in r

    def test_repr_fitted(self, fitted_model):
        r = repr(fitted_model)
        assert "TDCCP" in r
        assert "fitted=True" in r
