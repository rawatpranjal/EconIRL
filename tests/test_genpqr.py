"""Tests for GenPQR estimator (van der Laan, Kallus & Bibaut 2025)."""

import numpy as np
import pytest

from econirl.datasets import load_rust_bus
from econirl.estimators import GenPQR


@pytest.fixture(scope="module")
def rust_bus_df():
    return load_rust_bus()


@pytest.fixture(scope="module")
def fitted_basic(rust_bus_df):
    return GenPQR(n_states=90, n_actions=2, discount=0.9999, anchor_action=0).fit(
        rust_bus_df, state="mileage_bin", action="replaced", id="bus_id"
    )


class TestGenPQRBasic:
    def test_converged(self, fitted_basic):
        assert fitted_basic.converged_ is True

    def test_reward_shape(self, fitted_basic):
        assert fitted_basic.reward_.shape == (90, 2)

    def test_anchor_column_zero(self, fitted_basic):
        assert np.allclose(fitted_basic.reward_[:, 0], 0.0, atol=1e-10)

    def test_policy_sums_to_one(self, fitted_basic):
        assert np.allclose(fitted_basic.policy_.sum(axis=1), 1.0, atol=1e-10)

    def test_policy_shape(self, fitted_basic):
        assert fitted_basic.policy_.shape == (90, 2)

    def test_value_function_shape(self, fitted_basic):
        assert fitted_basic.value_function_.shape == (90,)

    def test_transitions_shape(self, fitted_basic):
        assert fitted_basic.transitions_.shape == (2, 90, 90)

    def test_no_params_without_features(self, fitted_basic):
        assert fitted_basic.params_ is None
        assert fitted_basic.se_ is None


class TestGenPQRWithFeatures:
    @pytest.fixture(scope="class")
    def fitted_features(self, rust_bus_df):
        n_states = 90
        s = np.arange(n_states)
        features = np.column_stack([s / 100, (s / 100) ** 2])
        return GenPQR(
            n_states=n_states,
            n_actions=2,
            discount=0.9999,
            feature_matrix=features,
            feature_names=["linear", "quadratic"],
            n_bootstrap=20,
            se_seed=42,
        ).fit(rust_bus_df, state="mileage_bin", action="replaced", id="bus_id")

    def test_params_present(self, fitted_features):
        assert fitted_features.params_ is not None
        assert set(fitted_features.params_.keys()) == {"linear", "quadratic"}

    def test_se_non_negative(self, fitted_features):
        assert all(v >= 0 for v in fitted_features.se_.values())

    def test_coef_shape(self, fitted_features):
        assert fitted_features.coef_.shape == (2,)

    def test_pvalues_present(self, fitted_features):
        assert fitted_features.pvalues_ is not None

    def test_anchor_still_zero(self, fitted_features):
        assert np.allclose(fitted_features.reward_[:, 0], 0.0, atol=1e-10)
