"""Tests for sklearn-style CCP estimator.

Tests the CCP estimator class which provides a scikit-learn style
interface for the Hotz-Miller / NPL algorithm (Hotz & Miller 1993,
Aguirregabiria & Mira 2002).
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp

from econirl.core.types import Panel, Trajectory


class TestCCPInit:
    """Tests for CCP initialization."""

    def test_ccp_init_defaults(self):
        """CCP can be initialized with default parameters."""
        from econirl.estimators import CCP

        estimator = CCP()

        assert estimator.n_states == 90
        assert estimator.n_actions == 2
        assert estimator.discount == 0.9999
        assert estimator.utility == "linear_cost"
        assert estimator.se_method == "robust"
        assert estimator.verbose is False

    def test_ccp_init_custom(self):
        """CCP can be initialized with custom parameters."""
        from econirl.estimators import CCP

        estimator = CCP(
            n_states=50,
            n_actions=3,
            discount=0.95,
            utility="linear_cost",
            se_method="asymptotic",
            verbose=True,
        )

        assert estimator.n_states == 50
        assert estimator.n_actions == 3
        assert estimator.discount == 0.95
        assert estimator.utility == "linear_cost"
        assert estimator.se_method == "asymptotic"
        assert estimator.verbose is True

    def test_ccp_init_with_config(self):
        """CCP can be initialized with various config options."""
        from econirl.estimators import CCP

        # Test with different configurations
        estimator = CCP(n_states=100, discount=0.99)
        assert estimator.n_states == 100
        assert estimator.discount == 0.99


class TestCCPFit:
    """Tests for CCP.fit() method."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n_individuals = 10
        n_periods = 20

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                # Simple stochastic policy
                action = 1 if state > 50 or np.random.random() < 0.05 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        return pd.DataFrame(data)

    def test_ccp_fit_returns_self(self, sample_dataframe):
        """fit() should return self for method chaining."""
        from econirl.estimators import CCP

        estimator = CCP(n_states=90, verbose=False)
        result = estimator.fit(
            data=sample_dataframe,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        assert result is estimator

    def test_ccp_fit_with_explicit_transitions(self, sample_dataframe):
        """Can provide pre-estimated transitions to fit()."""
        from econirl.estimators import CCP

        # Create a simple transition matrix
        n_states = 90
        transitions = np.zeros((n_states, n_states))
        for s in range(n_states):
            for delta, p in [(0, 0.3), (1, 0.6), (2, 0.1)]:
                s_next = min(s + delta, n_states - 1)
                transitions[s, s_next] += p

        estimator = CCP(n_states=n_states, verbose=False)
        result = estimator.fit(
            data=sample_dataframe,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
            transitions=transitions,
        )

        assert result is estimator
        # Transitions should match what we provided
        np.testing.assert_allclose(estimator.transitions_, transitions, atol=1e-6)


class TestCCPAttributes:
    """Tests for CCP attributes after fit()."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a CCP estimator."""
        from econirl.estimators import CCP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = CCP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_ccp_params_(self, fitted_estimator):
        """params_ should be a dict with theta_c and RC."""
        assert hasattr(fitted_estimator, "params_")
        assert isinstance(fitted_estimator.params_, dict)
        assert "theta_c" in fitted_estimator.params_
        assert "RC" in fitted_estimator.params_

        # Parameters should be finite
        assert np.isfinite(fitted_estimator.params_["theta_c"])
        assert np.isfinite(fitted_estimator.params_["RC"])

    def test_ccp_se_(self, fitted_estimator):
        """se_ should be a dict with standard errors."""
        assert hasattr(fitted_estimator, "se_")
        assert isinstance(fitted_estimator.se_, dict)
        assert "theta_c" in fitted_estimator.se_
        assert "RC" in fitted_estimator.se_

        # Standard errors should be non-negative
        assert fitted_estimator.se_["theta_c"] >= 0
        assert fitted_estimator.se_["RC"] >= 0

    def test_ccp_coef_(self, fitted_estimator):
        """coef_ should be a numpy array of coefficients."""
        assert hasattr(fitted_estimator, "coef_")
        assert isinstance(fitted_estimator.coef_, np.ndarray)
        assert len(fitted_estimator.coef_) == 2  # theta_c and RC

        # Should match params_
        assert np.isclose(fitted_estimator.coef_[0], fitted_estimator.params_["theta_c"])
        assert np.isclose(fitted_estimator.coef_[1], fitted_estimator.params_["RC"])

    def test_ccp_log_likelihood_(self, fitted_estimator):
        """log_likelihood_ should be available and negative."""
        assert hasattr(fitted_estimator, "log_likelihood_")
        assert isinstance(fitted_estimator.log_likelihood_, float)
        assert fitted_estimator.log_likelihood_ < 0  # Log-likelihood is negative for probabilities < 1

    def test_ccp_value_function_(self, fitted_estimator):
        """value_function_ should be a numpy array."""
        assert hasattr(fitted_estimator, "value_function_")
        assert isinstance(fitted_estimator.value_function_, np.ndarray)
        assert len(fitted_estimator.value_function_) == fitted_estimator.n_states

    def test_ccp_transitions_(self, fitted_estimator):
        """transitions_ should be available after fit."""
        assert hasattr(fitted_estimator, "transitions_")
        assert isinstance(fitted_estimator.transitions_, np.ndarray)
        assert fitted_estimator.transitions_.shape == (fitted_estimator.n_states, fitted_estimator.n_states)

        # Rows should sum to 1
        row_sums = fitted_estimator.transitions_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(fitted_estimator.n_states), atol=1e-6)

    def test_ccp_converged_(self, fitted_estimator):
        """converged_ should be a boolean."""
        assert hasattr(fitted_estimator, "converged_")
        assert isinstance(fitted_estimator.converged_, bool)


class TestCCPSameInterfaceAsNFXP:
    """Tests that CCP has the same interface as NFXP."""

    def test_ccp_same_interface_as_nfxp(self):
        """CCP has the same public interface as NFXP."""
        from econirl.estimators import CCP, NFXP

        # Create instances to check both class methods and instance attributes
        nfxp = NFXP()
        ccp = CCP()

        # Get public methods and attributes (not starting with _)
        nfxp_interface = {
            name for name in dir(nfxp)
            if not name.startswith('_')
        }
        ccp_interface = {
            name for name in dir(ccp)
            if not name.startswith('_')
        }

        # Key methods and attributes that must be present
        required_interface = {
            'fit',
            'params_',
            'se_',
            'coef_',
            'log_likelihood_',
            'value_function_',
            'transitions_',
            'converged_',
            'summary',
            'simulate',
            'counterfactual',
            'predict_proba',
            'n_states',
            'n_actions',
            'discount',
            'utility',
            'se_method',
            'verbose',
        }

        # CCP should have all required interface elements
        for item in required_interface:
            assert item in ccp_interface, f"CCP is missing {item}"

        # CCP should have same __init__ parameters as NFXP
        import inspect
        nfxp_params = set(inspect.signature(NFXP.__init__).parameters.keys())
        ccp_params = set(inspect.signature(CCP.__init__).parameters.keys())

        # Remove 'self' from comparison
        nfxp_params.discard('self')
        ccp_params.discard('self')

        # CCP should have at least the same parameters as NFXP
        # (it may have additional ones like num_policy_iterations)
        missing_params = nfxp_params - ccp_params
        assert len(missing_params) == 0, f"CCP is missing init params: {missing_params}"

    def test_ccp_fit_signature_matches_nfxp(self):
        """CCP.fit() has same signature as NFXP.fit()."""
        from econirl.estimators import CCP, NFXP
        import inspect

        nfxp_sig = inspect.signature(NFXP.fit)
        ccp_sig = inspect.signature(CCP.fit)

        # Get parameter names (excluding self)
        nfxp_params = [p for p in nfxp_sig.parameters.keys() if p != 'self']
        ccp_params = [p for p in ccp_sig.parameters.keys() if p != 'self']

        assert nfxp_params == ccp_params


class TestCCPSummary:
    """Tests for CCP.summary() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a CCP estimator."""
        from econirl.estimators import CCP

        np.random.seed(42)
        n_individuals = 15
        n_periods = 25

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 55 or np.random.random() < 0.03 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.4, 0.55, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = CCP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_ccp_summary_returns_string(self, fitted_estimator):
        """summary() should return a formatted string."""
        summary = fitted_estimator.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_ccp_summary_contains_parameters(self, fitted_estimator):
        """summary() should contain parameter names and values."""
        summary = fitted_estimator.summary()

        assert "theta_c" in summary or "operating" in summary.lower() or "cost" in summary.lower()
        assert "RC" in summary or "replacement" in summary.lower()


class TestCCPPredictProba:
    """Tests for CCP.predict_proba() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a CCP estimator."""
        from econirl.estimators import CCP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = CCP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_ccp_predict_proba_single_state(self, fitted_estimator):
        """predict_proba() works with a single state."""
        proba = fitted_estimator.predict_proba(states=np.array([0]))

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 2)  # 1 state, 2 actions

        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), [1.0], atol=1e-6)

        # Probabilities should be non-negative
        assert (proba >= 0).all()

    def test_ccp_predict_proba_multiple_states(self, fitted_estimator):
        """predict_proba() works with multiple states."""
        states = np.array([0, 10, 30, 50, 80])
        proba = fitted_estimator.predict_proba(states=states)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (5, 2)  # 5 states, 2 actions

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)

        # All probabilities should be non-negative
        assert (proba >= 0).all()


class TestCCPSimulate:
    """Tests for CCP.simulate() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a CCP estimator."""
        from econirl.estimators import CCP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = CCP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_ccp_simulate_returns_dataframe(self, fitted_estimator):
        """simulate() should return a pandas DataFrame."""
        result = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)

        assert isinstance(result, pd.DataFrame)

    def test_ccp_simulate_has_correct_columns(self, fitted_estimator):
        """simulate() should return DataFrame with required columns."""
        result = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)

        assert "agent_id" in result.columns
        assert "period" in result.columns
        assert "state" in result.columns
        assert "action" in result.columns

    def test_ccp_simulate_has_correct_shape(self, fitted_estimator):
        """simulate() should return correct number of rows."""
        n_agents = 5
        n_periods = 10
        result = fitted_estimator.simulate(n_agents=n_agents, n_periods=n_periods, seed=42)

        # Should have n_agents * n_periods rows
        assert len(result) == n_agents * n_periods


class TestCCPCounterfactual:
    """Tests for CCP.counterfactual() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a CCP estimator."""
        from econirl.estimators import CCP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = CCP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_ccp_counterfactual_returns_result(self, fitted_estimator):
        """counterfactual() should return a CounterfactualResult."""
        from econirl.estimators.nfxp import CounterfactualResult

        result = fitted_estimator.counterfactual(RC=15.0)

        assert isinstance(result, CounterfactualResult)

    def test_ccp_counterfactual_has_params(self, fitted_estimator):
        """CounterfactualResult should have params dict."""
        result = fitted_estimator.counterfactual(RC=15.0)

        assert hasattr(result, "params")
        assert isinstance(result.params, dict)
        assert "RC" in result.params
        assert result.params["RC"] == 15.0

    def test_ccp_counterfactual_has_policy(self, fitted_estimator):
        """CounterfactualResult should have policy array."""
        result = fitted_estimator.counterfactual(RC=15.0)

        assert hasattr(result, "policy")
        assert isinstance(result.policy, np.ndarray)
        assert result.policy.shape == (fitted_estimator.n_states, fitted_estimator.n_actions)


class TestCCPImport:
    """Tests for CCP import structure."""

    def test_can_import_from_estimators(self):
        """CCP can be imported from econirl.estimators."""
        from econirl.estimators import CCP

        assert CCP is not None

    def test_ccp_in_all(self):
        """CCP is in __all__ of econirl.estimators."""
        from econirl import estimators

        assert hasattr(estimators, "__all__")
        assert "CCP" in estimators.__all__
