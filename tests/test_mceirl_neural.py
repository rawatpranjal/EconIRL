"""Tests for MCEIRLNeural estimator.

Tests that MCEIRLNeural:
1. Basic fit with transitions
2. params_ populated when features provided
3. projection_r2_ is float
4. policy_ shape correct
5. EstimatorProtocol conformance
6. predict_proba() works
7. summary() returns string
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.reward_spec import RewardSpec
from econirl.estimators.mceirl_neural import MCEIRLNeural
from econirl.estimators.protocol import EstimatorProtocol

# ---------------------------------------------------------------------------
# Fixtures: small 5-state gridworld
# ---------------------------------------------------------------------------

_N_STATES = 5
_N_ACTIONS = 2
_DISCOUNT = 0.9


def _make_gridworld_transitions(
    n_states: int = _N_STATES,
    n_actions: int = _N_ACTIONS,
) -> jnp.ndarray:
    """Create simple deterministic-ish transitions.

    Action 0: move right (state+1, wrapping at end)
    Action 1: stay in place
    """
    T = jnp.zeros((n_actions, n_states, n_states), dtype=jnp.float32)

    for s in range(n_states):
        # Action 0: move right with some noise
        next_s = (s + 1) % n_states
        T = T.at[0, s, next_s].set(0.9)
        T = T.at[0, s, s].set(0.1)

        # Action 1: stay
        T = T.at[1, s, s].set(0.9)
        T = T.at[1, s, (s + 1) % n_states].set(0.1)

    return T


def _make_gridworld_data(
    n_states: int = _N_STATES,
    n_actions: int = _N_ACTIONS,
    n_individuals: int = 20,
    n_periods: int = 30,
    seed: int = 42,
):
    """Generate synthetic gridworld data as a DataFrame."""
    import pandas as pd

    np.random.seed(seed)
    T = np.asarray(_make_gridworld_transitions(n_states, n_actions))
    data = []

    for i in range(n_individuals):
        state = np.random.randint(n_states)
        for t in range(n_periods):
            # Simple policy: prefer action 0 (move) at low states,
            # prefer action 1 (stay) at high states
            p_action0 = 0.8 if state < n_states // 2 else 0.2
            action = 0 if np.random.random() < p_action0 else 1
            next_state = np.random.choice(n_states, p=T[action, state, :])
            data.append(
                {
                    "agent_id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                }
            )
            state = next_state

    return pd.DataFrame(data)


def _make_features(n_states: int = _N_STATES) -> RewardSpec:
    """Create state features for projection."""
    s = jnp.arange(n_states, dtype=jnp.float32)
    state_features = jnp.stack([s / n_states, (s / n_states) ** 2], axis=1)
    return RewardSpec(state_features, names=["linear", "quadratic"], n_actions=_N_ACTIONS)


@pytest.fixture(scope="module")
def transitions():
    return _make_gridworld_transitions()


@pytest.fixture(scope="module")
def gridworld_df():
    return _make_gridworld_data()


@pytest.fixture(scope="module")
def fitted_model(gridworld_df, transitions):
    """Fitted MCEIRLNeural model shared across test classes (state reward)."""
    features = _make_features()
    model = MCEIRLNeural(
        n_states=_N_STATES,
        n_actions=_N_ACTIONS,
        discount=_DISCOUNT,
        reward_type="state",
        max_epochs=100,
        lr=1e-2,
        reward_hidden_dim=32,
        reward_num_layers=1,
        verbose=False,
    )
    model.fit(
        gridworld_df,
        state="state",
        action="action",
        id="agent_id",
        transitions=transitions,
        features=features,
    )
    return model


@pytest.fixture(scope="module")
def fitted_model_no_features(gridworld_df, transitions):
    """Fitted MCEIRLNeural without feature projection (state reward)."""
    model = MCEIRLNeural(
        n_states=_N_STATES,
        n_actions=_N_ACTIONS,
        discount=_DISCOUNT,
        reward_type="state",
        max_epochs=50,
        lr=1e-2,
        reward_hidden_dim=32,
        reward_num_layers=1,
        verbose=False,
    )
    model.fit(
        gridworld_df,
        state="state",
        action="action",
        id="agent_id",
        transitions=transitions,
    )
    return model


@pytest.fixture(scope="module")
def fitted_model_state_action(gridworld_df, transitions):
    """Fitted MCEIRLNeural with reward_type='state_action'."""
    features = _make_features()
    model = MCEIRLNeural(
        n_states=_N_STATES,
        n_actions=_N_ACTIONS,
        discount=_DISCOUNT,
        reward_type="state_action",
        max_epochs=100,
        lr=1e-2,
        reward_hidden_dim=32,
        reward_num_layers=1,
        verbose=False,
    )
    model.fit(
        gridworld_df,
        state="state",
        action="action",
        id="agent_id",
        transitions=transitions,
        features=features,
    )
    return model


# ---------------------------------------------------------------------------
# 1. Basic fit with transitions
# ---------------------------------------------------------------------------


class TestBasicFit:
    """MCEIRLNeural fits with transitions provided."""

    def test_fit_returns_self(self, gridworld_df, transitions):
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state",
            max_epochs=10,
            verbose=False,
        )
        result = model.fit(
            gridworld_df,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
        )
        assert result is model

    def test_requires_transitions(self, gridworld_df):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        with pytest.raises(ValueError, match="requires transitions"):
            model.fit(
                gridworld_df,
                state="state",
                action="action",
                id="agent_id",
            )

    def test_reward_populated(self, fitted_model_no_features):
        assert fitted_model_no_features.reward_ is not None
        assert isinstance(fitted_model_no_features.reward_, np.ndarray)
        assert fitted_model_no_features.reward_.shape == (_N_STATES,)

    def test_converged_is_bool(self, fitted_model_no_features):
        assert fitted_model_no_features.converged_ is not None
        assert isinstance(fitted_model_no_features.converged_, bool)

    def test_n_epochs_positive(self, fitted_model_no_features):
        assert fitted_model_no_features.n_epochs_ is not None
        assert fitted_model_no_features.n_epochs_ > 0


# ---------------------------------------------------------------------------
# 2. params_ populated when features provided
# ---------------------------------------------------------------------------


class TestParamsWithFeatures:
    """params_ populated when features are provided."""

    def test_params_present(self, fitted_model):
        assert fitted_model.params_ is not None
        assert "linear" in fitted_model.params_
        assert "quadratic" in fitted_model.params_

    def test_projection_is_not_sampling_inference(self, fitted_model):
        assert fitted_model.se_ is None
        assert fitted_model.pvalues_ is None
        assert fitted_model.projection_diagnostics_["sampling_inference"] is False

    def test_projection_diagnostics_present(self, fitted_model):
        assert fitted_model.projection_diagnostics_["rank"] == 2
        assert np.isfinite(fitted_model.projection_diagnostics_["condition_number"])

    def test_coef_present(self, fitted_model):
        assert fitted_model.coef_ is not None
        assert isinstance(fitted_model.coef_, np.ndarray)
        assert len(fitted_model.coef_) == 2

    def test_params_none_without_features(self, fitted_model_no_features):
        assert fitted_model_no_features.params_ is None
        assert fitted_model_no_features.se_ is None
        assert fitted_model_no_features.pvalues_ is None


# ---------------------------------------------------------------------------
# 3. projection_r2_ is float
# ---------------------------------------------------------------------------


class TestProjectionR2:
    """projection_r2_ is a float when features are provided."""

    def test_r2_is_float(self, fitted_model):
        assert fitted_model.projection_r2_ is not None
        assert isinstance(fitted_model.projection_r2_, float)

    def test_r2_none_without_features(self, fitted_model_no_features):
        assert fitted_model_no_features.projection_r2_ is None


# ---------------------------------------------------------------------------
# 4. policy_ shape correct
# ---------------------------------------------------------------------------


class TestPolicyShape:
    """policy_ has correct shape."""

    def test_policy_shape(self, fitted_model):
        assert fitted_model.policy_ is not None
        assert fitted_model.policy_.shape == (_N_STATES, _N_ACTIONS)

    def test_policy_valid_probabilities(self, fitted_model):
        assert (fitted_model.policy_ >= 0).all()
        assert (fitted_model.policy_ <= 1).all()
        row_sums = fitted_model.policy_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(_N_STATES), atol=1e-6)

    def test_value_shape(self, fitted_model):
        assert fitted_model.value_ is not None
        assert fitted_model.value_.shape == (_N_STATES,)


# ---------------------------------------------------------------------------
# 5. EstimatorProtocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """MCEIRLNeural satisfies the EstimatorProtocol."""

    def test_satisfies_protocol(self, fitted_model):
        assert isinstance(fitted_model, EstimatorProtocol)

    def test_unfitted_has_protocol_attributes(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        assert hasattr(model, "params_")
        assert hasattr(model, "se_")
        assert hasattr(model, "pvalues_")
        assert hasattr(model, "policy_")
        assert hasattr(model, "value_")

    def test_protocol_methods_present(self, fitted_model):
        assert callable(getattr(fitted_model, "fit", None))
        assert callable(getattr(fitted_model, "summary", None))
        assert callable(getattr(fitted_model, "predict_proba", None))
        assert callable(getattr(fitted_model, "conf_int", None))


# ---------------------------------------------------------------------------
# 6. predict_proba() works
# ---------------------------------------------------------------------------


class TestPredictProba:
    """predict_proba() returns valid probabilities."""

    def test_predict_proba_shape(self, fitted_model):
        proba = fitted_model.predict_proba(np.array([0, 2, 4]))
        assert proba.shape == (3, _N_ACTIONS)

    def test_predict_proba_valid(self, fitted_model):
        proba = fitted_model.predict_proba(np.array([0, 1, 2]))
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)

    def test_predict_proba_unfitted_raises(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0]))


# ---------------------------------------------------------------------------
# 7. summary() returns string
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns a non-empty string."""

    def test_summary_returns_string(self, fitted_model):
        summary = fitted_model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "MCEIRLNeural" in summary

    def test_summary_unfitted(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        summary = model.summary()
        assert isinstance(summary, str)
        assert "Not fitted" in summary

    def test_repr(self, fitted_model):
        r = repr(fitted_model)
        assert "MCEIRLNeural" in r
        assert "fitted=True" in r

    def test_repr_unfitted(self):
        r = repr(MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS))
        assert "fitted=False" in r


# ---------------------------------------------------------------------------
# 8. conf_int() works with features
# ---------------------------------------------------------------------------


class TestConfInt:
    """Neural reward-map projection does not imply sampling inference."""

    @pytest.mark.parametrize("fixture_name", ["fitted_model", "fitted_model_no_features"])
    def test_conf_int_is_unsupported(self, request, fixture_name):
        model = request.getfixturevalue(fixture_name)
        with pytest.raises(NotImplementedError, match="does not report"):
            model.conf_int()


# ---------------------------------------------------------------------------
# 9. reward_type="state_action" tests
# ---------------------------------------------------------------------------


class TestStateActionRewardType:
    """Tests for reward_type='state_action' (R(s,a) network)."""

    def test_reward_shape_state_action(self, fitted_model_state_action):
        """R(s,a) reward should be (n_states, n_actions)."""
        assert fitted_model_state_action.reward_ is not None
        assert fitted_model_state_action.reward_.shape == (
            _N_STATES,
            _N_ACTIONS,
        )

    def test_policy_shape_state_action(self, fitted_model_state_action):
        """policy_ shape unchanged regardless of reward_type."""
        assert fitted_model_state_action.policy_ is not None
        assert fitted_model_state_action.policy_.shape == (
            _N_STATES,
            _N_ACTIONS,
        )

    def test_policy_valid_probabilities_state_action(self, fitted_model_state_action):
        policy = fitted_model_state_action.policy_
        assert (policy >= 0).all()
        assert (policy <= 1).all()
        row_sums = policy.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(_N_STATES), atol=1e-6)

    def test_params_populated_state_action(self, fitted_model_state_action):
        """params_ populated when features provided with state_action."""
        assert fitted_model_state_action.params_ is not None
        assert "linear" in fitted_model_state_action.params_
        assert "quadratic" in fitted_model_state_action.params_

    def test_projection_r2_state_action(self, fitted_model_state_action):
        assert fitted_model_state_action.projection_r2_ is not None
        assert isinstance(fitted_model_state_action.projection_r2_, float)

    def test_value_shape_state_action(self, fitted_model_state_action):
        assert fitted_model_state_action.value_ is not None
        assert fitted_model_state_action.value_.shape == (_N_STATES,)

    def test_predict_proba_state_action(self, fitted_model_state_action):
        proba = fitted_model_state_action.predict_proba(np.array([0, 2, 4]))
        assert proba.shape == (3, _N_ACTIONS)
        assert (proba >= 0).all()
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)

    def test_summary_state_action(self, fitted_model_state_action):
        summary = fitted_model_state_action.summary()
        assert isinstance(summary, str)
        assert "MCEIRLNeural" in summary
        assert "state_action" in summary

    def test_conf_int_state_action(self, fitted_model_state_action):
        with pytest.raises(NotImplementedError, match="does not report"):
            fitted_model_state_action.conf_int()

    def test_invalid_reward_type_raises(self):
        with pytest.raises(ValueError, match="reward_type"):
            MCEIRLNeural(
                n_states=_N_STATES,
                n_actions=_N_ACTIONS,
                reward_type="invalid",
            )

    def test_default_reward_type_is_state_action(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        assert model.reward_type == "state_action"

    def test_state_action_fit_returns_self(self, gridworld_df, transitions):
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state_action",
            max_epochs=10,
            verbose=False,
        )
        result = model.fit(
            gridworld_df,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
        )
        assert result is model
        # Without features, reward_ should still be (S, A)
        assert result.reward_.shape == (_N_STATES, _N_ACTIONS)


# ---------------------------------------------------------------------------
# Completion contract
# ---------------------------------------------------------------------------


class TestCompletionContract:
    def test_transition_orientation_is_checked(self, gridworld_df):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        wrong = np.zeros((_N_STATES, _N_STATES, _N_ACTIONS))
        with pytest.raises(ValueError, match="orientation"):
            model.fit(
                gridworld_df,
                state="state",
                action="action",
                id="agent_id",
                transitions=wrong,
            )

    def test_transition_rows_are_checked(self, gridworld_df, transitions):
        wrong = np.asarray(transitions).copy()
        wrong[0, 0] *= 0.5
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        with pytest.raises(ValueError, match="sum to one"):
            model.fit(
                gridworld_df,
                state="state",
                action="action",
                id="agent_id",
                transitions=wrong,
            )

    @pytest.mark.parametrize("column", ["state", "action"])
    def test_dataframe_codes_must_be_integer(self, gridworld_df, transitions, column):
        invalid = gridworld_df.copy()
        invalid[column] = invalid[column].astype(float)
        invalid.loc[invalid.index[0], column] = 1.5
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        with pytest.raises(ValueError, match=f"{column} values must be finite integer codes"):
            model.fit(
                invalid,
                state="state",
                action="action",
                id="agent_id",
                transitions=transitions,
            )

    def test_diagnostics_and_normalization(self, fitted_model_state_action):
        model = fitted_model_state_action
        assert model.diagnostics_["transition_orientation"] == ("(n_actions, n_states, n_states)")
        assert model.n_observations_ == 600
        np.testing.assert_allclose(model.reward_[:, 0], 0.0, atol=1e-8)
        assert model.termination_reason_ is not None
        assert model.bellman_residual_ is not None

    def test_action_mask_counterfactual(self, fitted_model_state_action):
        mask = np.ones((_N_STATES, _N_ACTIONS), dtype=bool)
        mask[:, 1] = False
        result = fitted_model_state_action.counterfactual(action_mask=mask)
        np.testing.assert_allclose(result.policy[:, 1], 0.0, atol=1e-8)
        assert result.metadata["changed_primitive"] == "action_availability"

    def test_reward_counterfactual(self, fitted_model_state_action):
        delta = np.zeros((_N_STATES, _N_ACTIONS))
        delta[:, 1] = 0.5
        result = fitted_model_state_action.counterfactual(reward_delta=delta)
        assert np.max(np.abs(result.policy_change)) > 0
        assert result.metadata["changed_primitive"] == "reward"

    def test_transition_counterfactual(self, fitted_model_state_action, transitions):
        changed = np.asarray(transitions)[::-1].copy()
        result = fitted_model_state_action.counterfactual(transitions=changed)
        assert np.max(np.abs(result.policy_change)) > 0
        assert result.metadata["changed_primitive"] == "transitions"

    def test_counterfactual_requires_one_change(self, fitted_model_state_action):
        with pytest.raises(ValueError, match="exactly one"):
            fitted_model_state_action.counterfactual()
        with pytest.raises(ValueError, match="exactly one"):
            fitted_model_state_action.counterfactual(
                reward_delta=np.zeros((_N_STATES, _N_ACTIONS)),
                transitions=np.asarray(fitted_model_state_action.transitions_),
            )

    def test_counterfactual_cannot_reenable_masked_action(self, gridworld_df, transitions):
        restricted_data = gridworld_df.loc[gridworld_df["action"] != 1].copy()
        mask = np.ones((_N_STATES, _N_ACTIONS), dtype=bool)
        mask[:, 1] = False
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state_action",
            max_epochs=1,
        )
        model.fit(
            restricted_data,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
            action_mask=mask,
        )
        with pytest.raises(ValueError, match="may only remove actions"):
            model.counterfactual(action_mask=np.ones_like(mask))

    def test_simulation_is_seeded(self, fitted_model_state_action):
        first = fitted_model_state_action.simulate(3, 4, seed=91)
        second = fitted_model_state_action.simulate(3, 4, seed=91)
        assert first.num_observations == 12
        np.testing.assert_array_equal(first.all_states, second.all_states)
        np.testing.assert_array_equal(first.all_actions, second.all_actions)

    def test_exhausting_epochs_is_not_automatic_convergence(self, gridworld_df, transitions):
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            max_epochs=1,
            occupancy_tol=1e-12,
        )
        model.fit(
            gridworld_df,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
        )
        assert model.converged_ is False
        assert model.termination_reason_ == "occupancy_tolerance_not_met"


# ---------------------------------------------------------------------------
# Custom reward architectures via the reward_network hook
# ---------------------------------------------------------------------------


class _CustomTanhMLP(eqx.Module):
    """Custom state-reward net, full (S, state_dim) -> (S,) contract."""

    layers: list

    def __init__(self, state_dim, n_actions, key, width=16):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(state_dim, width, key=k1),
            eqx.nn.Linear(width, 1, key=k2),
        ]

    def __call__(self, X):
        def one(x):
            return self.layers[1](jax.nn.tanh(self.layers[0](x))).squeeze(-1)

        return jax.vmap(one)(X)


class _CustomStateActionNet(eqx.Module):
    """Custom net returning the full (S, A) reward matrix."""

    lin: eqx.nn.Linear

    def __init__(self, state_dim, n_actions, key):
        self.lin = eqx.nn.Linear(state_dim, n_actions, key=key)

    def __call__(self, X):
        return jax.vmap(self.lin)(X)


class _CustomConvReward(eqx.Module):
    """CoordConv-style reward field over a square grid.

    Exercises the canonical-dtype cast: Conv2d requires matching dtypes and does
    not auto-promote, unlike the Linear matmul path.
    """

    convs: list
    g: int = eqx.field(static=True)

    def __init__(self, state_dim, n_actions, key, g):
        k1, k2 = jax.random.split(key)
        self.convs = [
            eqx.nn.Conv2d(state_dim, 4, 3, padding=1, key=k1),
            eqx.nn.Conv2d(4, 1, 3, padding=1, key=k2),
        ]
        self.g = g

    def __call__(self, X):
        img = X.T.reshape(X.shape[1], self.g, self.g)
        img = jax.nn.relu(self.convs[0](img))
        return self.convs[1](img).reshape(-1)


class TestCustomArchitecture:
    """The reward_network hook accepts custom reward architectures."""

    def test_custom_mlp_state_reward(self, gridworld_df, transitions):
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state",
            reward_network=lambda sd, na, key: _CustomTanhMLP(sd, na, key),
            max_epochs=60,
            lr=1e-2,
            verbose=False,
        )
        model.fit(
            gridworld_df,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
            features=_make_features(),
        )
        assert model.reward_.shape == (_N_STATES,)
        assert model.policy_.shape == (_N_STATES, _N_ACTIONS)
        assert np.all(np.isfinite(model.reward_))
        assert model.coef_.shape == (2,)
        assert isinstance(model.projection_r2_, float)

    def test_custom_state_action_matrix(self, gridworld_df, transitions):
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state_action",
            reward_network=lambda sd, na, key: _CustomStateActionNet(sd, na, key),
            max_epochs=60,
            lr=1e-2,
            verbose=False,
        )
        model.fit(
            gridworld_df,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
        )
        assert model.reward_.shape == (_N_STATES, _N_ACTIONS)
        assert np.all(np.isfinite(model.reward_))

    def test_custom_conv_net_runs(self):
        n, g = 9, 3
        df = _make_gridworld_data(n_states=n, n_actions=_N_ACTIONS, n_individuals=15, n_periods=20)
        T = _make_gridworld_transitions(n, _N_ACTIONS)
        enc = lambda s: jnp.stack(  # noqa: E731
            [
                (jnp.asarray(s) // g).astype(jnp.float32) / g,
                (jnp.asarray(s) % g).astype(jnp.float32) / g,
            ],
            axis=-1,
        )
        model = MCEIRLNeural(
            n_states=n,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state",
            reward_network=lambda sd, na, key: _CustomConvReward(sd, na, key, g=g),
            max_epochs=30,
            lr=5e-3,
            state_encoder=enc,
            state_dim=2,
            verbose=False,
        )
        model.fit(df, state="state", action="action", id="agent_id", transitions=T)
        assert model.reward_.shape == (n,)
        assert np.all(np.isfinite(model.reward_))
