"""Tests for NeuralGLADIUS estimator.

Tests cover:
- Basic fit without features
- Fit with feature projection
- Context conditioning
- Policy and value shapes
- EstimatorProtocol conformance
- predict_proba and predict_reward
- Confidence intervals
- Summary output
- Custom encoders
- Projection R-squared range
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from econirl.core.reward_spec import RewardSpec
from econirl.estimators.neural_gladius import NeuralGLADIUS
from econirl.estimators.protocol import EstimatorProtocol

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_data():
    """Small 10-state, 3-action dataset with 50 agents x 20 periods."""
    np.random.seed(42)
    records = []
    for uid in range(50):
        state = 0
        dest = np.random.randint(5, 10)  # destination context
        for t in range(20):
            action = np.random.randint(3)
            next_state = min(state + action, 9)
            records.append(
                {
                    "id": uid,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "dest": dest,
                }
            )
            state = next_state
    return pd.DataFrame(records)


@pytest.fixture
def small_features():
    """Feature matrix (10 states, 3 actions, 2 features) for projection."""
    n_states, n_actions, n_features = 10, 3, 2
    features = jnp.zeros((n_states, n_actions, n_features))
    # Feature 0: state index (normalized)
    for s in range(n_states):
        features = features.at[s, :, 0].set(-s / 9.0)
    # Feature 1: action cost (action 0 = 0, action 1 = -0.5, action 2 = -1)
    for a in range(n_actions):
        features = features.at[:, a, 1].set(-a / 2.0)
    return RewardSpec(features, names=["state_cost", "action_cost"])


@pytest.fixture
def fitted_model(small_data):
    """A fitted NeuralGLADIUS model without features."""
    model = NeuralGLADIUS(
        n_actions=3,
        discount=0.95,
        max_epochs=30,
        patience=10,
        q_hidden_dim=32,
        q_num_layers=2,
        ev_hidden_dim=32,
        ev_num_layers=2,
        batch_size=256,
    )
    model.fit(
        data=small_data,
        state="state",
        action="action",
        id="id",
    )
    return model


@pytest.fixture
def fitted_model_with_features(small_data, small_features):
    """A fitted NeuralGLADIUS model with feature projection."""
    model = NeuralGLADIUS(
        n_actions=3,
        discount=0.95,
        max_epochs=30,
        patience=10,
        q_hidden_dim=32,
        q_num_layers=2,
        ev_hidden_dim=32,
        ev_num_layers=2,
        batch_size=256,
    )
    model.fit(
        data=small_data,
        state="state",
        action="action",
        id="id",
        features=small_features,
    )
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFitBasic:
    """Test basic fit without features."""

    def test_fit_runs_without_error(self, fitted_model):
        """Fit should complete without raising."""
        assert fitted_model.policy_ is not None

    def test_params_none_without_features(self, fitted_model):
        """Without features, params_ should be None."""
        assert fitted_model.params_ is None
        assert fitted_model.se_ is None
        assert fitted_model.pvalues_ is None
        assert fitted_model.projection_r2_ is None
        assert fitted_model.coef_ is None

    def test_converged_is_set(self, fitted_model):
        """converged_ should be a boolean after fitting."""
        assert isinstance(fitted_model.converged_, bool)

    def test_n_epochs_is_set(self, fitted_model):
        """n_epochs_ should be a positive integer after fitting."""
        assert isinstance(fitted_model.n_epochs_, int)
        assert fitted_model.n_epochs_ > 0


class TestFitWithFeatures:
    """Test fit with feature projection."""

    def test_params_has_correct_keys(self, fitted_model_with_features):
        """params_ should have the correct feature names."""
        assert fitted_model_with_features.params_ is not None
        assert set(fitted_model_with_features.params_.keys()) == {
            "state_cost",
            "action_cost",
        }

    def test_se_has_correct_keys(self, fitted_model_with_features):
        """se_ should have the same keys as params_."""
        assert fitted_model_with_features.se_ is not None
        assert set(fitted_model_with_features.se_.keys()) == {
            "state_cost",
            "action_cost",
        }

    def test_pvalues_has_correct_keys(self, fitted_model_with_features):
        """pvalues_ should have the same keys as params_."""
        assert fitted_model_with_features.pvalues_ is not None
        assert set(fitted_model_with_features.pvalues_.keys()) == {
            "state_cost",
            "action_cost",
        }

    def test_projection_r2_is_float(self, fitted_model_with_features):
        """projection_r2_ should be a float."""
        assert isinstance(fitted_model_with_features.projection_r2_, float)

    def test_coef_array(self, fitted_model_with_features):
        """coef_ should be a numpy array with correct length."""
        assert fitted_model_with_features.coef_ is not None
        assert isinstance(fitted_model_with_features.coef_, np.ndarray)
        assert len(fitted_model_with_features.coef_) == 2


class TestFitWithContext:
    """Test fit with context conditioning."""

    def test_context_column(self, small_data):
        """Passing a context column name should work."""
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=20,
            patience=10,
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
            batch_size=256,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            context="dest",
        )
        assert model.policy_ is not None

    def test_context_tensor(self, small_data):
        """Passing a context tensor should work."""
        N = len(small_data)
        ctx = jnp.array(np.random.randint(0, 5, (N,)))
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=20,
            patience=10,
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
            batch_size=256,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            context=ctx,
        )
        assert model.policy_ is not None


class TestPolicyAndValue:
    """Test policy and value function shapes."""

    def test_policy_shape(self, fitted_model):
        """Policy should be (n_states, n_actions)."""
        assert fitted_model.policy_.shape == (10, 3)

    def test_policy_sums_to_one(self, fitted_model):
        """Each row of policy should sum to 1."""
        row_sums = fitted_model.policy_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_policy_nonnegative(self, fitted_model):
        """Policy probabilities should be non-negative."""
        assert (fitted_model.policy_ >= 0).all()

    def test_value_shape(self, fitted_model):
        """Value function should be (n_states,)."""
        assert fitted_model.value_.shape == (10,)


class TestProtocol:
    """Test EstimatorProtocol conformance."""

    def test_protocol_conformance(self, fitted_model_with_features):
        """Model should satisfy EstimatorProtocol."""
        assert isinstance(fitted_model_with_features, EstimatorProtocol)


class TestPredictProba:
    """Test predict_proba method."""

    def test_correct_shape(self, fitted_model):
        """predict_proba should return (len(states), n_actions)."""
        states = np.array([0, 3, 7])
        proba = fitted_model.predict_proba(states)
        assert proba.shape == (3, 3)

    def test_sums_to_one(self, fitted_model):
        """Probabilities should sum to 1."""
        states = np.array([0, 5, 9])
        proba = fitted_model.predict_proba(states)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_not_fitted_raises(self):
        """predict_proba should raise if model is not fitted."""
        model = NeuralGLADIUS()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0]))


class TestPredictReward:
    """Test predict_reward method."""

    def test_returns_correct_shape(self, fitted_model):
        """predict_reward should return (N,) tensor."""
        states = jnp.array([0, 3, 7])
        actions = jnp.array([0, 1, 2])
        rewards = fitted_model.predict_reward(states, actions)
        assert rewards.shape == (3,)

    def test_returns_array(self, fitted_model):
        """predict_reward should return an array."""
        states = jnp.array([0, 5])
        actions = jnp.array([1, 0])
        rewards = fitted_model.predict_reward(states, actions)
        assert hasattr(rewards, "shape")

    def test_with_context(self, fitted_model):
        """The paper path accepts the baseline context only."""
        states = jnp.array([0, 5])
        actions = jnp.array([1, 0])
        contexts = jnp.array([0, 0])
        rewards = fitted_model.predict_reward(states, actions, contexts)
        assert rewards.shape == (2,)

    def test_not_fitted_raises(self):
        """predict_reward should raise if not fitted."""
        model = NeuralGLADIUS()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_reward(jnp.array([0]), jnp.array([0]))


class TestConfInt:
    """Sampling intervals are gated on a completed trajectory bootstrap."""

    def test_rejects_descriptive_projection_intervals(self, fitted_model_with_features):
        """Projection SEs must not be presented as sampling uncertainty."""
        with pytest.raises(NotImplementedError, match="compute_se=True"):
            fitted_model_with_features.conf_int(alpha=0.05)

    def test_raises_without_features(self, fitted_model):
        """A fit without a bootstrap has no sampling intervals."""
        with pytest.raises(NotImplementedError, match="compute_se=True"):
            fitted_model.conf_int()


class TestSummary:
    """Test summary output."""

    def test_returns_nonempty_string(self, fitted_model_with_features):
        """summary() should return a non-empty string."""
        s = fitted_model_with_features.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_contains_method_name(self, fitted_model_with_features):
        """Summary should mention the method name."""
        s = fitted_model_with_features.summary()
        assert "NeuralGLADIUS" in s

    def test_contains_r2(self, fitted_model_with_features):
        """Summary should include R2 info."""
        s = fitted_model_with_features.summary()
        assert "R2" in s

    def test_contains_parameter_names(self, fitted_model_with_features):
        """Summary should include the parameter names."""
        s = fitted_model_with_features.summary()
        assert "state_cost" in s
        assert "action_cost" in s

    def test_not_fitted_message(self):
        """Summary should indicate not fitted for unfitted model."""
        model = NeuralGLADIUS()
        s = model.summary()
        assert "Not fitted" in s

    def test_summary_without_features(self, fitted_model):
        """Summary without features should mention no projection."""
        s = fitted_model.summary()
        assert "No feature projection" in s or "None" in s


class TestTransitionMatrix:
    """Transitions are excluded from fitting but retained for planning."""

    def test_valid_transitions_are_stored(self, small_data):
        """Passing a valid transition tensor should retain it."""
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=10,
            patience=5,
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
        )
        transitions = np.repeat(np.eye(10)[None, :, :], 3, axis=0)
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            transitions=transitions,
        )
        assert model.policy_ is not None
        np.testing.assert_array_equal(model.transitions_, transitions)


class TestCustomEncoders:
    """Test custom state and context encoders."""

    def test_custom_context_encoder(self, small_data):
        """Custom context encoder should be used."""

        # One-hot context encoder for 10 possible contexts
        def ctx_encoder(c):
            return jax.nn.one_hot(c, 10).astype(jnp.float32)

        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
            context_encoder=ctx_encoder,
            context_dim=10,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            context="dest",
        )
        assert model.policy_ is not None

    def test_custom_state_encoder(self, small_data):
        """Custom state encoder should be used."""

        # One-hot state encoder
        def state_encoder(s):
            return jax.nn.one_hot(s, 10).astype(jnp.float32)

        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
            state_encoder=state_encoder,
            state_dim=10,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
        )
        assert model.policy_ is not None


class TestProjectionR2:
    """Test projection R-squared range."""

    def test_r2_in_valid_range(self, fitted_model_with_features):
        """Projection R2 should be in a reasonable range.

        R2 can technically be negative for very poor fits, but for
        reasonable features it should be in [-inf, 1]. We check it
        is at most 1.0.
        """
        r2 = fitted_model_with_features.projection_r2_
        assert r2 <= 1.0 + 1e-6

    def test_r2_is_finite(self, fitted_model_with_features):
        """Projection R2 should be finite."""
        assert np.isfinite(fitted_model_with_features.projection_r2_)


class TestRepr:
    """Test __repr__ output."""

    def test_unfitted_repr(self):
        """Unfitted model repr should say fitted=False."""
        model = NeuralGLADIUS(n_actions=5)
        r = repr(model)
        assert "fitted=False" in r
        assert "n_actions=5" in r

    def test_fitted_repr(self, fitted_model):
        """Fitted model repr should say fitted=True."""
        r = repr(fitted_model)
        assert "fitted=True" in r


class TestConvergedFlag:
    """converged_ must reflect early stopping, not exhausting max_epochs (#1)."""

    def test_not_converged_when_max_epochs_exhausted(self, small_data):
        """A run that exhausts max_epochs without early stopping is not converged."""
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=4,
            patience=1000,  # never trips early stopping
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
        )
        model.fit(data=small_data, state="state", action="action", id="id")
        assert model.converged_ is False
        assert model.n_epochs_ == 4


class TestSummaryObsCount:
    """summary() reports the observation count, not the state count (#8)."""

    def test_reports_panel_rows(self, fitted_model_with_features, small_data):
        s = fitted_model_with_features.summary()
        assert f"Observations:    {len(small_data)}" in s
        # n_states (10) must not be reported as the observation count.
        assert "Observations:    10\n" not in s


class TestPredictFromFeaturesGuard:
    """predict_*_from_features must reject raw features of the wrong width (#3/#9)."""

    def test_q_from_features_wrong_width_raises(self, fitted_model):
        # Default encoder has state_dim=1; a raw 3-wide feature vector is invalid.
        with pytest.raises(ValueError, match="encoder space"):
            fitted_model.predict_q_from_features(np.zeros((4, 3)))

    def test_q_from_features_correct_width_ok(self, fitted_model):
        q = fitted_model.predict_q_from_features(np.zeros((4, 1)))
        assert q.shape == (4, 3)

    def test_reward_from_features_wrong_width_raises(self, fitted_model):
        with pytest.raises(ValueError, match="encoder space"):
            fitted_model.predict_reward_from_features(np.zeros((4, 3)), np.zeros(4))


class TestPredictProbaContext:
    """The paper path exposes its non-contextual boundary explicitly."""

    def test_context_zero_matches_stored_policy(self, fitted_model):
        states = np.array([0, 3, 7])
        recomputed = fitted_model.predict_proba(states, context=0)
        np.testing.assert_allclose(recomputed, fitted_model.policy_[states], atol=1e-5)

    def test_nonzero_context_requires_context_aware_fit(self, fitted_model):
        states = np.array([0, 5, 9])
        with pytest.raises(NotImplementedError, match="not context-conditioned"):
            fitted_model.predict_proba(states, context=1)


class TestAnchorValidation:
    """Anchor argument validation and the dead-anchor warning (#2)."""

    def test_anchor_action_without_rewards_warns(self, small_data):
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=4,
            patience=2,
            q_hidden_dim=8,
            q_num_layers=1,
            ev_hidden_dim=8,
            ev_num_layers=1,
            anchor_action=0,
        )
        with pytest.warns(UserWarning, match="anchor has no effect"):
            model.fit(data=small_data, state="state", action="action", id="id")

    def test_rewards_without_action_raises(self, small_data):
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=4,
            patience=2,
            q_hidden_dim=8,
            q_num_layers=1,
            ev_hidden_dim=8,
            ev_num_layers=1,
            anchor_rewards=tuple([0.0] * 10),
        )
        with pytest.raises(ValueError, match="anchor_action is None"):
            model.fit(data=small_data, state="state", action="action", id="id")

    def test_wrong_length_rewards_raises(self, small_data):
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=4,
            patience=2,
            q_hidden_dim=8,
            q_num_layers=1,
            ev_hidden_dim=8,
            ev_num_layers=1,
            anchor_action=0,
            anchor_rewards=(0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError, match="one known reward per state"):
            model.fit(data=small_data, state="state", action="action", id="id")


class TestTransitionsWarning:
    """transitions= is retained for planning and excluded from fitting."""

    def test_transitions_warns(self, small_data):
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=4,
            patience=2,
            q_hidden_dim=8,
            q_num_layers=1,
            ev_hidden_dim=8,
            ev_num_layers=1,
        )
        with pytest.warns(UserWarning, match="does not use a transition matrix"):
            model.fit(
                data=small_data,
                state="state",
                action="action",
                id="id",
                transitions=np.repeat(np.eye(10)[None, :, :], 3, axis=0),
            )


@pytest.mark.slow
class TestAnchorScaleRecovery:
    """NeuralGLADIUS recovers the reward on the near-identified ss-spine DGP.

    Guards the unweighted-NLL fix: with the old inverse-frequency class weighting
    the parameter cosine collapsed to ~0 here (ablation), so this test fails if
    the weighting is re-introduced as the default.
    """

    def test_anchor_recovers_scale(self):
        from econirl.environments.shapeshifter import (
            ShapeshifterConfig,
            ShapeshifterEnvironment,
        )
        from econirl.simulation.synthetic import simulate_panel

        env = ShapeshifterEnvironment(ShapeshifterConfig())
        true_theta = np.asarray(env.get_true_parameter_vector())
        phi = np.asarray(env.feature_matrix, dtype=np.float32)  # (S, A, K)
        K = phi.shape[-1]
        phi_state = phi[:, 1, :]
        panel = simulate_panel(env, n_individuals=600, n_periods=30, seed=0)

        model = NeuralGLADIUS(
            n_actions=env.num_actions,
            discount=0.95,
            state_encoder=lambda s: phi_state[np.asarray(s)],
            state_dim=K,
            anchor_action=0,
            anchor_rewards=tuple(float(x) for x in np.asarray(env.true_reward_matrix)[:, 0]),
            q_hidden_dim=128,
            q_num_layers=3,
            ev_hidden_dim=128,
            ev_num_layers=3,
            max_epochs=400,
            batch_size=512,
        )
        model.fit(panel, features=jnp.asarray(phi))
        theta_hat = np.asarray(model.coef_)
        cos = float(
            theta_hat @ true_theta / (np.linalg.norm(theta_hat) * np.linalg.norm(true_theta))
        )
        scale = float(np.linalg.norm(theta_hat) / np.linalg.norm(true_theta))
        assert cos >= 0.93, f"direction not recovered: cosine={cos:.3f}"
        assert 0.75 <= scale <= 1.3, f"scale not recovered: ratio={scale:.3f}"


class TestRawTensorFeatures:
    """Test using raw tensor features (not RewardSpec)."""

    def test_raw_tensor_features(self, small_data):
        """Should accept raw (S, A, K) tensor as features."""
        features = jnp.array(np.random.randn(10, 3, 2))
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
            feature_names=["f0", "f1"],
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            features=features,
        )
        assert model.params_ is not None
        assert set(model.params_.keys()) == {"f0", "f1"}

    def test_auto_feature_names(self, small_data):
        """Without feature_names, should auto-generate f0, f1, ..."""
        features = jnp.array(np.random.randn(10, 3, 3))
        model = NeuralGLADIUS(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            q_hidden_dim=16,
            q_num_layers=1,
            ev_hidden_dim=16,
            ev_num_layers=1,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            features=features,
        )
        assert set(model.params_.keys()) == {"f0", "f1", "f2"}
