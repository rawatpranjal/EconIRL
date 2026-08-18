"""Load-bearing 0.1.0 contract tests for public GLADIUS."""

from __future__ import annotations

import pickle
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from econirl.core.reward_spec import RewardSpec


def _tiny_contract_case():
    n_states = 4
    n_actions = 2
    rows = []
    for individual in range(8):
        state = individual % n_states
        for period in range(6):
            action = (individual + period) % n_actions
            next_state = state if action == 0 else (state + 1) % n_states
            rows.append(
                {
                    "id": individual,
                    "period": period,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                }
            )
            state = next_state
    data = pd.DataFrame(rows)
    feature_matrix = np.zeros((n_states, n_actions, 1), dtype=np.float32)
    feature_matrix[:, 1, 0] = 1.0
    features = RewardSpec(jnp.asarray(feature_matrix), names=["action_one"])
    transitions = np.zeros((n_actions, n_states, n_states), dtype=float)
    for state in range(n_states):
        transitions[0, state, state] = 1.0
        transitions[1, state, (state + 1) % n_states] = 1.0
    return data, features, transitions


@pytest.fixture(scope="module")
def fitted_contract_gladius():
    from econirl import GLADIUS

    data, features, transitions = _tiny_contract_case()
    model = GLADIUS(
        n_actions=2,
        discount=0.9,
        q_hidden_dim=8,
        q_num_layers=1,
        ev_hidden_dim=8,
        ev_num_layers=1,
        batch_size=16,
        max_epochs=4,
        patience=2,
        anchor_action=0,
        anchor_rewards=(0.0, 0.0, 0.0, 0.0),
        seed=7,
    )
    with pytest.warns((UserWarning, RuntimeWarning)):
        model.fit(
            data,
            state="state",
            action="action",
            id="id",
            features=features,
            transitions=transitions,
        )
    return model


def test_public_gladius_defaults_to_paper_reference_objective():
    """The canonical estimator must declare the paper-reference training path."""
    from econirl import GLADIUS, NeuralGLADIUS

    model = GLADIUS()

    assert GLADIUS is NeuralGLADIUS
    assert model.objective == "paper_minimax"
    assert model.network_mode == "shared_trunk"
    assert model.output_bias_init == 0.0
    assert model.gradient_clip_mode == "value"


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("state", -1, "state and next-state codes must be nonnegative"),
        ("action", 2, "action codes must lie"),
        ("state", 0.5, "state must contain integer codes"),
    ],
)
def test_public_fit_rejects_invalid_discrete_codes(column, value, message):
    from econirl import GLADIUS

    data, features, _ = _tiny_contract_case()
    if isinstance(value, float):
        data[column] = data[column].astype(float)
    data.loc[data.index[0], column] = value

    with pytest.raises(ValueError, match=message):
        GLADIUS(n_actions=2).fit(
            data,
            state="state",
            action="action",
            id="id",
            features=features,
        )


def test_paper_reference_bellman_terms_match_author_code():
    """The corrected Bellman residual must match the checked-in author code."""
    from econirl.estimation.gladius import _paper_reference_bellman_terms

    q_sa = jnp.asarray([-2.0, -1.5, 0.25])
    value_next = jnp.asarray([-1.0, -0.25, 0.75])
    continuation = jnp.asarray([-1.2, -0.5, 0.5])
    anchor_reward = jnp.asarray([-1.0, -1.0, -1.0])
    beta = 0.9

    actual = _paper_reference_bellman_terms(
        q_sa=q_sa,
        value_next=value_next,
        continuation=continuation,
        anchor_reward=anchor_reward,
        discount=beta,
    )
    td = q_sa - anchor_reward - beta * value_next
    deviation = value_next - continuation
    expected = np.abs(np.asarray(td**2 - beta**2 * deviation**2))

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_lower_level_default_decays_learning_rate_by_epoch():
    """Sample size must not silently accelerate the paper recipe's LR decay."""
    from econirl.estimation.gladius import GLADIUSConfig

    assert GLADIUSConfig().lr_decay_unit == "epoch"


def test_public_paper_fit_exposes_finite_identified_functionals(fitted_contract_gladius):
    model = fitted_contract_gladius

    assert model.objective_ == "paper_minimax"
    assert model.q_.shape == (4, 2)
    assert model.continuation_value_.shape == (4, 2)
    assert model.reward_.shape == (4, 2)
    assert model.policy_.shape == (4, 2)
    assert np.isfinite(model.q_).all()
    assert np.isfinite(model.continuation_value_).all()
    assert np.isfinite(model.reward_).all()
    np.testing.assert_allclose(model.policy_.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(model.reward_[:, 0].mean(), 0.0, atol=1e-6)
    assert model.diagnostics_["identification"]["anchor_available"] is True
    assert model.diagnostics_["optimization"]["termination_reason"]
    assert model.is_fitted_ is True
    assert model.n_iter_ == model.n_epochs_
    assert model.fit_time_ > 0
    assert model.n_observations_ == 48
    assert model.result_ is not None
    np.testing.assert_allclose(model.value_function_, model.value_)
    assert isinstance(model.capabilities_, MappingProxyType)
    assert set(model.capabilities_) == {
        "inference",
        "prediction",
        "simulation",
        "counterfactual",
        "serialization",
    }


def test_declared_state_universe_is_not_shrunk_by_partial_panel_coverage():
    from econirl import GLADIUS

    data, features, transitions = _tiny_contract_case()
    partial = data.loc[data["state"].isin([0, 1])].copy()
    model = GLADIUS(
        n_actions=2,
        discount=0.9,
        q_hidden_dim=4,
        q_num_layers=1,
        ev_hidden_dim=4,
        ev_num_layers=1,
        batch_size=16,
        max_epochs=1,
        patience=2,
        anchor_action=0,
        anchor_rewards=(0.0, 0.0, 0.0, 0.0),
        seed=19,
    )
    with pytest.warns((UserWarning, RuntimeWarning)):
        model.fit(
            partial,
            state="state",
            action="action",
            id="id",
            features=features,
            transitions=transitions,
        )

    assert model.q_.shape == (4, 2)
    assert model.diagnostics_["data"]["state_coverage"] == 0.5


def test_structural_counterfactual_requires_anchor():
    from econirl import GLADIUS

    data, features, transitions = _tiny_contract_case()
    model = GLADIUS(
        n_actions=2,
        discount=0.9,
        q_hidden_dim=8,
        q_num_layers=1,
        ev_hidden_dim=8,
        ev_num_layers=1,
        batch_size=16,
        max_epochs=2,
        patience=1,
    )
    with pytest.warns((UserWarning, RuntimeWarning)):
        model.fit(
            data,
            state="state",
            action="action",
            id="id",
            features=features,
            transitions=transitions,
        )
    with pytest.raises(NotImplementedError, match="unsupported without anchor_action"):
        model.counterfactual(reward_delta=np.zeros((4, 2)))


def test_anchored_counterfactual_re_solves_behavior(fitted_contract_gladius):
    delta = np.zeros((4, 2))
    delta[:, 1] = 1.0

    result = fitted_contract_gladius.counterfactual(reward_delta=delta)

    assert result.metadata["changed_primitive"] == "reward"
    assert np.max(np.abs(np.asarray(result.policy_change))) > 1e-3
    assert np.isfinite(result.welfare_change)


def test_pickle_round_trip_preserves_supported_results(fitted_contract_gladius):
    restored = pickle.loads(pickle.dumps(fitted_contract_gladius))

    states = np.arange(4)
    np.testing.assert_allclose(
        restored.predict_proba(states),
        fitted_contract_gladius.predict_proba(states),
    )
    np.testing.assert_allclose(restored.reward_, fitted_contract_gladius.reward_)
    assert restored.objective_ == "paper_minimax"


def test_prediction_rejects_invalid_state_codes(fitted_contract_gladius):
    with pytest.raises(ValueError, match="states must lie"):
        fitted_contract_gladius.predict_proba(np.array([-1, 0]))
    with pytest.raises(ValueError, match="states must lie"):
        fitted_contract_gladius.predict_proba(np.array([4]))


def test_simulation_uses_stored_planning_transitions(fitted_contract_gladius):
    first = fitted_contract_gladius.simulate(3, n_periods=5, seed=123)
    second = fitted_contract_gladius.simulate(3, n_periods=5, seed=123)

    np.testing.assert_array_equal(first.get_all_states(), second.get_all_states())
    np.testing.assert_array_equal(first.get_all_actions(), second.get_all_actions())
    assert first.num_individuals == 3
    assert first.num_observations == 15


def test_summary_has_the_frozen_manager_sections(fitted_contract_gladius):
    summary = fitted_contract_gladius.summary()
    headings = (
        "Estimator",
        "Data",
        "Model",
        "Pre-estimation checks",
        "Fit",
        "Outcome",
        "Uncertainty",
        "Limitations",
    )
    positions = [summary.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "stopping reason:" in summary


def test_trajectory_bootstrap_records_functional_draws():
    from econirl import GLADIUS
    from econirl.inference import FunctionalBootstrapResult

    data, features, _ = _tiny_contract_case()
    model = GLADIUS(
        n_actions=2,
        discount=0.9,
        q_hidden_dim=4,
        q_num_layers=1,
        ev_hidden_dim=4,
        ev_num_layers=1,
        batch_size=24,
        max_epochs=2,
        patience=1,
        anchor_action=0,
        anchor_rewards=(0.0, 0.0, 0.0, 0.0),
        compute_se=True,
        n_bootstrap=2,
        seed=11,
        se_seed=12,
    )
    with pytest.warns((UserWarning, RuntimeWarning)):
        model.fit(
            data,
            state="state",
            action="action",
            id="id",
            features=features,
        )

    assert isinstance(model.bootstrap_, FunctionalBootstrapResult)
    assert model.bootstrap_.n_successful == 2
    assert model.bootstrap_.reward_draws.shape == (2, 4, 2)
    assert model.bootstrap_.policy_draws.shape == (2, 4, 2)
    assert model.bootstrap_.estimates.shape == (2, 16)
    intervals = model.conf_int()
    assert len(intervals) == 16
    with pytest.raises(NotImplementedError, match="simulation requires a transition tensor"):
        model.simulate(1)
