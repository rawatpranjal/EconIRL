"""EconIRL 0.1.0 public-contract tests for MCE-IRL."""

from __future__ import annotations

import inspect
import pickle
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from econirl import MCEIRL
from econirl.core.tasks import MCEIRLTask
from econirl.core.types import Panel, Trajectory
from econirl.transitions import DeterministicTransitions


def _panel(seed: int = 73, n_individuals: int = 80) -> Panel:
    rng = np.random.default_rng(seed)
    probability = 1.0 / (1.0 + np.exp(-0.4))
    actions = rng.binomial(1, probability, size=n_individuals)
    return Panel(
        [
            Trajectory(
                states=jnp.array([0]),
                actions=jnp.array([action]),
                next_states=jnp.array([1]),
                individual_id=individual,
                metadata={"task_id": "binary"},
            )
            for individual, action in enumerate(actions)
        ]
    )


def _transitions() -> DeterministicTransitions:
    return DeterministicTransitions(
        next_state=np.array([[1, 1], [1, -1]]),
        valid_action=np.array([[True, True], [True, False]]),
    )


def _tasks() -> list[MCEIRLTask]:
    return [
        MCEIRLTask(
            task_id="binary",
            initial_state=0,
            terminal_states=np.array([1]),
            horizon=1,
        )
    ]


def _features() -> np.ndarray:
    features = np.zeros((2, 2, 1), dtype=np.float32)
    features[0, 1, 0] = 1.0
    return features


@pytest.fixture(scope="module")
def fitted_mce() -> MCEIRL:
    model = MCEIRL(
        n_states=2,
        n_actions=2,
        discount=1.0,
        horizon=1,
        feature_matrix=_features(),
        feature_names=["action_one"],
        se_method="asymptotic",
    )
    return model.fit(_panel(), transitions=_transitions(), tasks=_tasks())


def test_fit_uses_keyword_only_workflow_arguments() -> None:
    signature = inspect.signature(MCEIRL.fit)
    assert signature.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in (
        "state",
        "action",
        "id",
        "next_state",
        "transitions",
        "reward",
        "tasks",
        "task",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_states": 0}, "n_states"),
        ({"n_actions": 1}, "n_actions"),
        ({"discount": 0.0}, "discount"),
        ({"discount": 1.0}, "finite horizon"),
        ({"horizon": 0}, "horizon"),
        ({"n_bootstrap": -1}, "n_bootstrap"),
        ({"se_method": "unknown"}, "se_method"),
        ({"se_method": "bootstrap", "n_bootstrap": 1}, "n_bootstrap"),
        ({"l2_regularization": -1.0}, "l2_regularization"),
    ],
)
def test_constructor_rejects_invalid_shared_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MCEIRL(**kwargs)


def test_unfitted_state_is_explicit() -> None:
    model = MCEIRL()
    assert model.is_fitted_ is False
    assert model.result_ is None
    assert "Not fitted" in model.summary()


def test_fitted_state_and_diagnostics_follow_contract(fitted_mce: MCEIRL) -> None:
    assert fitted_mce.is_fitted_ is True
    assert fitted_mce.result_ is fitted_mce._result
    assert fitted_mce.termination_reason_
    assert fitted_mce.failure_reason_ is None
    assert fitted_mce.n_iter_ > 0
    assert fitted_mce.fit_time_ > 0
    assert fitted_mce.n_observations_ == 80
    assert set(fitted_mce.diagnostics_) == {
        "data",
        "identification",
        "transitions",
        "optimization",
    }
    assert fitted_mce.diagnostics_["identification"]["contrast_rank"] == 1
    assert fitted_mce.diagnostics_["identification"]["verdict"] == "identified"
    assert fitted_mce.diagnostics_["transitions"]["source"] == ("compiled deterministic task views")


def test_capabilities_are_complete_and_read_only(fitted_mce: MCEIRL) -> None:
    capabilities = fitted_mce.capabilities_
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


def test_summary_has_manager_facing_sections_in_order(fitted_mce: MCEIRL) -> None:
    report = fitted_mce.summary()
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
    assert "Observations: 80" in report
    assert "Reward levels are normalized" in report


@pytest.mark.parametrize(
    "states",
    [
        np.array([-1]),
        np.array([2]),
        np.array([0.5]),
        np.array([[0, 1]]),
    ],
)
def test_predict_proba_rejects_invalid_state_codes(fitted_mce: MCEIRL, states: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fitted_mce.predict_proba(states, task_id="binary")


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_conf_int_rejects_invalid_alpha(fitted_mce: MCEIRL, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        fitted_mce.conf_int(alpha=alpha)


def test_rank_deficient_action_features_stop_before_optimization() -> None:
    features = np.zeros((2, 2, 2), dtype=float)
    features[0, 1, 0] = 1.0
    features[:, :, 1] = 1.0
    model = MCEIRL(
        n_states=2,
        n_actions=2,
        discount=1.0,
        horizon=1,
        feature_matrix=features,
        feature_names=["action_one", "constant"],
        se_method="asymptotic",
    )
    with pytest.raises(ValueError, match="action-contrast rank"):
        model.fit(_panel(), transitions=_transitions(), tasks=_tasks())
    assert model.is_fitted_ is False


def test_dense_transition_validation_is_fail_closed() -> None:
    transitions = np.zeros((2, 2, 2), dtype=float)
    transitions[:, :, 0] = 0.6
    model = MCEIRL(
        n_states=2,
        n_actions=2,
        discount=0.9,
        feature_matrix=_features(),
        se_method="asymptotic",
    )
    with pytest.raises(ValueError, match="sum to one"):
        model.fit(_panel(), transitions=transitions)
    assert model.is_fitted_ is False


def test_counterfactual_withholds_reward_level_welfare(fitted_mce: MCEIRL) -> None:
    result = fitted_mce.counterfactual(params={"action_one": 0.8})
    assert result.welfare_change is None
    assert result.metadata["reward_level_identified"] is False
    assert "not identified in levels" in result.summary(reward_level_identified=False)


def test_pickle_round_trip_preserves_supported_results(fitted_mce: MCEIRL) -> None:
    states = np.array([0])
    before_summary = fitted_mce.summary()
    before_predictions = fitted_mce.predict_proba(states, task_id="binary")
    before_cf = fitted_mce.counterfactual(params={"action_one": 0.8})

    restored = pickle.loads(pickle.dumps(fitted_mce))

    assert restored.econirl_version_ == fitted_mce.econirl_version_
    assert restored.summary() == before_summary
    np.testing.assert_allclose(restored.predict_proba(states, task_id="binary"), before_predictions)
    after_cf = restored.counterfactual(params={"action_one": 0.8})
    np.testing.assert_allclose(after_cf.counterfactual_policy, before_cf.counterfactual_policy)


def test_bootstrap_result_exposes_draws_and_percentile_intervals() -> None:
    model = MCEIRL(
        n_states=2,
        n_actions=2,
        discount=1.0,
        horizon=1,
        feature_matrix=_features(),
        feature_names=["action_one"],
        se_method="bootstrap",
        n_bootstrap=3,
        se_seed=19,
    )
    model.fit(_panel(n_individuals=120), transitions=_transitions(), tasks=_tasks())

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
    assert model.conf_int()["action_one"] == pytest.approx(tuple(model.bootstrap_.intervals[0]))
    assert "Bootstrap successful draws:" in model.summary()
