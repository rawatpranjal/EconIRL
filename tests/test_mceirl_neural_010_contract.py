"""EconIRL 0.1.0 public-contract tests for Neural MCE-IRL."""

from __future__ import annotations

import inspect
import pickle
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from econirl import MCEIRLNeural
from econirl.core.types import Panel, Trajectory
from econirl.inference import FunctionalBootstrapResult


def _transitions() -> np.ndarray:
    transitions = np.zeros((2, 2, 2), dtype=np.float32)
    transitions[:, 0, 0] = 1.0
    transitions[:, 1, 1] = 1.0
    return transitions


def _panel(seed: int = 17, n_individuals: int = 120) -> Panel:
    rng = np.random.default_rng(seed)
    trajectories = []
    for individual in range(n_individuals):
        state = individual % 2
        probability = 0.75 if state == 0 else 0.35
        action = int(rng.binomial(1, probability))
        trajectories.append(
            Trajectory(
                states=jnp.array([state]),
                actions=jnp.array([action]),
                next_states=jnp.array([state]),
                individual_id=individual,
            )
        )
    return Panel(trajectories)


def _model(**kwargs: object) -> MCEIRLNeural:
    defaults = dict(
        n_states=2,
        n_actions=2,
        discount=0.9,
        reward_hidden_dim=8,
        reward_num_layers=1,
        max_epochs=150,
        lr=0.05,
        occupancy_tol=0.04,
        patience=50,
        seed=4,
    )
    defaults.update(kwargs)
    return MCEIRLNeural(**defaults)


@pytest.fixture(scope="module")
def fitted_bootstrap() -> MCEIRLNeural:
    return _model(se_method="bootstrap", n_bootstrap=3, se_seed=29).fit(
        _panel(), transitions=_transitions()
    )


def test_fit_uses_keyword_only_workflow_arguments() -> None:
    signature = inspect.signature(MCEIRLNeural.fit)
    assert signature.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("state", "action", "id", "features", "transitions", "action_mask", "context"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_states": 0}, "n_states"),
        ({"n_actions": 1}, "n_actions"),
        ({"discount": 1.0}, "discount"),
        ({"reward_hidden_dim": 0}, "network dimensions"),
        ({"lr": 0.0}, "lr"),
        ({"occupancy_tol": float("nan")}, "tolerances"),
        ({"inner_solver": "unknown"}, "inner_solver"),
        ({"se_method": "unknown"}, "se_method"),
        ({"n_bootstrap": -1}, "n_bootstrap"),
        ({"se_method": "bootstrap", "n_bootstrap": 1}, "n_bootstrap"),
    ],
)
def test_constructor_rejects_invalid_inference_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MCEIRLNeural(**kwargs)


def test_custom_state_encoder_requires_declared_dimension() -> None:
    with pytest.raises(ValueError, match="state_dim"):
        MCEIRLNeural(state_encoder=lambda states: states)


def test_unfitted_state_and_capabilities_are_explicit() -> None:
    model = MCEIRLNeural()
    assert model.is_fitted_ is False
    assert model.result_ is None
    assert isinstance(model.capabilities_, MappingProxyType)
    assert set(model.capabilities_) == {
        "inference",
        "prediction",
        "simulation",
        "counterfactual",
        "serialization",
    }
    assert all(item["status"] == "supported" for item in model.capabilities_.values())
    with pytest.raises(TypeError):
        model.capabilities_["inference"] = {"status": "unsupported"}
    with pytest.raises(RuntimeError, match="not fitted"):
        model.conf_int()


def test_bootstrap_failure_resets_fit_and_records_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_bootstrap(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced bootstrap failure")

    monkeypatch.setattr(MCEIRLNeural, "_run_bootstrap", fail_bootstrap)
    model = _model(se_method="bootstrap", n_bootstrap=3)

    with pytest.raises(RuntimeError, match="bootstrap inference failed"):
        model.fit(_panel(), transitions=_transitions())

    assert model.is_fitted_ is False
    assert model.policy_ is None
    assert model.result_ is None
    assert model.termination_reason_ == "bootstrap_failure"
    assert model.failure_reason_ == "RuntimeError: forced bootstrap failure"
    assert model.fit_time_ is not None and model.fit_time_ > 0


def test_fitted_state_and_summary_follow_shared_contract(fitted_bootstrap: MCEIRLNeural) -> None:
    model = fitted_bootstrap
    assert model.is_fitted_ is True
    assert model.result_ is model._result
    assert np.isfinite(model.result_.log_likelihood)
    assert model.n_iter_ == model.n_epochs_
    assert model.fit_time_ > 0
    assert model.failure_reason_ is None
    assert set(model.diagnostics_) == {
        "data",
        "identification",
        "transitions",
        "optimization",
    }
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
    summary = model.summary()
    positions = [summary.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Network weights are not economic parameters" in summary


def test_bootstrap_exposes_only_reward_and_policy_functionals(
    fitted_bootstrap: MCEIRLNeural,
) -> None:
    model = fitted_bootstrap
    assert isinstance(model.bootstrap_, FunctionalBootstrapResult)
    bootstrap = model.bootstrap_
    assert bootstrap.method == "pairs_cluster"
    assert bootstrap.unit == "individual_trajectory"
    assert bootstrap.n_requested == 3
    assert bootstrap.n_successful >= 2
    assert bootstrap.reward_draws.shape == (bootstrap.n_successful, 2, 2)
    assert bootstrap.policy_draws.shape == (bootstrap.n_successful, 2, 2)
    assert all(name.startswith(("reward[", "policy[")) for name in bootstrap.estimand_names)
    assert all("a=0" not in name for name in bootstrap.estimand_names if name.startswith("reward"))
    assert model.se_ is None
    assert model.pvalues_ is None
    intervals = model.conf_int()
    assert set(intervals) == set(bootstrap.estimand_names)
    assert all(np.isfinite(bound) for interval in intervals.values() for bound in interval)
    narrower = model.conf_int(alpha=0.20)
    assert all(
        intervals[name][0] <= narrower[name][0] <= narrower[name][1] <= intervals[name][1]
        for name in intervals
    )
    for alpha in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            model.conf_int(alpha=alpha)


@pytest.mark.parametrize("family", ["reward", "transitions", "action_mask"])
def test_counterfactual_reports_bootstrap_intervals(
    fitted_bootstrap: MCEIRLNeural, family: str
) -> None:
    if family == "reward":
        delta = np.zeros((2, 2))
        delta[:, 1] = 0.2
        result = fitted_bootstrap.counterfactual(reward_delta=delta)
    elif family == "transitions":
        changed = _transitions()
        changed[:, 0] = np.array([0.2, 0.8])
        result = fitted_bootstrap.counterfactual(transitions=changed)
    else:
        mask = np.ones((2, 2), dtype=bool)
        mask[1, 1] = False
        result = fitted_bootstrap.counterfactual(action_mask=mask)
    receipt = result.metadata["bootstrap_intervals"]
    assert receipt["n_successful"] == fitted_bootstrap.bootstrap_.n_successful
    assert len(receipt["mean_policy_tv"]) == 2
    assert len(receipt["mean_value_change"]) == 2
    assert receipt["mean_policy_tv"][1] >= receipt["mean_policy_tv"][0]


def test_pickle_round_trip_preserves_supported_outputs(
    fitted_bootstrap: MCEIRLNeural,
) -> None:
    before = fitted_bootstrap.predict_proba(np.array([0, 1]))
    restored = pickle.loads(pickle.dumps(fitted_bootstrap))
    assert restored.econirl_version_ == fitted_bootstrap.econirl_version_
    assert restored.summary() == fitted_bootstrap.summary()
    np.testing.assert_allclose(restored.predict_proba(np.array([0, 1])), before)
    assert restored.conf_int() == pytest.approx(fitted_bootstrap.conf_int())


def test_seeded_bootstrap_is_exactly_reproducible(fitted_bootstrap: MCEIRLNeural) -> None:
    repeated = _model(se_method="bootstrap", n_bootstrap=3, se_seed=29).fit(
        _panel(), transitions=_transitions()
    )
    np.testing.assert_array_equal(
        repeated.bootstrap_.estimates,
        fitted_bootstrap.bootstrap_.estimates,
    )
