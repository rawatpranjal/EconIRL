"""Drift guard + sanity for the forms capability registry (chunk F0)."""

from __future__ import annotations

import pytest

import econirl
from econirl.forms import CAPABILITIES, EstimatorCapability, Form, FormSpec

# Public classes that expose a fit/estimate API but are NOT registry choice
# estimators: a transition-matrix utility, the two deprecated legacy
# estimate()-only aliases of NFXP/CCP, and MPEC (exported at top level but
# deliberately kept out of CAPABILITIES so run_form does not surface it
# automatically; it runs via the direct .estimate() path -- see
# scripts/study_route_choice.py). Anything else with such an API must be
# registered (this is what makes the drift guard module-agnostic).
_NOT_REGISTRY_ESTIMATORS = {"TransitionEstimator", "NFXPEstimator", "CCPEstimator", "MPEC"}


def _public_choice_estimators() -> set[str]:
    """Every public class with a fit/estimate API, minus the explicit deny-list.

    Module-agnostic on purpose: a new estimator added anywhere (estimators/ or
    estimation/) is caught, not just ones under econirl.estimators.
    """
    out = set()
    for name in econirl.__all__:
        obj = getattr(econirl, name)
        if not isinstance(obj, type) or name in _NOT_REGISTRY_ESTIMATORS:
            continue
        if callable(getattr(obj, "fit", None)) or callable(getattr(obj, "estimate", None)):
            out.add(name)
    return out


def test_capabilities_cover_exactly_public_estimators():
    """Drift guard: add/remove a public estimator and this fails until the
    registry is updated. Catches both missing entries and orphans, in any module."""
    assert set(CAPABILITIES) == _public_choice_estimators()


def test_registered_names_are_public_classes():
    for name in CAPABILITIES:
        assert name in econirl.__all__, f"{name} registered but not exported by econirl"
        assert isinstance(getattr(econirl, name), type)


def test_capability_fields_sane():
    for name, c in CAPABILITIES.items():
        assert isinstance(c, EstimatorCapability)
        assert c.name == name
        assert c.path in ("estimate", "fit_features")
        assert c.family in ("structural", "behavioral")
        assert c.reward_forms and set(c.reward_forms) <= {"linear", "nonlinear", "neural"}
        # Invariant: the "other" group is exactly the not-shown set.
        assert (c.group == "other") == (not c.shown_in_studies)
        # Model-free estimators do not use transitions; model-based ones do.
        if c.model_free:
            assert not c.needs_transitions


def test_support_matrix_matches_verified_rows():
    for n in ("NFXP", "CCP", "NNES", "SEES", "TDCCP", "UFXP"):
        c = CAPABILITIES[n]
        assert c.family == "structural"
        assert c.path == "estimate"
        assert c.needs_transitions
        assert c.reward_forms == ("linear",)
        assert not c.model_free
    for n in ("MaxEntIRL", "MaxMarginIRL", "MCEIRL", "IQLearn"):
        c = CAPABILITIES[n]
        assert c.path == "estimate"
        assert c.needs_transitions
        assert c.reward_forms == ("linear",)
    for n in ("MCEIRLNeural", "NeuralUFXP"):
        c = CAPABILITIES[n]
        assert c.path == "fit_features"
        assert c.needs_transitions
        assert "neural" in c.reward_forms
        assert not c.model_free
    for n in ("NeuralAIRL", "AIRL"):
        c = CAPABILITIES[n]
        assert c.path == "fit_features"
        assert c.needs_transitions
        assert c.reward_forms == ("linear",)
        assert not c.model_free
    for n in ("NeuralGLADIUS", "GLADIUS"):
        c = CAPABILITIES[n]
        assert c.path == "fit_features"
        assert c.model_free
        assert not c.needs_transitions
    assert CAPABILITIES["IQLearn"].generalizes_to_unvisited is False


def test_aliases_match_canonical():
    import dataclasses

    # Every field except the name must equal the canonical record.
    def _fields_minus_name(c):
        return {k: v for k, v in dataclasses.asdict(c).items() if k != "name"}

    assert _fields_minus_name(CAPABILITIES["GLADIUS"]) == _fields_minus_name(
        CAPABILITIES["NeuralGLADIUS"]
    )
    assert _fields_minus_name(CAPABILITIES["AIRL"]) == _fields_minus_name(
        CAPABILITIES["NeuralAIRL"]
    )


def test_formspec_finite_theta_and_validation():
    lin = FormSpec(topology="tabular", reward_form="linear", num_states=5, num_actions=2)
    neu = FormSpec(topology="grid", reward_form="neural", num_states=9, num_actions=4)
    assert lin.has_finite_theta is True
    assert neu.has_finite_theta is False
    with pytest.raises(ValueError):
        FormSpec(topology="bogus", reward_form="linear", num_states=5, num_actions=2)
    with pytest.raises(ValueError):
        FormSpec(topology="tabular", reward_form="bogus", num_states=5, num_actions=2)


def test_form_wraps_env_and_delegates():
    from econirl.environments.rust_bus import RustBusEnvironment

    env = RustBusEnvironment()
    spec = FormSpec(
        topology="tabular",
        reward_form="linear",
        num_states=env.num_states,
        num_actions=env.num_actions,
        name="rust-bus",
    )
    form = Form(spec=spec, env=env)
    assert form.transition_matrices.shape[0] == env.num_actions
    assert form.parameter_names == env.parameter_names
    # problem_spec is a property that builds a fresh DDCProblem each call, so
    # compare its content, not identity.
    assert form.problem_spec.num_states == env.num_states
    assert form.problem_spec.num_actions == env.num_actions
