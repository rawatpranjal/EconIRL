"""Drift guard + sanity for the forms capability registry (chunk F0)."""

from __future__ import annotations

import pytest

import econirl
from econirl.forms import CAPABILITIES, EstimatorCapability, Form, FormSpec

# The one public estimator that does not live under econirl.estimators.* (legacy
# module). Every other public estimator class is exported from econirl.estimators.
_NON_ESTIMATORS_MODULE_PUBLIC = {"IQLearn"}


def _estimators_module_classes() -> set[str]:
    """Public names whose class is defined under econirl.estimators.*"""
    out = set()
    for name in econirl.__all__:
        obj = getattr(econirl, name)
        if isinstance(obj, type) and obj.__module__.startswith("econirl.estimators"):
            out.add(name)
    return out


def test_capabilities_cover_exactly_public_estimators():
    """Drift guard: add/remove a public estimator and this fails until the
    registry is updated. Catches both missing entries and orphans."""
    expected = _estimators_module_classes() | _NON_ESTIMATORS_MODULE_PUBLIC
    assert set(CAPABILITIES) == expected


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
    for n in ("NeuralAIRL", "NeuralGLADIUS", "AIRL", "GLADIUS"):
        c = CAPABILITIES[n]
        assert c.path == "fit_features"
        assert c.model_free
        assert not c.needs_transitions
    assert CAPABILITIES["IQLearn"].generalizes_to_unvisited is False


def test_aliases_match_canonical():
    assert CAPABILITIES["GLADIUS"].path == CAPABILITIES["NeuralGLADIUS"].path
    assert CAPABILITIES["GLADIUS"].model_free == CAPABILITIES["NeuralGLADIUS"].model_free
    assert CAPABILITIES["AIRL"].model_free == CAPABILITIES["NeuralAIRL"].model_free


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
        topology="tabular", reward_form="linear",
        num_states=env.num_states, num_actions=env.num_actions, name="rust-bus",
    )
    form = Form(spec=spec, env=env)
    assert form.transition_matrices.shape[0] == env.num_actions
    assert form.parameter_names == env.parameter_names
    # problem_spec is a property that builds a fresh DDCProblem each call, so
    # compare its content, not identity.
    assert form.problem_spec.num_states == env.num_states
    assert form.problem_spec.num_actions == env.num_actions
