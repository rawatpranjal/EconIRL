"""Guard tests for the ``rust_bus_reward_spec`` migration.

Confirms the generic RewardSpec built for the Rust bus problem has the
expected shape, names, and feature values, that each structural wrapper
accepts it, and that the removed ``"linear_cost"`` string preset now raises
a clear ValueError instead of silently building bus features.
"""

from __future__ import annotations

import numpy as np
import pytest

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel
from econirl.datasets import rust_bus_reward_spec
from econirl.estimators import CCP, NFXP, NNES, SEES, TDCCP, UFXP


def test_rust_bus_reward_spec_shape_and_names():
    spec = rust_bus_reward_spec(90)

    assert isinstance(spec, RewardSpec)
    assert spec.feature_matrix.shape == (90, 2, 2)
    assert spec.parameter_names == ["operating_cost", "replacement_cost"]


def test_rust_bus_reward_spec_feature_values():
    spec = rust_bus_reward_spec(90)
    features = np.asarray(spec.feature_matrix)

    assert np.array_equal(features[:, 0, 0], -np.arange(90))
    assert np.array_equal(features[:, 1, 1], -np.ones(90))

    # Everything else is zero: keep's replacement-cost feature and
    # replace's operating-cost feature.
    assert np.array_equal(features[:, 0, 1], np.zeros(90))
    assert np.array_equal(features[:, 1, 0], np.zeros(90))


@pytest.mark.parametrize("estimator_cls", [NFXP, CCP, NNES, SEES, TDCCP, UFXP])
def test_estimator_constructs_with_rust_bus_reward_spec(estimator_cls):
    utility = rust_bus_reward_spec(90)
    model = estimator_cls(n_states=90, utility=utility)

    assert model.utility is utility


def _tiny_panel() -> Panel:
    states = np.array([0, 1, 2, 0, 1])
    actions = np.array([0, 1, 0, 1, 0])
    next_states = np.array([1, 2, 0, 1, 2])
    ids = np.array([0, 0, 0, 1, 1])
    return Panel.from_numpy(states, actions, next_states, ids)


@pytest.mark.parametrize("estimator_cls", [NFXP, CCP, NNES, SEES, TDCCP, UFXP])
def test_linear_cost_string_preset_removed(estimator_cls):
    panel = _tiny_panel()
    model = estimator_cls(n_states=3, n_actions=2, utility="linear_cost")

    with pytest.raises(ValueError, match="RewardSpec"):
        model.fit(panel)
