"""Unified AIRL public API tests.

The public AIRL facade should expose only paper-identified variants:
Fu-Luo-Levine state-only AIRL and Lee-Sudhir-Wang anchored modes.
The unanchored state-action AIRL diagnostic remains available only through
the legacy concrete class, not through the unified ``AIRL`` entry point.
"""

import pytest

from econirl.estimation.adversarial import (
    AIRL,
    AIRLConfig,
    AIRLEstimator,
    AIRLHetEstimator,
)


def test_default_airl_routes_to_fu_state_only_estimator():
    estimator = AIRL(AIRLConfig(max_rounds=3, compute_se=False))

    assert estimator.version == "state_only"
    assert isinstance(estimator.delegate, AIRLEstimator)
    assert estimator.delegate.config.reward_arg == "state"
    assert estimator.name == "AIRL (Fu et al. 2018, state-only)"


def test_unified_airl_rejects_unanchored_state_action_reward():
    with pytest.raises(ValueError, match="not identified"):
        AIRL(AIRLConfig(reward_arg="state_action", compute_se=False))


def test_anchored_airl_routes_to_single_segment_airl_het():
    estimator = AIRL(
        AIRLConfig(
            version="anchored",
            exit_action=1,
            absorbing_state=2,
            max_rounds=3,
            compute_se=False,
        )
    )

    assert estimator.version == "anchored"
    assert isinstance(estimator.delegate, AIRLHetEstimator)
    assert estimator.delegate.config.num_segments == 1
    assert estimator.delegate.config.exit_action == 1
    assert estimator.delegate.config.absorbing_state == 2


def test_anchored_airl_requires_both_identification_anchors():
    with pytest.raises(ValueError, match="exit_action"):
        AIRL(AIRLConfig(version="anchored", absorbing_state=2))
    with pytest.raises(ValueError, match="absorbing_state"):
        AIRL(AIRLConfig(version="anchored", exit_action=1))


def test_heterogeneous_airl_routes_to_airl_het():
    estimator = AIRL(
        AIRLConfig(
            version="heterogeneous",
            num_segments=3,
            exit_action=1,
            absorbing_state=2,
            max_rounds=3,
            compute_se=False,
        )
    )

    assert estimator.version == "heterogeneous"
    assert isinstance(estimator.delegate, AIRLHetEstimator)
    assert estimator.delegate.config.num_segments == 3


def test_public_airl_aliases_point_to_unified_facade():
    from econirl import AIRL as top_level_airl
    from econirl.estimation import AIRL as estimation_airl
    from econirl.estimators import AIRL as estimators_airl
    from econirl.estimators import NeuralAIRL

    assert top_level_airl is AIRL
    assert estimation_airl is AIRL
    assert estimators_airl is AIRL
    assert estimators_airl is not NeuralAIRL
