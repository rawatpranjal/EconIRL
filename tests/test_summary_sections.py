"""Tests for the expanded EstimationSummary sections.

Covers ``compute_fit_diagnostics`` (DATA, PRE-ESTIMATION, FIRST-STAGE
TRANSITION) on a hand-built panel with known answers, and the sectioned
``summary()`` render. These do not run an estimator, so they are fast and
deterministic.
"""

from __future__ import annotations

import numpy as np

from econirl.core.types import Panel
from econirl.inference.results import (
    EstimationSummary,
    compute_fit_diagnostics,
)


def _hand_panel() -> Panel:
    # ind0: (s,a,s') = (0,0,1),(0,0,1),(0,0,2)   ind1: (1,1,0),(2,1,0)
    states = np.array([0, 0, 0, 1, 2])
    actions = np.array([0, 0, 0, 1, 1])
    next_states = np.array([1, 1, 2, 0, 0])
    ids = np.array([0, 0, 0, 1, 1])
    return Panel.from_numpy(states, actions, next_states, ids)


def test_dataset_block_counts():
    panel = _hand_panel()
    dataset, _pre, _trans = compute_fit_diagnostics(panel, num_states=3, num_actions=2)

    assert dataset.num_states == 3
    assert dataset.num_actions == 2
    assert dataset.num_observations == 5
    assert dataset.num_individuals == 2
    assert dataset.num_periods == 3  # longest trajectory
    assert dataset.states_visited == 3
    # s0->only a0, s1->only a1, s2->only a1 : all three are single-action
    assert dataset.single_action_states == 3
    # obs per (visited) state = [3, 1, 1]
    assert dataset.obs_per_state["max"] == 3
    assert dataset.obs_per_state["min"] == 1


def test_transition_first_stage_multinomial_se():
    panel = _hand_panel()
    _dataset, _pre, trans = compute_fit_diagnostics(panel, num_states=3, num_actions=2)

    assert trans is not None
    assert trans.num_transitions == 5
    # pair (0,0) has 2-way support -> 1 free param; the two singletons add 0
    assert trans.num_free_parameters == 1
    assert trans.rows_with_support == 3  # observed (s,a) pairs
    assert trans.rows_total == 6  # 3 states x 2 actions

    # (s0,a0): p(1|.)=2/3, p(2|.)=1/3, N=3 -> se = sqrt(p(1-p)/N)
    expected = float(np.sqrt((2 / 3) * (1 / 3) / 3))
    assert abs(trans.se_quantiles["max"] - expected) < 1e-6
    # the two deterministic rows contribute se = 0
    assert trans.se_quantiles["min"] == 0.0


def test_pre_estimation_contrast_rank_bites():
    # K=2: feature 0 varies across actions (identified); feature 1 is state-only
    # (constant across actions) so it differences out -> contrast rank 1 < 2.
    phi = np.zeros((3, 2, 2), dtype=np.float64)
    phi[:, 1, 0] = 1.0  # feature 0: 0 for a=0, 1 for a=1
    for s in range(3):
        phi[s, :, 1] = s  # feature 1: state-only, same across actions

    panel = _hand_panel()
    _dataset, pre, _trans = compute_fit_diagnostics(
        panel, num_states=3, num_actions=2, feature_matrix=phi
    )

    assert pre is not None
    assert pre.kind == "feature"
    assert pre.num_features == 2
    assert pre.feature_rank == 2  # raw design is full rank
    assert pre.contrast_rank == 1  # but only one feature survives differencing
    assert "under-identified" in pre.verdict


def test_summary_renders_all_sections():
    panel = _hand_panel()
    dataset, pre, trans = compute_fit_diagnostics(
        panel, num_states=3, num_actions=2, feature_matrix=np.ones((3, 2, 2))
    )
    s = EstimationSummary(
        parameters=np.array([1.0, -2.0]),
        parameter_names=["a", "b"],
        standard_errors=np.array([0.1, 0.2]),
        method="NFXP",
        num_states=3,
        num_actions=2,
        dataset=dataset,
        pre_estimation=pre,
        transition_first_stage=trans,
    )
    out = s.summary()
    assert "[1] DATA" in out
    assert "[2] PRE-ESTIMATION CHECKS" in out
    assert "FIRST-STAGE" in out
    assert "[4] RESULTS" in out
    assert "4a. Estimation" in out


def test_summary_shows_obs_count_without_dataset_block():
    # Estimators that build EstimationSummary directly (no diagnostics block,
    # e.g. neural/IRL) must still report the observation and individual counts
    # in the header, not lose them with the [1] DATA block.
    s = EstimationSummary(
        parameters=np.array([1.0]),
        parameter_names=["a"],
        standard_errors=np.array([0.1]),
        method="Neural AIRL",
        num_observations=1234,
        num_individuals=56,
    )
    out = s.summary()
    assert "Observations:  1,234" in out
    assert "Individuals:   56" in out
    assert "[1] DATA" not in out  # no diagnostics block was attached


def test_pre_estimation_coverage_block_when_no_feature_matrix():
    panel = _hand_panel()
    _dataset, pre, _trans = compute_fit_diagnostics(panel, num_states=3, num_actions=2)

    assert pre is not None
    assert pre.kind == "coverage"
    # all 3 states visited (s0, s1, s2)
    assert pre.state_coverage == 1.0
    # observed (s,a) pairs: (0,0), (1,1), (2,1) out of 3 states x 2 actions
    assert pre.state_action_coverage == 0.5
    assert pre.demo_policy_entropy is not None
    assert pre.demo_policy_entropy >= 0.0
    # every visited state has a single action here -> zero-entropy demo policy
    assert abs(pre.demo_policy_entropy - 0.0) < 1e-9
    # two trajectories, first states 0 and 1 -> 2 distinct initial states
    assert pre.initial_states == 2
    assert pre.initial_state_entropy is not None
    assert abs(pre.initial_state_entropy - np.log(2)) < 1e-9


def test_summary_oracle_columns():
    s = EstimationSummary(
        parameters=np.array([1.0, -2.0]),
        parameter_names=["a", "b"],
        standard_errors=np.array([0.1, 0.2]),
        method="NFXP",
        true_parameters={"a": 1.0, "b": -1.5},
    )
    out = s.summary()
    assert "true" in out and "bias" in out
    # bias for b = -2.0 - (-1.5) = -0.5
    assert "-0.5000" in out
