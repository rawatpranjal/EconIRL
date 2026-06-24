"""Fast self-tests for the content-consumption environment and mixture panel.

These verify the construction contract (shapes, valid transitions, the leave /
absorbing anchors), that distinct reward thetas induce distinct optimal policies,
and that the public ``simulate_mixture_panel`` carries per-agent segment labels.
They are cheap and run no estimator fits (the AIRL-Het smoke is run separately).
"""

from __future__ import annotations

import numpy as np
import pytest

from econirl.environments import content_consumption
from econirl.preprocessing.diagnostics import feature_diagnostics
from econirl.simulation.synthetic import _compute_optimal_policy, simulate_mixture_panel


# ---------------------------------------------------------------------------
# (a) Construction and validation
# ---------------------------------------------------------------------------


def test_shapes_and_anchor_indices() -> None:
    env = content_consumption()
    # Defaults: 3 categories x 4 satiation levels -> 64 regular + 1 absorbing.
    assert env.num_states == 65
    assert env.num_actions == 4
    assert env.num_features == 4
    assert env.transition_matrices.shape == (4, 65, 65)
    assert env.feature_matrix.shape == (65, 4, 4)
    # Anchor indices exposed for the study + AIRL-Het.
    assert env.leave_action == 3
    assert env.session_ended_state == 64


def test_transition_rows_sum_to_one() -> None:
    env = content_consumption()
    T = np.asarray(env.transition_matrices)
    row_sums = T.sum(axis=2)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)
    assert (T >= -1e-12).all()


def test_leave_action_routes_to_absorbing_state() -> None:
    env = content_consumption()
    T = np.asarray(env.transition_matrices)
    leave = env.leave_action
    absorbing = env.session_ended_state
    n_regular = absorbing  # absorbing is the top index
    # From every regular state, leaving sends all mass to the absorbing state.
    for s in range(n_regular):
        assert T[leave, s, absorbing] == pytest.approx(1.0)


def test_absorbing_state_is_self_absorbing() -> None:
    env = content_consumption()
    T = np.asarray(env.transition_matrices)
    absorbing = env.session_ended_state
    for a in range(env.num_actions):
        assert T[a, absorbing, absorbing] == pytest.approx(1.0)


def test_leave_action_and_absorbing_state_have_zero_features() -> None:
    env = content_consumption()
    phi = np.asarray(env.feature_matrix)
    leave = env.leave_action
    absorbing = env.session_ended_state
    # Exit action: zero-reward anchor for every state.
    np.testing.assert_allclose(phi[:, leave, :], 0.0)
    # Absorbing state: zero-value anchor for every action.
    np.testing.assert_allclose(phi[absorbing, :, :], 0.0)


def test_features_are_identified() -> None:
    # Action-contrast rank must equal K, else some reward weight is unrecoverable.
    env = content_consumption()
    diag = feature_diagnostics(np.asarray(env.feature_matrix))
    assert diag["contrast_rank"] == diag["num_features"] == 4
    assert diag["contrast_condition_number"] < 100.0


# ---------------------------------------------------------------------------
# (b) Different theta -> different optimal policy
# ---------------------------------------------------------------------------


def test_two_thetas_give_different_policies() -> None:
    # Type 1 binge-watcher: high enjoyment weight, low satiation cost.
    binge = content_consumption(theta=np.array([2.0, 0.2, 0.5, 0.3]))
    # Type 2 variety-seeker: high satiation cost and variety bonus.
    sampler = content_consumption(theta=np.array([1.0, 1.5, 0.5, 2.0]))

    pol_binge = np.asarray(_compute_optimal_policy(binge))
    pol_sampler = np.asarray(_compute_optimal_policy(sampler))

    assert pol_binge.shape == pol_sampler.shape == (65, 4)
    # The two policies must differ materially, not by float noise.
    max_tv = float(np.abs(pol_binge - pol_sampler).sum(axis=1).max())
    assert max_tv > 0.1
    # And differ in the greedy action of many regular states.
    n_regular = binge.session_ended_state
    greedy_binge = pol_binge[:n_regular].argmax(axis=1)
    greedy_sampler = pol_sampler[:n_regular].argmax(axis=1)
    assert (greedy_binge != greedy_sampler).sum() > 0


# ---------------------------------------------------------------------------
# (c) Mixture panel carries per-agent segment labels
# ---------------------------------------------------------------------------


def test_mixture_panel_segment_labels() -> None:
    binge = content_consumption(theta=np.array([2.0, 0.2, 0.5, 0.3]))
    sampler = content_consumption(theta=np.array([1.0, 1.5, 0.5, 2.0]))

    n_ind = 120
    panel = simulate_mixture_panel(
        segment_envs=[binge, sampler],
        segment_probs=[0.5, 0.5],
        n_individuals=n_ind,
        n_periods=30,
        seed=7,
    )

    assert panel.num_individuals == n_ind
    labels = panel.metadata["segment_labels"]
    assert len(labels) == n_ind
    # Both K=2 segments must appear (with this seed and 50/50 split).
    assert set(labels) == {0, 1}
    # Per-trajectory labels agree with the panel-level list.
    for i, traj in enumerate(panel.trajectories):
        assert traj.metadata["segment"] == labels[i]
    assert panel.metadata["num_segments"] == 2


def test_mixture_panel_rejects_mismatched_probs() -> None:
    binge = content_consumption(theta=np.array([2.0, 0.2, 0.5, 0.3]))
    sampler = content_consumption(theta=np.array([1.0, 1.5, 0.5, 2.0]))
    with pytest.raises(ValueError):
        simulate_mixture_panel([binge, sampler], [1.0], n_individuals=5, n_periods=5)
