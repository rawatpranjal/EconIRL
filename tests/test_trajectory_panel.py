"""Tests for TrajectoryPanel and SufficientStats.

Covers:
- from_dataframe / to_dataframe round-trip
- next_state inference (with and without explicit column)
- sufficient_stats correctness (counts, transitions, CCPs, initial dist)
- resample_individuals (count, reproducibility)
- iter_transitions (coverage, shapes)
- stacked tensor properties (all_states, all_actions, all_next_states, offsets)
- backward compatibility (Panel alias, inherited methods)
"""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from econirl.core.types import Panel, Trajectory, TrajectoryPanel
from econirl.core.sufficient_stats import SufficientStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_trajectories():
    """Two small trajectories with known data."""
    traj0 = Trajectory(
        states=jnp.array([0, 1, 2]),
        actions=jnp.array([0, 0, 1]),
        next_states=jnp.array([1, 2, 0]),
        individual_id=0,
    )
    traj1 = Trajectory(
        states=jnp.array([1, 2, 3, 4]),
        actions=jnp.array([0, 0, 0, 1]),
        next_states=jnp.array([2, 3, 4, 0]),
        individual_id=1,
    )
    return [traj0, traj1]


@pytest.fixture
def simple_panel(simple_trajectories):
    """TrajectoryPanel from simple_trajectories."""
    return TrajectoryPanel(trajectories=simple_trajectories)


@pytest.fixture
def simple_df():
    """Small DataFrame suitable for from_dataframe tests."""
    return pd.DataFrame(
        {
            "bus_id": [0, 0, 0, 1, 1, 1, 1],
            "state": [0, 1, 2, 1, 2, 3, 4],
            "action": [0, 0, 1, 0, 0, 0, 1],
            "ns": [1, 2, 0, 2, 3, 4, 0],
        }
    )


@pytest.fixture
def simple_df_no_ns():
    """DataFrame without explicit next_state column."""
    return pd.DataFrame(
        {
            "bus_id": [0, 0, 0, 1, 1, 1, 1],
            "state": [0, 1, 2, 1, 2, 3, 4],
            "action": [0, 0, 1, 0, 0, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# from_dataframe round-trip
# ---------------------------------------------------------------------------


class TestFromDataFrame:
    def test_roundtrip_with_next_state(self, simple_df):
        """from_dataframe -> to_dataframe preserves data."""
        panel = TrajectoryPanel.from_dataframe(
            simple_df, state="state", action="action", id="bus_id", next_state="ns"
        )
        out = panel.to_dataframe()
        assert set(out.columns) == {"id", "period", "state", "action", "next_state"}
        assert len(out) == len(simple_df)
        # Values should match (order may differ by id then period)
        out_sorted = out.sort_values(["id", "period"]).reset_index(drop=True)
        np.testing.assert_array_equal(out_sorted["state"].values, simple_df["state"].values)
        np.testing.assert_array_equal(out_sorted["action"].values, simple_df["action"].values)
        np.testing.assert_array_equal(out_sorted["next_state"].values, simple_df["ns"].values)

    def test_from_dataframe_with_next_state_column(self, simple_df):
        """Explicit next_state column is used correctly."""
        panel = TrajectoryPanel.from_dataframe(
            simple_df, state="state", action="action", id="bus_id", next_state="ns"
        )
        # Check individual 0
        traj0 = panel.trajectories[0]
        assert jnp.array_equal(traj0.next_states, jnp.array([1, 2, 0]))

    def test_from_dataframe_without_next_state(self, simple_df_no_ns):
        """Next states inferred from sequential rows."""
        panel = TrajectoryPanel.from_dataframe(
            simple_df_no_ns, state="state", action="action", id="bus_id"
        )
        # Individual 0: states=[0,1,2], actions=[0,0,1]
        # next_states[0] = states[1] = 1
        # next_states[1] = states[2] = 2
        # next_states[2]: action=1 -> 0
        traj0 = panel.trajectories[0]
        assert jnp.array_equal(traj0.next_states, jnp.array([1, 2, 0]))

        # Individual 1: states=[1,2,3,4], actions=[0,0,0,1]
        # next_states = [2, 3, 4, 0]  (last: action=1 -> 0)
        traj1 = panel.trajectories[1]
        assert jnp.array_equal(traj1.next_states, jnp.array([2, 3, 4, 0]))

    def test_from_dataframe_last_row_keep(self):
        """Last row with action=0: next_state = min(state+1, max_state)."""
        df = pd.DataFrame(
            {"bus_id": [0, 0], "state": [3, 4], "action": [0, 0]}
        )
        # max_state = 4, so last next_state = min(4+1, 4) = 4
        panel = TrajectoryPanel.from_dataframe(
            df, state="state", action="action", id="bus_id"
        )
        assert int(panel.trajectories[0].next_states[-1]) == 4

    def test_from_dataframe_type_error(self):
        """Non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Expected pd.DataFrame"):
            TrajectoryPanel.from_dataframe(
                {"state": [0]}, state="state", action="action", id="id"
            )

    def test_from_dataframe_num_individuals(self, simple_df):
        panel = TrajectoryPanel.from_dataframe(
            simple_df, state="state", action="action", id="bus_id", next_state="ns"
        )
        assert panel.num_individuals == 2


# ---------------------------------------------------------------------------
# from_panel
# ---------------------------------------------------------------------------


class TestFromPanel:
    def test_from_panel_wraps_trajectories(self, simple_trajectories):
        old_panel = Panel(trajectories=simple_trajectories)
        new_panel = TrajectoryPanel.from_panel(old_panel)
        assert new_panel.num_individuals == old_panel.num_individuals
        assert new_panel.num_observations == old_panel.num_observations
        # Same trajectory objects
        for orig, wrapped in zip(old_panel.trajectories, new_panel.trajectories):
            assert jnp.array_equal(orig.states, wrapped.states)


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------


class TestSufficientStats:
    def test_counts_match_manual(self, simple_panel):
        """state_action_counts match hand-computed values."""
        n_states, n_actions = 5, 2
        stats = simple_panel.sufficient_stats(n_states, n_actions)

        # Manual counts from simple_trajectories:
        # traj0: (0,0),(1,0),(2,1)
        # traj1: (1,0),(2,0),(3,0),(4,1)
        expected = jnp.zeros((n_states, n_actions))
        expected = expected.at[0, 0].set(1)   # state 0, action 0
        expected = expected.at[1, 0].set(2)   # state 1, action 0 (both trajs)
        expected = expected.at[2, 0].set(1)   # state 2, action 0 (traj1)
        expected = expected.at[2, 1].set(1)   # state 2, action 1 (traj0)
        expected = expected.at[3, 0].set(1)   # state 3, action 0 (traj1)
        expected = expected.at[4, 1].set(1)   # state 4, action 1 (traj1)

        np.testing.assert_allclose(
            np.asarray(stats.state_action_counts),
            np.asarray(expected.astype(jnp.float32)),
            atol=1e-6,
        )

    def test_transitions_valid_probabilities(self, simple_panel):
        """Transition rows sum to 1 and are non-negative."""
        stats = simple_panel.sufficient_stats(5, 2)
        assert (stats.transitions >= 0).all()
        row_sums = stats.transitions.sum(axis=2)
        np.testing.assert_allclose(
            np.asarray(row_sums),
            np.ones_like(np.asarray(row_sums)),
            atol=1e-5,
        )

    def test_empirical_ccps_match_panel(self, simple_panel):
        """Empirical CCPs match Panel.compute_choice_frequencies."""
        n_states, n_actions = 5, 2
        stats = simple_panel.sufficient_stats(n_states, n_actions)
        panel_ccps = simple_panel.compute_choice_frequencies(n_states, n_actions)

        # They should agree on states that have observations
        obs_mask = np.asarray(stats.state_action_counts.sum(axis=1)) > 0
        np.testing.assert_allclose(
            np.asarray(stats.empirical_ccps[obs_mask]),
            np.asarray(panel_ccps[obs_mask]),
            atol=1e-5,
        )

    def test_initial_distribution(self, simple_panel):
        """Initial distribution reflects starting states of trajectories."""
        stats = simple_panel.sufficient_stats(5, 2)
        # traj0 starts at 0, traj1 starts at 1 -> [0.5, 0.5, 0, 0, 0]
        expected = jnp.array([0.5, 0.5, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            np.asarray(stats.initial_distribution),
            np.asarray(expected.astype(jnp.float32)),
            atol=1e-5,
        )

    def test_n_observations_and_individuals(self, simple_panel):
        stats = simple_panel.sufficient_stats(5, 2)
        assert stats.n_observations == 7  # 3 + 4
        assert stats.n_individuals == 2

    def test_zero_observation_states_get_uniform(self):
        """States with no observations get uniform CCP and transitions."""
        traj = Trajectory(
            states=jnp.array([0, 0]),
            actions=jnp.array([0, 1]),
            next_states=jnp.array([0, 0]),
            individual_id=0,
        )
        panel = TrajectoryPanel(trajectories=[traj])
        stats = panel.sufficient_stats(n_states=3, n_actions=2)

        # State 1 and 2 have no observations -> uniform CCP = 0.5
        assert abs(float(stats.empirical_ccps[1, 0]) - 0.5) < 1e-6
        assert abs(float(stats.empirical_ccps[2, 0]) - 0.5) < 1e-6

    def test_sufficient_stats_shapes(self, simple_panel):
        stats = simple_panel.sufficient_stats(5, 2)
        assert stats.state_action_counts.shape == (5, 2)
        assert stats.transitions.shape == (2, 5, 5)
        assert stats.empirical_ccps.shape == (5, 2)
        assert stats.initial_distribution.shape == (5,)
        assert stats.n_states == 5
        assert stats.n_actions == 2


# ---------------------------------------------------------------------------
# Resample individuals
# ---------------------------------------------------------------------------


class TestResampleIndividuals:
    def test_default_n(self, simple_panel):
        """Default n resamples same number of individuals."""
        resampled = simple_panel.resample_individuals(seed=42)
        assert resampled.num_individuals == simple_panel.num_individuals

    def test_custom_n(self, simple_panel):
        """Custom n changes number of individuals."""
        resampled = simple_panel.resample_individuals(n=5, seed=42)
        assert resampled.num_individuals == 5

    def test_reproducibility(self, simple_panel):
        """Same seed produces identical resamples."""
        r1 = simple_panel.resample_individuals(seed=123)
        r2 = simple_panel.resample_individuals(seed=123)
        assert r1.num_individuals == r2.num_individuals
        for t1, t2 in zip(r1.trajectories, r2.trajectories):
            assert jnp.array_equal(t1.states, t2.states)
            assert jnp.array_equal(t1.actions, t2.actions)
            assert jnp.array_equal(t1.next_states, t2.next_states)

    def test_returns_trajectory_panel(self, simple_panel):
        """Resampled object is a TrajectoryPanel."""
        resampled = simple_panel.resample_individuals(seed=0)
        assert isinstance(resampled, TrajectoryPanel)


# ---------------------------------------------------------------------------
# iter_transitions
# ---------------------------------------------------------------------------


class TestIterTransitions:
    def test_all_observations_yielded(self, simple_panel):
        """All N observations appear across batches."""
        total = 0
        all_s, all_a, all_ns = [], [], []
        for s, a, ns in simple_panel.iter_transitions(batch_size=3):
            total += s.shape[0]
            all_s.append(s)
            all_a.append(a)
            all_ns.append(ns)
        assert total == simple_panel.num_observations

        # Check that the *set* of observations matches (order may differ)
        cat_s = jnp.sort(jnp.concatenate(all_s))
        cat_a = jnp.sort(jnp.concatenate(all_a))
        expected_s = jnp.sort(simple_panel.all_states)
        expected_a = jnp.sort(simple_panel.all_actions)
        assert jnp.array_equal(cat_s, expected_s)
        assert jnp.array_equal(cat_a, expected_a)

    def test_batch_shape(self, simple_panel):
        """Each batch has shape (B,) with B <= batch_size."""
        for s, a, ns in simple_panel.iter_transitions(batch_size=4):
            assert s.ndim == 1
            assert a.ndim == 1
            assert ns.ndim == 1
            assert s.shape[0] <= 4
            assert s.shape == a.shape == ns.shape


# ---------------------------------------------------------------------------
# Stacked tensor properties
# ---------------------------------------------------------------------------


class TestStackedTensors:
    def test_all_states_match_get_all_states(self, simple_panel):
        assert jnp.array_equal(simple_panel.all_states, simple_panel.get_all_states())

    def test_all_actions_match_get_all_actions(self, simple_panel):
        assert jnp.array_equal(simple_panel.all_actions, simple_panel.get_all_actions())

    def test_all_next_states_match_get_all_next_states(self, simple_panel):
        assert jnp.array_equal(
            simple_panel.all_next_states, simple_panel.get_all_next_states()
        )

    def test_offsets_cumulative(self, simple_panel):
        """Offsets encode cumulative trajectory lengths."""
        offsets = simple_panel.offsets
        # Two trajectories of length 3 and 4
        expected = jnp.array([0, 3, 7], dtype=jnp.int32)
        assert jnp.array_equal(offsets, expected)

    def test_offsets_last_equals_num_observations(self, simple_panel):
        assert int(simple_panel.offsets[-1]) == simple_panel.num_observations


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_panel_alias_is_trajectory_panel(self):
        """Panel is an alias for TrajectoryPanel."""
        assert Panel is TrajectoryPanel

    def test_panel_constructor_returns_trajectory_panel(self, simple_trajectories):
        """Panel(...) returns a TrajectoryPanel instance."""
        panel = Panel(trajectories=simple_trajectories)
        assert isinstance(panel, TrajectoryPanel)

    def test_inherited_methods(self, simple_panel):
        """All original Panel methods still work."""
        assert simple_panel.num_individuals == 2
        assert simple_panel.num_observations == 7
        assert simple_panel.num_periods_per_individual == [3, 4]
        assert len(simple_panel) == 2
        assert simple_panel[0].individual_id == 0

    def test_iteration(self, simple_panel):
        """Can iterate over trajectories."""
        ids = [t.individual_id for t in simple_panel]
        assert ids == [0, 1]

    def test_compute_state_frequencies(self, simple_panel):
        freqs = simple_panel.compute_state_frequencies(5)
        assert freqs.shape == (5,)
        assert abs(float(freqs.sum()) - 1.0) < 1e-6

    def test_compute_choice_frequencies(self, simple_panel):
        ccps = simple_panel.compute_choice_frequencies(5, 2)
        assert ccps.shape == (5, 2)
        # Rows with observations should sum to 1
        obs_mask = np.asarray(ccps.sum(axis=1)) > 0
        row_sums = ccps[obs_mask].sum(axis=1)
        np.testing.assert_allclose(
            np.asarray(row_sums),
            np.ones_like(np.asarray(row_sums)),
            atol=1e-6,
        )

    def test_from_numpy_returns_trajectory_panel(self):
        """Panel.from_numpy now returns a TrajectoryPanel."""
        states = np.array([0, 1, 2, 0, 1])
        actions = np.array([0, 0, 1, 0, 1])
        next_states = np.array([1, 2, 0, 1, 0])
        ids = np.array([0, 0, 0, 1, 1])
        panel = Panel.from_numpy(states, actions, next_states, ids)
        assert isinstance(panel, TrajectoryPanel)
        assert panel.num_individuals == 2

    def test_to_device_returns_trajectory_panel(self, simple_panel):
        """to() returns a TrajectoryPanel, not a plain Panel."""
        moved = simple_panel.to("cpu")
        assert isinstance(moved, TrajectoryPanel)


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


class TestToDataFrame:
    def test_columns(self, simple_panel):
        df = simple_panel.to_dataframe()
        assert list(df.columns) == ["id", "period", "state", "action", "next_state"]

    def test_length(self, simple_panel):
        df = simple_panel.to_dataframe()
        assert len(df) == simple_panel.num_observations

    def test_period_resets_per_individual(self, simple_panel):
        df = simple_panel.to_dataframe()
        for _, group in df.groupby("id"):
            assert list(group["period"]) == list(range(len(group)))


# ---------------------------------------------------------------------------
# SufficientStats dataclass
# ---------------------------------------------------------------------------


class TestSufficientStatsDataclass:
    def test_validation_inconsistent_states(self):
        """Mismatched state dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Inconsistent state dimensions"):
            SufficientStats(
                state_action_counts=jnp.zeros((5, 2)),
                transitions=jnp.zeros((2, 4, 4)),  # 4 != 5
                empirical_ccps=jnp.zeros((5, 2)),
                initial_distribution=jnp.zeros(5),
                n_observations=10,
                n_individuals=2,
            )

    def test_validation_inconsistent_actions(self):
        """Mismatched action dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Inconsistent action dimensions"):
            SufficientStats(
                state_action_counts=jnp.zeros((5, 2)),
                transitions=jnp.zeros((3, 5, 5)),  # 3 != 2
                empirical_ccps=jnp.zeros((5, 2)),
                initial_distribution=jnp.zeros(5),
                n_observations=10,
                n_individuals=2,
            )

    def test_valid_construction(self):
        stats = SufficientStats(
            state_action_counts=jnp.ones((3, 2)),
            transitions=jnp.ones((2, 3, 3)) / 3,
            empirical_ccps=jnp.ones((3, 2)) / 2,
            initial_distribution=jnp.ones(3) / 3,
            n_observations=6,
            n_individuals=2,
        )
        assert stats.n_states == 3
        assert stats.n_actions == 2
