"""Tests for equipment replacement variants dataset."""

import pytest
import pandas as pd

from econirl.datasets import load_equipment_replacement
from econirl.datasets.equipment_replacement import get_equipment_replacement_info
from econirl.core.types import Panel


class TestLoadEquipmentReplacement:
    """Tests for load_equipment_replacement function."""

    def test_basic_loading_returns_dataframe(self) -> None:
        """Basic loading returns a DataFrame."""
        df = load_equipment_replacement()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_correct_columns(self) -> None:
        """DataFrame has expected columns."""
        df = load_equipment_replacement(n_machines=10, n_periods=5)
        expected_columns = ["id", "period", "state", "action", "wear_level", "variant"]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_as_panel_returns_panel(self) -> None:
        """as_panel=True returns Panel object."""
        panel = load_equipment_replacement(n_machines=10, n_periods=5, as_panel=True)
        assert isinstance(panel, Panel)
        assert panel.num_individuals == 10

    def test_n_machines_parameter(self) -> None:
        """Different n_machines values work correctly."""
        for n in [10, 50, 100]:
            df = load_equipment_replacement(n_machines=n, n_periods=5)
            assert df["id"].nunique() == n

    def test_n_periods_parameter(self) -> None:
        """Different n_periods values work correctly."""
        for periods in [10, 20, 50]:
            df = load_equipment_replacement(n_machines=5, n_periods=periods)
            periods_per_machine = df.groupby("id")["period"].count()
            assert (periods_per_machine == periods).all()

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical data."""
        df1 = load_equipment_replacement(n_machines=20, n_periods=10, seed=42)
        df2 = load_equipment_replacement(n_machines=20, n_periods=10, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds produce different data."""
        df1 = load_equipment_replacement(n_machines=20, n_periods=10, seed=42)
        df2 = load_equipment_replacement(n_machines=20, n_periods=10, seed=123)
        # Actions or states should differ (with very high probability)
        assert not df1["action"].equals(df2["action"]) or not df1["state"].equals(df2["state"])

    def test_invalid_variant_raises_error(self) -> None:
        """Invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="variant must be one of"):
            load_equipment_replacement(variant="invalid")


class TestBinaryVariant:
    """Tests for binary variant (keep/replace)."""

    def test_binary_has_two_actions(self) -> None:
        """Binary variant has exactly 2 possible actions."""
        df = load_equipment_replacement(
            variant="binary", n_machines=100, n_periods=100
        )
        unique_actions = df["action"].unique()
        # All actions should be in [0, 1]
        assert set(unique_actions).issubset({0, 1})
        # With enough data, both actions should appear
        assert len(unique_actions) == 2, f"Only {len(unique_actions)} actions observed"

    def test_binary_has_90_states_max(self) -> None:
        """Binary variant has at most 90 states."""
        df = load_equipment_replacement(
            variant="binary", n_machines=100, n_periods=100
        )
        assert df["state"].min() >= 0
        assert df["state"].max() < 90

    def test_binary_variant_label(self) -> None:
        """Binary variant label is stored correctly."""
        df = load_equipment_replacement(variant="binary", n_machines=5, n_periods=5)
        assert (df["variant"] == "binary").all()


class TestTernaryVariant:
    """Tests for ternary variant (keep/minor_repair/major_repair)."""

    def test_ternary_has_three_actions(self) -> None:
        """Ternary variant has exactly 3 possible actions."""
        df = load_equipment_replacement(
            variant="ternary", n_machines=200, n_periods=100
        )
        unique_actions = df["action"].unique()
        # All actions should be in [0, 1, 2]
        assert set(unique_actions).issubset({0, 1, 2})
        # With enough data, all 3 actions should appear
        assert len(unique_actions) == 3, f"Only {len(unique_actions)} actions observed"

    def test_ternary_has_90_states_max(self) -> None:
        """Ternary variant has at most 90 states."""
        df = load_equipment_replacement(
            variant="ternary", n_machines=100, n_periods=100
        )
        assert df["state"].min() >= 0
        assert df["state"].max() < 90

    def test_ternary_variant_label(self) -> None:
        """Ternary variant label is stored correctly."""
        df = load_equipment_replacement(variant="ternary", n_machines=5, n_periods=5)
        assert (df["variant"] == "ternary").all()


class TestContinuousStateVariant:
    """Tests for continuous_state variant (finer state granularity)."""

    def test_continuous_state_has_two_actions(self) -> None:
        """Continuous state variant has exactly 2 possible actions."""
        df = load_equipment_replacement(
            variant="continuous_state", n_machines=100, n_periods=100
        )
        unique_actions = df["action"].unique()
        # All actions should be in [0, 1]
        assert set(unique_actions).issubset({0, 1})
        # With enough data, both actions should appear
        assert len(unique_actions) == 2, f"Only {len(unique_actions)} actions observed"

    def test_continuous_state_has_200_states_max(self) -> None:
        """Continuous state variant has at most 200 states."""
        df = load_equipment_replacement(
            variant="continuous_state", n_machines=100, n_periods=100
        )
        assert df["state"].min() >= 0
        assert df["state"].max() < 200

    def test_continuous_state_more_states_than_binary(self) -> None:
        """Continuous state variant explores more states than binary."""
        df_binary = load_equipment_replacement(
            variant="binary", n_machines=100, n_periods=100, seed=42
        )
        df_continuous = load_equipment_replacement(
            variant="continuous_state", n_machines=100, n_periods=100, seed=42
        )
        # Continuous state should have higher max state values
        # (due to finer granularity and more states)
        max_binary = df_binary["state"].max()
        max_continuous = df_continuous["state"].max()
        # The continuous variant should reach higher state indices
        assert max_continuous > max_binary

    def test_continuous_state_variant_label(self) -> None:
        """Continuous state variant label is stored correctly."""
        df = load_equipment_replacement(
            variant="continuous_state", n_machines=5, n_periods=5
        )
        assert (df["variant"] == "continuous_state").all()


class TestPanelConversion:
    """Tests for Panel conversion functionality."""

    def test_panel_num_observations(self) -> None:
        """Panel has correct total observations."""
        n_machines = 20
        n_periods = 15
        panel = load_equipment_replacement(
            n_machines=n_machines, n_periods=n_periods, as_panel=True
        )
        assert panel.num_observations == n_machines * n_periods

    def test_panel_trajectory_lengths(self) -> None:
        """Each trajectory has correct length."""
        n_machines = 10
        n_periods = 25
        panel = load_equipment_replacement(
            n_machines=n_machines, n_periods=n_periods, as_panel=True
        )
        for traj in panel.trajectories:
            assert len(traj) == n_periods

    def test_panel_states_actions_match_dataframe(self) -> None:
        """Panel states and actions match DataFrame values."""
        n_machines = 5
        n_periods = 10
        seed = 999

        df = load_equipment_replacement(
            n_machines=n_machines, n_periods=n_periods, seed=seed
        )
        panel = load_equipment_replacement(
            n_machines=n_machines, n_periods=n_periods, seed=seed, as_panel=True
        )

        # Check that states and actions match
        for traj in panel.trajectories:
            machine_df = df[df["id"] == traj.individual_id].sort_values("period")
            assert list(traj.states.numpy()) == list(machine_df["state"].values)
            assert list(traj.actions.numpy()) == list(machine_df["action"].values)

    def test_panel_for_each_variant(self) -> None:
        """Panel conversion works for all variants."""
        for variant in ["binary", "ternary", "continuous_state"]:
            panel = load_equipment_replacement(
                variant=variant, n_machines=5, n_periods=10, as_panel=True
            )
            assert isinstance(panel, Panel)
            assert panel.num_individuals == 5


class TestDataQuality:
    """Tests for data quality and realistic patterns."""

    def test_wear_level_non_negative(self) -> None:
        """Wear level is always non-negative."""
        for variant in ["binary", "ternary", "continuous_state"]:
            df = load_equipment_replacement(
                variant=variant, n_machines=50, n_periods=50
            )
            assert df["wear_level"].min() >= 0

    def test_state_non_negative(self) -> None:
        """State values are always non-negative."""
        for variant in ["binary", "ternary", "continuous_state"]:
            df = load_equipment_replacement(
                variant=variant, n_machines=50, n_periods=50
            )
            assert df["state"].min() >= 0

    def test_period_zero_indexed(self) -> None:
        """Periods are 0-indexed and sequential."""
        df = load_equipment_replacement(n_machines=10, n_periods=20)
        for machine_id in df["id"].unique():
            periods = df[df["id"] == machine_id]["period"].values
            assert list(periods) == list(range(20))

    def test_replacement_resets_state(self) -> None:
        """Replacement action (action=1 in binary) resets state to 0."""
        # Generate data and check transitions after replacement
        df = load_equipment_replacement(
            variant="binary", n_machines=100, n_periods=100, seed=42
        )

        # For each machine, check that after action=1, next period starts at state 0
        for machine_id in df["id"].unique():
            machine_df = df[df["id"] == machine_id].sort_values("period")
            replacements = machine_df[machine_df["action"] == 1]["period"].values

            for replace_period in replacements:
                if replace_period < 99:  # Not the last period
                    next_period_data = machine_df[machine_df["period"] == replace_period + 1]
                    if len(next_period_data) > 0:
                        assert next_period_data["state"].values[0] == 0


class TestInfoFunction:
    """Tests for get_equipment_replacement_info function."""

    def test_info_returns_dict(self) -> None:
        """Info function returns dictionary with expected keys."""
        info = get_equipment_replacement_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "num_states" in info
        assert "num_actions" in info
        assert "variant" in info

    def test_info_for_binary(self) -> None:
        """Info for binary variant is correct."""
        info = get_equipment_replacement_info(variant="binary")
        assert info["num_states"] == 90
        assert info["num_actions"] == 2
        assert info["variant"] == "binary"

    def test_info_for_ternary(self) -> None:
        """Info for ternary variant is correct."""
        info = get_equipment_replacement_info(variant="ternary")
        assert info["num_states"] == 90
        assert info["num_actions"] == 3
        assert info["variant"] == "ternary"

    def test_info_for_continuous_state(self) -> None:
        """Info for continuous_state variant is correct."""
        info = get_equipment_replacement_info(variant="continuous_state")
        assert info["num_states"] == 200
        assert info["num_actions"] == 2
        assert info["variant"] == "continuous_state"

    def test_info_contains_action_names(self) -> None:
        """Info contains action names for each variant."""
        for variant in ["binary", "ternary", "continuous_state"]:
            info = get_equipment_replacement_info(variant=variant)
            assert "action_names" in info
            assert len(info["action_names"]) == info["num_actions"]
