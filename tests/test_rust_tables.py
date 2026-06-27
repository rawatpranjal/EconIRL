"""Tests for Rust (1987) table replication."""

from pathlib import Path

import numpy as np
import pytest

from econirl.replication.rust1987.table_ix import (
    load_stordat_group4_panel,
    table_ix_group4,
)
from econirl.replication.rust1987.tables import table_ii_descriptives, table_iv_transitions

OFFICIAL_GROUP4_RAW = Path("downloads/nfxp_unzip/nfxp/dat/a530875.asc")


class TestTableII:
    """Tests for Table II replication."""

    def test_table_ii_structure(self):
        """Table II should have correct structure."""
        table = table_ii_descriptives()

        # Should have rows for each group
        assert len(table) >= 4

        # Should have required columns
        required_cols = ['n_buses', 'n_replacements', 'mean_mileage', 'std_mileage']
        for col in required_cols:
            assert col in table.columns

    def test_table_ii_values_reasonable(self):
        """Table II values should be in reasonable ranges."""
        table = table_ii_descriptives()

        # Mean mileage at replacement should be positive (where defined)
        # Some groups may have no replacements, resulting in NaN
        valid_mileage = table['mean_mileage'].dropna()
        if len(valid_mileage) > 0:
            assert (valid_mileage > 0).all()

        # Should have some replacements across all groups
        assert table['n_replacements'].sum() > 0

    def test_table_ii_with_synthetic_data(self):
        """Table II should work with synthetic data."""
        table = table_ii_descriptives(original=False)

        # Synthetic data should have all 8 groups
        assert len(table) == 8

        # All groups should have positive mileage values
        assert (table['mean_mileage'] > 0).all()

        # Number of buses should be positive
        assert (table['n_buses'] > 0).all()


class TestTableIV:
    """Tests for Table IV replication."""

    def test_table_iv_structure(self):
        """Table IV should have correct structure."""
        table = table_iv_transitions(original=False)

        # Should have required columns
        required_cols = ['theta_0', 'theta_1', 'theta_2', 'n_transitions']
        for col in required_cols:
            assert col in table.columns

    def test_table_iv_probabilities_valid(self):
        """Transition probabilities should be valid."""
        table = table_iv_transitions(original=False)

        # Probabilities should be non-negative
        for col in ['theta_0', 'theta_1', 'theta_2']:
            assert (table[col] >= 0).all()

        # Probabilities should sum to 1 for each group
        prob_sums = table['theta_0'] + table['theta_1'] + table['theta_2']
        assert np.allclose(prob_sums, 1.0, atol=1e-6)

    def test_table_iv_n_transitions_positive(self):
        """Number of transitions should be positive."""
        table = table_iv_transitions(original=False)

        assert (table['n_transitions'] > 0).all()


class TestTableV:
    """Tests for Table V replication (structural estimates)."""

    def test_table_v_runs(self):
        """Table V estimation should complete without error."""
        from econirl.replication.rust1987.tables import table_v_structural

        # Use synthetic data and single group for speed
        table = table_v_structural(groups=[1], estimators=["NFXP"], original=False)

        assert table is not None
        assert 'theta_c' in table.columns

    def test_table_v_has_standard_errors(self):
        """Table V should include standard errors."""
        from econirl.replication.rust1987.tables import table_v_structural

        table = table_v_structural(groups=[1], estimators=["NFXP"], original=False)

        # Should have SE columns
        se_cols = [c for c in table.columns if 'se' in c.lower()]
        assert len(se_cols) > 0

    def test_table_v_converges(self):
        """Estimation should converge."""
        from econirl.replication.rust1987.tables import table_v_structural

        table = table_v_structural(groups=[1], estimators=["Hotz-Miller"], original=False)
        assert table['converged'].all()


class TestTableIXProfile:
    """Tests for the exact Rust Table IX replication profile."""

    def test_stordat_group4_panel_matches_official_counts(self):
        """STORDAT mirror should reproduce the official group-4 sample."""
        if not OFFICIAL_GROUP4_RAW.exists():
            pytest.skip("official NFXP archive not downloaded")

        df, metadata = load_stordat_group4_panel(OFFICIAL_GROUP4_RAW)

        assert len(df) == 4292
        assert metadata["sample_observations"] == 4292
        assert metadata["replacement_count"] == 33
        assert metadata["transition_counts"] == {0: 1682, 1: 2555, 2: 55}
        assert metadata["transition_probabilities"][0] == pytest.approx(0.3918918919)
        assert metadata["transition_probabilities"][1] == pytest.approx(0.5952935694)
        assert metadata["transition_probabilities"][2] == pytest.approx(0.0128145387)
        assert df["mileage_bin"].between(0, 89).all()
        assert set(df["replaced"].unique()) <= {0, 1}

    @pytest.mark.slow
    def test_table_ix_profile_matches_paper_values(self, tmp_path):
        """Strict BHHH profile should match Rust's reported group-4 estimates."""
        if not OFFICIAL_GROUP4_RAW.exists():
            pytest.skip("official NFXP archive not downloaded")

        note_path = tmp_path / "nfxp_rust1987_table_ix.md"
        table = table_ix_group4(
            OFFICIAL_GROUP4_RAW,
            replication_note_path=note_path,
            command="make rust-table-ix",
        )

        beta_9999 = table[np.isclose(table["beta"], 0.9999)].iloc[0]
        beta_0 = table[np.isclose(table["beta"], 0.0)].iloc[0]

        assert bool(beta_9999["converged"])
        assert beta_9999["theta_1_paper_units"] == pytest.approx(2.2930, abs=5e-4)
        assert beta_9999["theta_1_se_paper_units"] == pytest.approx(0.639, abs=5e-4)
        assert beta_9999["RC"] == pytest.approx(10.0750, abs=5e-4)
        assert beta_9999["RC_se"] == pytest.approx(1.582, abs=5e-4)
        assert beta_9999["full_log_likelihood"] == pytest.approx(-3304.155, abs=5e-3)

        assert bool(beta_0["converged"])
        assert beta_0["theta_1_paper_units"] == pytest.approx(71.5133, abs=5e-4)
        assert beta_0["theta_1_se_paper_units"] == pytest.approx(13.778, abs=1e-3)
        assert beta_0["RC"] == pytest.approx(7.6358, abs=5e-4)
        assert beta_0["RC_se"] == pytest.approx(0.7197, abs=5e-4)
        assert beta_0["choice_log_likelihood"] == pytest.approx(-165.458, abs=5e-3)
        receipt = note_path.read_text(encoding="utf-8")
        assert "Verdict: pass" in receipt
        assert "Command: `make rust-table-ix`" in receipt
        assert "Rust (1987), Table IX" in receipt


class TestExport:
    """Tests for LaTeX export."""

    def test_table_ii_latex(self):
        """Table II should export to LaTeX."""
        from econirl.replication.rust1987 import table_ii_descriptives
        from econirl.replication.rust1987.export import table_to_latex

        table = table_ii_descriptives(original=False)
        latex = table_to_latex(table, caption="Table II: Descriptive Statistics")

        assert "\\begin{table}" in latex
        assert "Descriptive Statistics" in latex

    def test_save_all_tables(self):
        """save_all_tables should create output files."""
        import os
        import tempfile

        from econirl.replication.rust1987.export import save_all_tables

        with tempfile.TemporaryDirectory() as tmpdir:
            save_all_tables(output_dir=tmpdir, original=False, groups=[1])

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "table_ii.csv"))
            assert os.path.exists(os.path.join(tmpdir, "table_ii.tex"))
            assert os.path.exists(os.path.join(tmpdir, "table_iv.csv"))
            assert os.path.exists(os.path.join(tmpdir, "table_v.csv"))
