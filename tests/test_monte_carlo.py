"""Tests for Monte Carlo parameter recovery."""

import pytest
import numpy as np

from econirl.replication.rust1987.monte_carlo import run_monte_carlo, summarize_monte_carlo


class TestMonteCarlo:
    """Tests for Monte Carlo validation."""

    def test_monte_carlo_runs(self):
        """Monte Carlo should complete without error."""
        results = run_monte_carlo(
            n_simulations=2,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller"],  # Fast estimator
            seed=42,
        )

        assert results is not None
        assert len(results) == 2  # 2 simulations

    def test_monte_carlo_recovers_params(self):
        """Monte Carlo should recover true parameters on average.

        Uses the default small environment parameters which are well-suited
        for estimation. Tests parameter recovery within reasonable bounds.
        """
        results = run_monte_carlo(
            n_simulations=3,
            n_individuals=200,
            n_periods=50,
            estimators=["Hotz-Miller"],
            seed=42,
        )

        # Filter to converged results
        converged = results[results['converged'] == True]

        # Mean estimates should be close to true values
        mean_theta_c = converged['theta_c'].mean()
        mean_RC = converged['RC'].mean()

        # True values: theta_c=0.01, RC=2.0 (new defaults)
        # Allow 50% relative error for small sample
        assert 0.005 < mean_theta_c < 0.02, f"theta_c out of range: {mean_theta_c}"
        assert 1.0 < mean_RC < 3.0, f"RC out of range: {mean_RC}"

    def test_monte_carlo_multiple_estimators(self):
        """Monte Carlo should work with multiple estimators."""
        results = run_monte_carlo(
            n_simulations=2,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller", "NPL"],
            seed=42,
        )

        # Should have results for both estimators (2 sims * 2 estimators = 4 rows)
        assert len(results) == 4
        assert set(results['estimator'].unique()) == {"Hotz-Miller", "NPL"}

    def test_monte_carlo_returns_dataframe(self):
        """Monte Carlo should return a pandas DataFrame with expected columns."""
        results = run_monte_carlo(
            n_simulations=2,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller"],
            seed=42,
        )

        # Check expected columns
        expected_cols = ['simulation', 'estimator', 'theta_c', 'RC',
                        'converged', 'true_theta_c', 'true_RC']
        for col in expected_cols:
            assert col in results.columns, f"Missing column: {col}"

    def test_monte_carlo_verbose_mode(self, capsys):
        """Monte Carlo verbose mode should print progress."""
        results = run_monte_carlo(
            n_simulations=2,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller"],
            seed=42,
            verbose=True,
        )

        # Should have output (captured by capsys in verbose mode)
        # Just check it runs without error
        assert results is not None


class TestMonteCarloSummarize:
    """Tests for Monte Carlo summary statistics."""

    def test_summarize_monte_carlo(self):
        """Summary should compute bias, RMSE, coverage."""
        results = run_monte_carlo(
            n_simulations=5,
            n_individuals=100,
            n_periods=30,
            estimators=["Hotz-Miller"],
            seed=42,
        )

        summary = summarize_monte_carlo(results)

        # Check expected columns
        expected_cols = ['estimator', 'parameter', 'true_value', 'mean_estimate',
                        'bias', 'std', 'rmse', 'n_converged']
        for col in expected_cols:
            assert col in summary.columns, f"Missing column: {col}"

        # Should have summary for both parameters
        assert len(summary) == 2  # theta_c and RC

    def test_summarize_multiple_estimators(self):
        """Summary should handle multiple estimators."""
        results = run_monte_carlo(
            n_simulations=3,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller", "NPL"],
            seed=42,
        )

        summary = summarize_monte_carlo(results)

        # Should have 2 params * 2 estimators = 4 rows
        assert len(summary) == 4
        assert set(summary['estimator'].unique()) == {"Hotz-Miller", "NPL"}


class TestMonteCarloEdgeCases:
    """Tests for edge cases."""

    def test_single_simulation(self):
        """Should work with a single simulation."""
        results = run_monte_carlo(
            n_simulations=1,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller"],
            seed=42,
        )

        assert len(results) == 1

    def test_nfxp_estimator(self):
        """Should work with NFXP estimator (slower)."""
        results = run_monte_carlo(
            n_simulations=1,
            n_individuals=30,
            n_periods=15,
            estimators=["NFXP"],
            seed=42,
        )

        assert len(results) == 1
        assert results['estimator'].iloc[0] == "NFXP"

    def test_unknown_estimator_raises(self):
        """Should raise error for unknown estimator."""
        with pytest.raises(ValueError, match="Unknown estimator"):
            run_monte_carlo(
                n_simulations=1,
                n_individuals=50,
                n_periods=20,
                estimators=["InvalidEstimator"],
                seed=42,
            )

    def test_custom_true_parameters(self):
        """Should allow custom true parameters."""
        results = run_monte_carlo(
            n_simulations=2,
            n_individuals=50,
            n_periods=20,
            true_operating_cost=0.02,
            true_replacement_cost=4.0,
            estimators=["Hotz-Miller"],
            seed=42,
        )

        assert results['true_theta_c'].iloc[0] == 0.02
        assert results['true_RC'].iloc[0] == 4.0
