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


def _group4_inputs(beta=0.9999):
    """Build (panel, utility, problem, transitions) for Rust Group-4 at a discount beta."""
    from econirl.datasets import load_rust_bus
    from econirl.estimation.transitions import estimate_transition_probs_by_group
    from econirl.environments.rust_bus import RustBusEnvironment
    from econirl.preferences.linear import LinearUtility
    from econirl.replication.rust1987.tables import _df_to_panel

    df = load_rust_bus(original=True)
    probs = estimate_transition_probs_by_group(df)
    panel = _df_to_panel(df[df["group"] == 4])
    env = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        mileage_transition_probs=tuple(probs[4]),
        discount_factor=beta,
    )
    return panel, LinearUtility.from_environment(env), env.problem_spec, env.transition_matrices


def _fit_group4(estimator):
    """Fit one estimator on Rust Group-4 data (beta=0.9999) and return (theta_c, RC, loglik)."""
    panel, utility, problem, transitions = _group4_inputs()
    res = estimator.estimate(panel, utility, problem, transitions)
    return res.parameters[0].item(), res.parameters[1].item(), float(res.log_likelihood)


@pytest.mark.slow
class TestNPLConvergenceAM2002:
    """Replicate Aguirregabiria & Mira (2002) on the Rust (1987) bus data.

    AM2002 Section 5.2: the Hotz-Miller 1-stage estimator is poor, the gains from
    extra policy iterations come fast, and NPL run to its fixed point attains the
    MLE (Lemma 2 / footnote 15: "NFXP and NPL ML estimates equal to the twelfth
    digit"). On Group 4 (identical data and transitions for every estimator), NPL
    run to convergence reaches the NFXP MLE. At beta=0.9999 the choice likelihood
    is very flat in the replacement cost, so NPL and NFXP both sit within ~5e-4 of
    the exact argmax and agree to the 3rd-4th figure (NPL RC 10.1432, NFXP 10.1423;
    NPL theta_c 0.0022640, NFXP 0.0022636); at beta<=0.99 NPL matches the exact MLE
    to five figures. The earlier 4th-figure gap was a premature NPL stop: a loose
    parameter tolerance broke the policy-iteration loop before the CCP fixed point.
    """

    def test_hotz_miller_one_stage_is_poorer_than_nfxp(self):
        from econirl.estimation.ccp import CCPEstimator
        from econirl.estimation.nfxp import NFXPEstimator

        _, _, ll_nfxp = _fit_group4(NFXPEstimator(verbose=False, outer_max_iter=200))
        _, _, ll_hm = _fit_group4(CCPEstimator(num_policy_iterations=1, verbose=False))
        # AM2002: the 1-PI (Hotz-Miller) estimator performs poorly.
        assert ll_hm < ll_nfxp - 1.0

    def test_npl_run_to_convergence_attains_the_mle(self):
        from econirl.estimation.ccp import CCPEstimator
        from econirl.estimation.nfxp import NFXPEstimator

        tc_n, rc_n, ll_n = _fit_group4(NFXPEstimator(verbose=False, outer_max_iter=300))
        tc_p, rc_p, ll_p = _fit_group4(CCPEstimator(num_policy_iterations=-1, verbose=False))
        # NPL run to its fixed point attains the NFXP MLE (AM2002 Lemma 2 /
        # footnote 15). The likelihood is flat in RC at beta=0.9999, so the two
        # agree to the 3rd-4th figure rather than the 12th the paper reports.
        assert tc_p == pytest.approx(tc_n, abs=1e-5)
        assert rc_p == pytest.approx(rc_n, abs=0.01)
        assert ll_p == pytest.approx(ll_n, abs=1e-3)


@pytest.mark.slow
class TestNFXPSolverScalingILRSS2016:
    """Reproduce Iskhakov-Lee-Rust-Schjerning-Seo (2016) on the Rust bus data.

    ILRSS show that Su-Judd's "MPEC is much faster than NFXP" is an artifact of
    NFXP-SA (successive approximations): with the efficient NFXP-NK the inner-loop
    iteration count is insensitive to the discount factor. Measured here on Group
    4, inner iterations per fixed-point solve:

        beta:        0.975   0.99   0.995   0.999   0.9999
        nk    :          8      9       9       9        9
        hybrid:         12     13      13      13       13
        sa    :        844   2104    4202  (slow)  (impractical)

    The "nk" (policy iteration / Newton-Kantorovich) and "hybrid" solves stay flat
    across all beta; NFXP-SA grows sharply. All solvers recover the same MLE.

    The "hybrid" solver switches SA -> NK on a shape-recovery ratio test (the
    successive-error ratio approaching beta, Rust 2000 p.28), which keeps its
    warm-up beta-flat. Before that switch criterion was added it grew ~211 ->
    ~48,513 over this range, dominated by the phase-1 SA warm-up needed to reach
    error < switch_tol=1e-3 (slow at high beta because the constant offset decays
    only at rate beta, while NK strips it in one step).
    """

    @staticmethod
    def _final_inner(beta, solver):
        from econirl.estimation.nfxp import NFXPEstimator

        panel, utility, problem, transitions = _group4_inputs(beta)
        s = NFXPEstimator(
            inner_solver=solver, inner_tol=1e-10, inner_max_iter=2_000_000,
            outer_max_iter=200, verbose=False,
        ).estimate(panel, utility, problem, transitions)
        return s.metadata["final_inner_iterations"], s.parameters[1].item()

    def test_nk_inner_iterations_beta_insensitive_unlike_sa(self):
        sa_lo, _ = self._final_inner(0.975, "sa")
        sa_hi, _ = self._final_inner(0.995, "sa")
        nk_lo, _ = self._final_inner(0.975, "nk")
        nk_hi, _ = self._final_inner(0.9999, "nk")
        # NFXP-NK: inner iterations per solve stay flat across beta (ILRSS).
        assert nk_lo < 30
        assert nk_hi < 30
        # NFXP-SA: inner iterations grow sharply with beta (Su-Judd's slow NFXP).
        assert sa_hi > 3 * sa_lo

    def test_hybrid_switch_stays_beta_flat(self):
        # The hybrid SA->NK switch (shape-recovery ratio test, Rust 2000) keeps the
        # warm-up beta-flat. Before the fix this grew ~211 -> ~48,513 over the range.
        h_lo, _ = self._final_inner(0.975, "hybrid")
        h_hi, _ = self._final_inner(0.9999, "hybrid")
        assert h_lo < 30
        assert h_hi < 30

    def test_sa_and_nk_recover_the_same_estimate(self):
        _, rc_sa = self._final_inner(0.995, "sa")
        _, rc_nk = self._final_inner(0.995, "nk")
        # The inner solver is a numerical choice; the MLE is the same.
        assert rc_sa == pytest.approx(rc_nk, abs=1e-3)


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


class TestMPECStordatProfile:
    """MPEC reproduces the exact Rust Table IX estimates (Su-Judd 2012, Prop 1).

    Su and Judd publish no real-data table (Monte Carlo only). Their
    Proposition 1 is that MPEC and NFXP solve the same MLE, so the matchable
    target is Rust's published Table IX, group 4, beta=0.9999. This locks the
    MPEC replication recorded in project/replications/mpec_rust1987.md.
    """

    @pytest.mark.slow
    def test_mpec_stordat_matches_table_ix(self):
        if not OFFICIAL_GROUP4_RAW.exists():
            pytest.skip("official NFXP archive not downloaded")

        import jax

        jax.config.update("jax_enable_x64", True)
        from econirl.environments.rust_bus import RustBusEnvironment
        from econirl.estimation.mpec import MPECConfig, MPECEstimator
        from econirl.preferences.linear import LinearUtility
        from econirl.replication.rust1987.table_ix import PAPER_TABLE_IX_GROUP4
        from econirl.replication.rust1987.tables import _df_to_panel

        df, metadata = load_stordat_group4_panel(OFFICIAL_GROUP4_RAW)
        panel = _df_to_panel(df)
        probs = tuple(float(metadata["transition_probabilities"][k]) for k in range(3))

        env = RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            num_mileage_bins=int(metadata["n_states"]),
            mileage_transition_probs=probs,
            discount_factor=0.9999,
        )
        utility = LinearUtility.from_environment(env)
        estimator = MPECEstimator(
            config=MPECConfig(solver="sqp"),
            se_method="asymptotic",
            compute_hessian=True,
        )
        result = estimator.estimate(
            panel, utility, env.problem_spec, env.transition_matrices
        )

        paper = PAPER_TABLE_IX_GROUP4[0.9999]
        se = np.asarray(result.standard_errors, dtype=np.float64)

        assert bool(result.converged)
        # Estimates match the published Table IX to four significant figures.
        assert float(result.parameters[0]) * 1000.0 == pytest.approx(
            paper["theta_1_paper_units"], abs=5e-4
        )
        assert float(result.parameters[1]) == pytest.approx(paper["RC"], abs=5e-4)
        assert float(result.log_likelihood) == pytest.approx(
            paper["choice_log_likelihood"], abs=5e-3
        )
        # Standard errors match Table IX to its reported (three-figure) precision.
        assert float(se[0]) * 1000.0 == pytest.approx(
            paper["theta_1_se_paper_units"], abs=1e-3
        )
        assert float(se[1]) == pytest.approx(paper["RC_se"], abs=1e-3)


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
