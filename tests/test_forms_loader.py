"""Tests for F1: forms factories, make_form dispatcher, and run_form loader.

Coverage:
- make_form dispatch: tabular/grid/factored linear -> correct topology + non-empty true_parameters
- make_form("graph",...) raises ValueError
- RunResult shape: .results is dict, .skipped is list of {name, reason}
- Compatibility skip: tabular-neural form + NFXP -> NFXP in .skipped with reward-form reason
- Known-truth recovery (SLOW): small tabular-linear form, NFXP + CCP both converge and recover params
"""

from __future__ import annotations

import numpy as np
import pytest

from econirl.forms import make_form, run_form, RunResult, Form, FormSpec, CAPABILITIES


# ---------------------------------------------------------------------------
# make_form dispatch
# ---------------------------------------------------------------------------


def test_make_form_tabular_linear():
    form = make_form("tabular", reward_form="linear", num_states=8, num_actions=2, seed=0)
    assert isinstance(form, Form)
    assert form.spec.topology == "tabular"
    assert form.spec.reward_form == "linear"
    assert form.spec.num_states == 8
    assert form.spec.num_actions == 2
    # Must expose non-empty true_parameters for a linear form
    tp = form.env.true_parameters
    assert isinstance(tp, dict)
    assert len(tp) > 0
    # feature_matrix must be 3D (S, A, K)
    phi = np.asarray(form.env.feature_matrix)
    assert phi.ndim == 3
    assert phi.shape[0] == 8 and phi.shape[1] == 2


def test_make_form_grid_linear():
    form = make_form("grid", reward_form="linear", grid_size=4, seed=0)
    assert isinstance(form, Form)
    assert form.spec.topology == "grid"
    assert form.spec.reward_form == "linear"
    assert form.spec.num_states == 16  # 4x4
    assert form.spec.num_actions == 5
    tp = form.env.true_parameters
    assert isinstance(tp, dict) and len(tp) > 0


def test_make_form_factored_linear():
    form = make_form("factored", reward_form="linear", K=2, M=5, seed=0)
    assert isinstance(form, Form)
    assert form.spec.topology == "factored"
    assert form.spec.reward_form == "linear"
    assert form.spec.num_states == 25  # 5^2
    assert form.spec.num_actions == 2
    tp = form.env.true_parameters
    assert isinstance(tp, dict) and len(tp) > 0


def test_make_form_graph_linear():
    # F2 is implemented: graph-linear should succeed, not raise.
    form = make_form("graph", reward_form="linear", num_nodes=10, num_actions=3, seed=0)
    assert isinstance(form, Form)
    assert form.spec.topology == "graph"
    assert form.spec.reward_form == "linear"
    tp = form.env.true_parameters
    assert isinstance(tp, dict) and len(tp) > 0


def test_make_form_unknown_topology_raises():
    with pytest.raises(ValueError, match="unknown topology"):
        make_form("bogus", reward_form="linear")


def test_make_form_tabular_neural():
    # Neural form: no finite theta (true_parameters is empty dict)
    form = make_form("tabular", reward_form="neural", num_states=8, num_actions=2, seed=0)
    assert form.spec.topology == "tabular"
    assert form.spec.reward_form == "neural"
    tp = form.env.true_parameters
    assert isinstance(tp, dict)
    # Neural ShapeshifterEnvironment returns {} for true_parameters
    assert len(tp) == 0


def test_make_form_tabular_nonlinear_raises():
    with pytest.raises(NotImplementedError, match="nonlinear"):
        make_form("tabular", reward_form="nonlinear")


def test_make_form_grid_nonlinear():
    # grid-nonlinear is implemented (ObjectworldEnvironment)
    form = make_form("grid", reward_form="nonlinear", grid_size=4, seed=0)
    assert form.spec.topology == "grid"
    assert form.spec.reward_form == "nonlinear"
    assert form.spec.num_states == 16


# ---------------------------------------------------------------------------
# RunResult shape
# ---------------------------------------------------------------------------


def test_run_result_shape():
    """run_form returns a RunResult with .results dict and .skipped list."""
    rr = RunResult()
    assert isinstance(rr.results, dict)
    assert isinstance(rr.skipped, list)


def test_run_form_returns_run_result():
    """Tiny form, single-estimator smoke test: check the result type."""
    form = make_form("tabular", reward_form="linear", num_states=6, num_actions=2, seed=0)
    rr = run_form(form, estimators=["NFXP"], n_individuals=50, n_periods=10, seed=0)
    assert isinstance(rr, RunResult)
    assert isinstance(rr.results, dict)
    assert isinstance(rr.skipped, list)
    # Every skip entry must have name and reason keys
    for item in rr.skipped:
        assert "name" in item and "reason" in item


# ---------------------------------------------------------------------------
# Compatibility / skip
# ---------------------------------------------------------------------------


def test_compatibility_skip_neural_form_nfxp():
    """NFXP is linear-only; running it on a neural form must land in .skipped."""
    form = make_form(
        "tabular", reward_form="neural", num_states=8, num_actions=2, seed=0
    )
    rr = run_form(form, estimators=["NFXP"], n_individuals=50, n_periods=10, seed=0)
    # NFXP must not appear in results
    assert "NFXP" not in rr.results
    # Must appear in skipped with a reward-form reason
    names_skipped = [s["name"] for s in rr.skipped]
    assert "NFXP" in names_skipped
    nfxp_skip = next(s for s in rr.skipped if s["name"] == "NFXP")
    reason = nfxp_skip["reason"].lower()
    assert "reward" in reason or "form" in reason or "linear" in reason or "neural" in reason


def test_alias_skip():
    """GLADIUS is an alias of NeuralGLADIUS; when both are in the roster,
    GLADIUS should be in .skipped with an alias reason."""
    form = make_form("tabular", reward_form="linear", num_states=6, num_actions=2, seed=0)
    # Only request the alias and its canonical together
    rr = run_form(
        form,
        estimators=["NeuralGLADIUS", "GLADIUS"],
        n_individuals=20, n_periods=5, seed=0,
    )
    names_skipped = [s["name"] for s in rr.skipped]
    # GLADIUS is an alias and must be skipped when NeuralGLADIUS is also in roster
    assert "GLADIUS" in names_skipped
    gladius_skip = next(s for s in rr.skipped if s["name"] == "GLADIUS")
    assert "alias" in gladius_skip["reason"].lower()


def test_unknown_estimator_name_raises():
    form = make_form("tabular", reward_form="linear", num_states=6, num_actions=2, seed=0)
    with pytest.raises(ValueError, match="unknown estimator"):
        run_form(form, estimators=["DOES_NOT_EXIST"])


# ---------------------------------------------------------------------------
# Known-truth recovery (SLOW)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_known_truth_recovery_nfxp_ccp():
    """NFXP and CCP must both converge and recover theta within loose tolerance.

    The MDP is tiny (8 states, 2 actions) so both should finish quickly even
    in the slow-mark tier.  The 0.2 RMSE tolerance is comfortably above the
    ~0.03 actual recovery but well below a failed estimate (all-zeros ~0.57,
    sign-flipped ~1.1), so it genuinely guards recovery.
    """
    form = make_form(
        "tabular", reward_form="linear",
        num_states=8, num_actions=2,
        num_features=2, seed=42,
    )
    rr = run_form(
        form,
        estimators=["NFXP", "CCP"],
        n_individuals=500,
        n_periods=50,
        seed=42,
    )

    # Both must appear in results (not skipped)
    assert "NFXP" in rr.results, f"NFXP missing from results; skipped: {rr.skipped}"
    assert "CCP" in rr.results, f"CCP missing from results; skipped: {rr.skipped}"

    true_theta = np.array(list(form.env.true_parameters.values()))

    for est_name in ("NFXP", "CCP"):
        res = rr.results[est_name]

        # Convergence flag
        converged = getattr(res, "converged", None)
        assert converged, f"{est_name}: converged={converged!r}"

        # Recovered parameters
        params = getattr(res, "parameters", None)
        assert params is not None, f"{est_name}: parameters is None"
        recovered = np.asarray(params).reshape(-1)

        # Must have the same number of parameters
        assert recovered.shape == true_theta.shape, (
            f"{est_name}: shape mismatch {recovered.shape} vs {true_theta.shape}"
        )

        # Recovery must be genuine. Actual RMSE is ~0.03; 0.2 still catches a
        # regression to garbage (all-zeros ~0.57, sign-flipped ~1.1 both fail).
        rmse = float(np.sqrt(np.mean((recovered - true_theta) ** 2)))
        assert rmse < 0.2, (
            f"{est_name}: RMSE={rmse:.4f} vs true_theta={true_theta}, "
            f"recovered={recovered}"
        )
        print(
            f"  {est_name}: true={true_theta.tolist()}, "
            f"recovered={recovered.tolist()}, RMSE={rmse:.4f}"
        )
