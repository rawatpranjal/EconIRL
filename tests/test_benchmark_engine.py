"""Smoke tests for the cross-estimator benchmark engine.

Fast: a tiny cell, a couple of estimators, two replications. Verifies the JSON
shape, that behavioral metrics populate for every estimator, that parameter
bias/coverage populate for the structural family, and that the segmentation
holds (a behavioral-classified estimator gets no parameter metrics).
"""

from __future__ import annotations

import numpy as np

from econirl.environments import random_mdp
from validation.benchmark import metrics as M
from validation.benchmark.cells import BenchmarkCell
from validation.benchmark.runner import EstimatorSpec, _run_ccp, _run_nfxp, run_benchmark


def _tiny_cell() -> BenchmarkCell:
    return BenchmarkCell(
        cell_id="tiny",
        label="Tiny",
        difficulty=0,
        stresses="none",
        description="Tiny abstract MDP for the smoke test.",
        builder=lambda: random_mdp(num_states=8, num_actions=2, num_features=2,
                                   branching=3, discount_factor=0.9, seed=11),
        n_individuals=120,
        n_periods=40,
        seed=1,
    )


def test_metric_helpers() -> None:
    pol = np.array([[0.5, 0.5], [0.2, 0.8]])
    assert M.policy_tv(pol, pol) == 0.0
    assert M.policy_tv(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])) == 1.0
    assert M.value_rmse(None, np.zeros(3)) is None
    assert M.value_rmse(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == 1.0


def test_segmentation_behavioral_gets_no_param_metrics() -> None:
    # Same estimator wrapped twice: once structural, once behavioral.
    estimators = (
        EstimatorSpec("NFXP", "structural", _run_nfxp),
        EstimatorSpec("CCP-behavioral", "behavioral", _run_ccp),
    )
    out = run_benchmark(
        cells=(_tiny_cell(),), estimators=estimators,
        n_replications=2, date="2026-06-11", package_version="test", verbose=False,
    )
    assert out["meta"]["n_replications"] == 2
    assert len(out["cells"]) == 1
    cell = out["cells"][0]
    assert cell["num_states"] == 8
    assert "diagnostics" in cell and "feature_rank" in cell["diagnostics"]

    by_name = {e["estimator"]: e for e in cell["estimators"]}

    nfxp = by_name["NFXP"]
    assert nfxp["behavioral"]["policy_tv_mean"] is not None
    assert nfxp["runtime_mean"] is not None and nfxp["runtime_mean"] > 0
    # Structural -> parameter metrics present, with real coverage.
    assert nfxp["parameters"] is not None
    assert len(nfxp["parameters"]["bias"]) == 2
    assert nfxp["parameters"]["se_available"] is True
    assert all(c is not None for c in nfxp["parameters"]["coverage_95"])

    beh = by_name["CCP-behavioral"]
    assert beh["behavioral"]["policy_tv_mean"] is not None
    # Behavioral -> NO parameter metrics, even though the underlying estimator has them.
    assert beh["parameters"] is None
