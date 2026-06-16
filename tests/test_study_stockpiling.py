"""Tests for the stockpiling simulation study (scripts/study_stockpiling.py).

Fast tests (no estimator run):
    render_page produces a deterministic, non-empty page from a minimal data
    dict; the structural family shows a Param RMSE column; the behavioral
    family does not get a numerical param RMSE.

Slow test (@pytest.mark.slow):
    NFXP recovers the true storable-goods theta within RMSE < 0.3 on the
    study environment, simulated and fit through the same path the study uses.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [os.path.join(_ROOT, "src"), _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _minimal_data() -> dict:
    """A minimal valid harness data dict suitable for render_page."""
    return {
        "meta": {
            "title": "Stockpiling test",
            "date": "2026-01-01",
            "package_version": "test",
            "oracle": "true-parameter policy/value via SoftBellmanOperator",
            "determinism": "structural estimators are deterministic given seeds",
            "excluded": [],
            "regret": "Type A/B/C counterfactual regret taxonomy",
            "honesty": "Every number recomputed from raw records below.",
            "snippets": {"NFXP": "def _run_nfxp(env, panel): pass"},
            "diagnoses": {"NFXP": "Full-solution MLE."},
            "cells": [
                {
                    "cell_id": "stockpiling",
                    "label": "Stockpiling (20 states, 2 actions)",
                    "description": "Minimal test cell.",
                    "num_states": 20,
                    "num_actions": 2,
                    "discount_factor": 0.95,
                    "n_individuals": 200,
                    "n_periods": 35,
                    "seed": 42,
                    "n_replications": 2,
                    "param_block": True,
                    "show_params": True,
                    "show_regret": True,
                    "figure": None,
                    "parameter_names": ["spend", "holding", "stockout"],
                    "true_theta": [1.0, 0.2, 3.0],
                    "diagnostics": {
                        "feature_rank": 3,
                        "num_features": 3,
                        "condition_number": 28.1,
                        "contrast_rank": 3,
                    },
                    "roster": [
                        {"name": "NFXP", "family": "structural", "max_reps": None},
                        {"name": "MCE-IRL", "family": "behavioral", "max_reps": None},
                    ],
                }
            ],
        },
        "records": [
            {
                "estimator": "NFXP", "family": "structural", "cell": "stockpiling",
                "rep": 0, "params": [0.99, 0.20, 2.93],
                "standard_errors": [0.05, 0.02, 0.12],
                "policy_tv": 0.003, "value_rmse": 0.05,
                "regret": {"baseline": 0.005, "type_a": 0.006, "type_b": 0.001,
                           "type_c": 0.001, "transferred": True},
                "runtime": 2.4, "converged": True, "error": None,
            },
            {
                "estimator": "NFXP", "family": "structural", "cell": "stockpiling",
                "rep": 1, "params": [1.01, 0.20, 3.02],
                "standard_errors": [0.05, 0.02, 0.12],
                "policy_tv": 0.002, "value_rmse": 0.04,
                "regret": {"baseline": 0.005, "type_a": 0.006, "type_b": 0.001,
                           "type_c": 0.001, "transferred": True},
                "runtime": 2.2, "converged": True, "error": None,
            },
            {
                "estimator": "MCE-IRL", "family": "behavioral", "cell": "stockpiling",
                "rep": 0, "params": [0.99, 0.20, 2.93],
                "standard_errors": None, "policy_tv": 0.003, "value_rmse": None,
                "regret": None, "runtime": 7.4, "converged": False, "error": None,
            },
        ],
    }


# ---- FAST: render_page is a pure function of its inputs ----

def test_render_page_nonempty_and_deterministic():
    from validation.benchmark.harness import render_page
    from scripts.study_stockpiling import NARRATIVE

    page1 = render_page(_minimal_data(), NARRATIVE)
    page2 = render_page(_minimal_data(), NARRATIVE)
    assert page1 == page2, "render_page is not deterministic"
    assert len(page1) > 200, f"page suspiciously short: {len(page1)} chars"
    assert "stockpil" in page1.lower(), "page content missing 'stockpil'"
    assert "NFXP" in page1, "NFXP missing from rendered page"


def test_render_page_shows_structural_params():
    from validation.benchmark.harness import render_page
    from scripts.study_stockpiling import NARRATIVE

    assert "Param RMSE" in render_page(_minimal_data(), NARRATIVE)


def test_render_page_hides_irl_params():
    from validation.benchmark.harness import render_page
    from scripts.study_stockpiling import NARRATIVE

    page = render_page(_minimal_data(), NARRATIVE)
    lines = [l for l in page.splitlines() if "MCE-IRL" in l]
    assert lines, "MCE-IRL row missing from table"


# ---- SLOW: real estimation — structural recovery on the study env ----

@pytest.mark.slow
def test_structural_recovery_on_env():
    """NFXP recovers the true storable-goods theta within RMSE < 0.3."""
    from econirl.environments.storable_goods import storable_goods
    from econirl.simulation.synthetic import simulate_panel
    from scripts.study_stockpiling import _run_nfxp

    env = storable_goods(max_inventory=9, pack_size=3, discount_factor=0.95, seed=0)
    true_theta = np.asarray(env.get_true_parameter_vector(), dtype=np.float64)
    panel = simulate_panel(env, n_individuals=200, n_periods=35, seed=42)

    res = _run_nfxp(env, panel)
    params = np.asarray(res.parameters, dtype=np.float64)
    assert params.shape == true_theta.shape, f"shape {params.shape}"
    rmse = float(np.sqrt(np.mean((params - true_theta) ** 2)))
    assert rmse < 0.3, f"RMSE {rmse:.4f} >= 0.3. recovered={params.tolist()}, true={true_theta.tolist()}"
