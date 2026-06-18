"""Tests for the vehicle scrappage simulation study (scripts/study_vehicle_scrappage.py).

Fast tests (no estimator run):
    render_page produces a deterministic, non-empty page from a minimal data
    dict; the structural family shows a Param RMSE column; the behavioral
    family does not get a numerical param RMSE.

Slow test (@pytest.mark.slow):
    NFXP recovers the true RDW theta within RMSE < 0.3 on the study
    environment, simulated and fit through the same path the study uses.
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
            "title": "Vehicle scrappage test",
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
                    "cell_id": "vehicle_scrappage",
                    "label": "Vehicle scrappage (75 states, 2 actions)",
                    "description": "Minimal test cell.",
                    "num_states": 75,
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
                    "parameter_names": [
                        "age_cost", "minor_defect_cost",
                        "major_defect_cost", "replacement_cost",
                    ],
                    "true_theta": [0.15, 0.5, 1.5, 3.0],
                    "diagnostics": {
                        "feature_rank": 4,
                        "num_features": 4,
                        "condition_number": 33.98,
                        "contrast_rank": 4,
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
                "estimator": "NFXP", "family": "structural", "cell": "vehicle_scrappage",
                "rep": 0, "params": [0.141, 0.505, 1.477, 2.872],
                "standard_errors": [0.02, 0.05, 0.10, 0.12],
                "policy_tv": 0.008, "value_rmse": 0.04,
                "regret": {"baseline": 0.003, "type_a": 0.004, "type_b": 0.001,
                           "type_c": 0.001, "transferred": True},
                "runtime": 5.6, "converged": True, "error": None,
            },
            {
                "estimator": "NFXP", "family": "structural", "cell": "vehicle_scrappage",
                "rep": 1, "params": [0.162, 0.518, 1.353, 3.053],
                "standard_errors": [0.02, 0.05, 0.10, 0.12],
                "policy_tv": 0.015, "value_rmse": 0.05,
                "regret": {"baseline": 0.003, "type_a": 0.004, "type_b": 0.001,
                           "type_c": 0.001, "transferred": True},
                "runtime": 5.6, "converged": True, "error": None,
            },
            {
                "estimator": "MCE-IRL", "family": "behavioral", "cell": "vehicle_scrappage",
                "rep": 0, "params": None,
                "standard_errors": None, "policy_tv": 0.008, "value_rmse": None,
                "regret": None, "runtime": 13.6, "converged": False, "error": None,
            },
        ],
    }


# ---- FAST: render_page is a pure function of its inputs ----

def test_render_page_nonempty_and_deterministic():
    from validation.benchmark.harness import render_page
    from scripts.study_vehicle_scrappage import NARRATIVE

    page1 = render_page(_minimal_data(), NARRATIVE)
    page2 = render_page(_minimal_data(), NARRATIVE)
    assert page1 == page2, "render_page is not deterministic"
    assert len(page1) > 200, f"page suspiciously short: {len(page1)} chars"
    assert "scrappage" in page1.lower(), "page content missing 'scrappage'"
    assert "NFXP" in page1, "NFXP missing from rendered page"


def test_render_page_shows_structural_params():
    from validation.benchmark.harness import render_page
    from scripts.study_vehicle_scrappage import NARRATIVE

    assert "Param RMSE" in render_page(_minimal_data(), NARRATIVE)


def test_render_page_hides_irl_params():
    from validation.benchmark.harness import render_page
    from scripts.study_vehicle_scrappage import NARRATIVE

    page = render_page(_minimal_data(), NARRATIVE)
    lines = [l for l in page.splitlines() if "MCE-IRL" in l]
    assert lines, "MCE-IRL row missing from table"


# ---- SLOW: real estimation — structural recovery on the study env ----

@pytest.mark.slow
def test_structural_recovery_on_env():
    """NFXP recovers the true RDW theta within RMSE < 0.3."""
    from econirl.environments.rdw_scrappage import RDWScrapageEnvironment
    from econirl.simulation.synthetic import simulate_panel
    from scripts.study_vehicle_scrappage import _run_nfxp

    env = RDWScrapageEnvironment(discount_factor=0.95, seed=0)
    true_theta = np.asarray(env.get_true_parameter_vector(), dtype=np.float64)
    panel = simulate_panel(env, n_individuals=200, n_periods=35, seed=42)

    res = _run_nfxp(env, panel)
    params = np.asarray(res.parameters, dtype=np.float64)
    assert params.shape == true_theta.shape, f"shape {params.shape}"
    rmse = float(np.sqrt(np.mean((params - true_theta) ** 2)))
    assert rmse < 0.3, (
        f"RMSE {rmse:.4f} >= 0.3. "
        f"recovered={params.tolist()}, true={true_theta.tolist()}"
    )
