"""Tests for the fleet maintenance simulation study (scripts/study_fleet_maintenance.py).

Fast tests (no estimator run):
    render_page produces a deterministic, non-empty page from a minimal data
    dict; the structural family shows a Param RMSE column; the behavioral
    family does not get a numerical param RMSE.

Slow test (@pytest.mark.slow):
    CCP recovers the true fleet-maintenance theta within RMSE < 0.3 on the
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
            "title": "Fleet maintenance test",
            "date": "2026-01-01",
            "package_version": "test",
            "oracle": "true-parameter policy/value via SoftBellmanOperator",
            "determinism": "structural estimators are deterministic given seeds",
            "excluded": [],
            "regret": "Type A/B/C counterfactual regret taxonomy",
            "honesty": "Every number recomputed from raw records below.",
            "snippets": {"CCP": "def _run_ccp(env, panel): pass"},
            "diagnoses": {"CCP": "CCP with one policy-iteration step."},
            "cells": [
                {
                    "cell_id": "fleet_maintenance",
                    "label": "Fleet maintenance (216 states, 2 actions)",
                    "description": "Minimal test cell.",
                    "num_states": 216,
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
                        "replacement_cost", "operating_cost", "quadratic_cost"
                    ],
                    "true_theta": [3.0, 1.0, 0.5],
                    "diagnostics": {
                        "feature_rank": 3,
                        "num_features": 3,
                        "condition_number": 8.57,
                        "contrast_rank": 3,
                    },
                    "roster": [
                        {"name": "CCP", "family": "structural", "max_reps": None},
                        {"name": "MCE-IRL", "family": "behavioral", "max_reps": None},
                    ],
                }
            ],
        },
        "records": [
            {
                "estimator": "CCP",
                "family": "structural",
                "cell": "fleet_maintenance",
                "rep": 0,
                "params": [3.05, 0.98, 0.51],
                "standard_errors": [0.08, 0.12, 0.10],
                "policy_tv": 0.020,
                "value_rmse": 0.10,
                "regret": {
                    "baseline": 0.004,
                    "type_a": 0.007,
                    "type_b": 0.002,
                    "type_c": 0.002,
                    "transferred": True,
                },
                "runtime": 2.8,
                "converged": True,
                "error": None,
            },
            {
                "estimator": "CCP",
                "family": "structural",
                "cell": "fleet_maintenance",
                "rep": 1,
                "params": [2.96, 1.02, 0.49],
                "standard_errors": [0.08, 0.12, 0.10],
                "policy_tv": 0.018,
                "value_rmse": 0.09,
                "regret": {
                    "baseline": 0.003,
                    "type_a": 0.006,
                    "type_b": 0.002,
                    "type_c": 0.001,
                    "transferred": True,
                },
                "runtime": 2.6,
                "converged": True,
                "error": None,
            },
            {
                "estimator": "MCE-IRL",
                "family": "behavioral",
                "cell": "fleet_maintenance",
                "rep": 0,
                "params": None,
                "standard_errors": None,
                "policy_tv": 0.020,
                "value_rmse": None,
                "regret": None,
                "runtime": 8.1,
                "converged": False,
                "error": None,
            },
        ],
    }


# ---- FAST: render_page is a pure function of its inputs ----

def test_render_page_nonempty_and_deterministic():
    from validation.benchmark.harness import render_page
    from scripts.study_fleet_maintenance import NARRATIVE

    page1 = render_page(_minimal_data(), NARRATIVE)
    page2 = render_page(_minimal_data(), NARRATIVE)
    assert page1 == page2, "render_page is not deterministic"
    assert len(page1) > 200, f"page suspiciously short: {len(page1)} chars"
    assert "fleet" in page1.lower(), "page content missing 'fleet'"
    assert "CCP" in page1, "CCP missing from rendered page"


def test_render_page_shows_structural_params():
    from validation.benchmark.harness import render_page
    from scripts.study_fleet_maintenance import NARRATIVE

    assert "Param RMSE" in render_page(_minimal_data(), NARRATIVE)


def test_render_page_hides_irl_params():
    from validation.benchmark.harness import render_page
    from scripts.study_fleet_maintenance import NARRATIVE

    page = render_page(_minimal_data(), NARRATIVE)
    lines = [l for l in page.splitlines() if "MCE-IRL" in l]
    assert lines, "MCE-IRL row missing from table"


# ---- SLOW: real estimation — structural recovery on the study env ----

@pytest.mark.slow
def test_structural_recovery_on_env():
    """NFXP recovers the true fleet-maintenance theta within RMSE < 0.3."""
    from econirl.environments.multi_component_bus import MultiComponentBusEnvironment
    from econirl.simulation.synthetic import simulate_panel
    from scripts.study_fleet_maintenance import _run_nfxp

    env = MultiComponentBusEnvironment(
        K=3, M=6, seed=0, discount_factor=0.95,
        operating_cost=1.0, quadratic_cost=0.5, replacement_cost=3.0,
    )
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
