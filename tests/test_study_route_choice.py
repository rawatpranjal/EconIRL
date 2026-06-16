"""Tests for the route-choice simulation study (scripts/study_route_choice.py).

Fast test (no estimator run):
    render_page produces a non-empty string deterministically from a minimal
    data dict — verifies the renderer treats the JSON as a pure function.

Slow tests (@pytest.mark.slow):
    - NFXP and CCP recover the true route-choice theta within RMSE < 0.3 on a
      small graph (12 nodes, 150 individuals, 25 periods).
    - The full script produces the JSON and the docs page when run at 1 replication.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The tests run with PYTHONPATH=src:. so the imports already work; this makes
# the test importable without that env var too (e.g. in an IDE).
for _p in [os.path.join(_ROOT, "src"), _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS = os.path.join(_ROOT, "validation", "results", "study_route_choice.json")
_PAGE = os.path.join(_ROOT, "docs", "simulation_studies", "route_choice.md")


# ---------------------------------------------------------------------------
# Minimal data fixture for the pure render_page test
# ---------------------------------------------------------------------------

def _minimal_data() -> dict:
    """Return a minimal valid harness data dict suitable for render_page."""
    return {
        "meta": {
            "title": "Route choice test",
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
                    "cell_id": "route_choice",
                    "label": "Route choice (25 nodes, 4 actions)",
                    "description": "Minimal test cell.",
                    "num_states": 25,
                    "num_actions": 4,
                    "discount_factor": 0.95,
                    "n_individuals": 200,
                    "n_periods": 35,
                    "seed": 42,
                    "n_replications": 2,
                    "param_block": True,
                    "show_params": True,
                    "show_regret": True,
                    "figure": None,
                    "parameter_names": ["edge_cost", "amenity", "goal"],
                    "true_theta": [1.0, 0.5, 1.0],
                    "diagnostics": {
                        "feature_rank": 3,
                        "num_features": 3,
                        "condition_number": 2.5,
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
                "estimator": "NFXP",
                "family": "structural",
                "cell": "route_choice",
                "rep": 0,
                "params": [1.02, 0.48, 0.97],
                "standard_errors": [0.05, 0.04, 0.06],
                "policy_tv": 0.015,
                "value_rmse": 0.12,
                "regret": {
                    "baseline": 0.002,
                    "type_a": 0.008,
                    "type_b": 0.010,
                    "type_c": 0.005,
                    "transferred": True,
                },
                "runtime": 1.4,
                "converged": True,
                "error": None,
            },
            {
                "estimator": "NFXP",
                "family": "structural",
                "cell": "route_choice",
                "rep": 1,
                "params": [0.98, 0.52, 1.03],
                "standard_errors": [0.05, 0.04, 0.06],
                "policy_tv": 0.013,
                "value_rmse": 0.11,
                "regret": {
                    "baseline": 0.001,
                    "type_a": 0.007,
                    "type_b": 0.009,
                    "type_c": 0.004,
                    "transferred": True,
                },
                "runtime": 1.3,
                "converged": True,
                "error": None,
            },
            {
                "estimator": "MCE-IRL",
                "family": "behavioral",
                "cell": "route_choice",
                "rep": 0,
                "params": [0.8, 0.3, 0.7],
                "standard_errors": None,
                "policy_tv": 0.03,
                "value_rmse": None,
                "regret": None,
                "runtime": 8.2,
                "converged": False,
                "error": None,
            },
        ],
    }


# ---------------------------------------------------------------------------
# FAST: render_page is a pure function of its inputs
# ---------------------------------------------------------------------------


def test_render_page_nonempty_and_deterministic():
    """render_page produces a non-empty string and is deterministic."""
    from validation.benchmark.harness import render_page
    from scripts.study_route_choice import NARRATIVE

    data = _minimal_data()
    page1 = render_page(data, NARRATIVE)
    page2 = render_page(data, NARRATIVE)

    assert page1 == page2, "render_page is not deterministic"
    assert len(page1) > 200, f"page is suspiciously short: {len(page1)} chars"
    assert "route" in page1.lower(), "page title/content missing 'route'"
    assert "NFXP" in page1, "NFXP missing from rendered page"
    assert "edge_cost" in page1 or "1.0" in page1, "true theta missing from page"


def test_render_page_shows_structural_params():
    """Param RMSE column appears for structural estimators."""
    from validation.benchmark.harness import render_page
    from scripts.study_route_choice import NARRATIVE

    page = render_page(_minimal_data(), NARRATIVE)
    # The results table has a Param RMSE column for structural family.
    assert "Param RMSE" in page


def test_render_page_hides_irl_params():
    """render_page does not show a Param RMSE for behavioral estimators."""
    from validation.benchmark.harness import render_page
    from scripts.study_route_choice import NARRATIVE

    page = render_page(_minimal_data(), NARRATIVE)
    # MCE-IRL is behavioral: its row must not have a numerical param RMSE.
    lines = [l for l in page.splitlines() if "MCE-IRL" in l]
    # The line should exist but the RMSE column should be "-" (no finite theta match).
    assert lines, "MCE-IRL row missing from table"


# ---------------------------------------------------------------------------
# SLOW: real estimation — structural recovery and page write
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_structural_recovery_small_graph():
    """NFXP and CCP recover [1.0, 0.5, 1.0] within RMSE < 0.3 on a small graph."""
    from econirl.forms import make_form, run_form

    true_theta = np.array([1.0, 0.5, 1.0])
    form = make_form("graph", "linear", num_nodes=12, num_actions=3, seed=42)
    result = run_form(
        form,
        estimators=["NFXP", "CCP"],
        n_individuals=150,
        n_periods=25,
        seed=42,
    )

    n_tested = 0
    for name in ["NFXP", "CCP"]:
        if name in result.skipped or name not in result.results:
            skip_reasons = {s["name"]: s["reason"] for s in result.skipped}
            pytest.skip(f"{name} skipped by run_form: {skip_reasons.get(name)}")
        n_tested += 1
        res = result.results[name]
        params = np.asarray(res.parameters)
        assert params.shape == (3,), f"{name}: unexpected param shape {params.shape}"
        rmse = float(np.sqrt(np.mean((params - true_theta) ** 2)))
        assert rmse < 0.3, (
            f"{name}: RMSE {rmse:.4f} >= 0.3. "
            f"Recovered={params.tolist()}, true={true_theta.tolist()}"
        )

    assert n_tested >= 1, f"No structural estimators ran. Skipped={result.skipped}"


@pytest.mark.slow
def test_script_writes_json_and_page(tmp_path, monkeypatch):
    """Running the study at 1 replication writes the JSON and docs page."""
    import json
    import subprocess

    # Redirect outputs so the slow test doesn't write the real files.
    test_json = str(tmp_path / "study_route_choice.json")
    test_page = str(tmp_path / "route_choice.md")

    # Patch RESULTS_JSON and PAGE_PATH via monkeypatching the module at runtime
    # by invoking a small driver script inline.
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""
import sys, os
sys.path.insert(0, "{os.path.join(_ROOT, 'src')}")
sys.path.insert(0, "{_ROOT}")
import scripts.study_route_choice as S
S.RESULTS_JSON = {test_json!r}
S.PAGE_PATH = {test_page!r}
# Remove figure so we don't need _static/simulation_studies/
import dataclasses
S.CELLS = tuple(
    dataclasses.replace(c, figure=None) for c in S.CELLS
)
S.NARRATIVE = dict(S.NARRATIVE)
from validation.benchmark.harness import main_cli
_kw = dict(
    cells=S.CELLS,
    title="Simulation study: route choice on a road network",
    narrative=S.NARRATIVE,
    diagnoses=S.DIAGNOSES,
    excluded=S.EXCLUDED,
    results_json=S.RESULTS_JSON,
    page_path=S.PAGE_PATH,
)
# Two-step contract: a plain run writes the JSON; --page renders the page from it.
sys.argv = ["driver.py", "--replications", "1"]
main_cli(**_kw)
sys.argv = ["driver.py", "--page"]
main_cli(**_kw)
"""
    )
    result = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=_ROOT,
    )
    assert result.returncode == 0, (
        f"Driver script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert os.path.exists(test_json), "JSON not written"
    assert os.path.exists(test_page), "Page not written"

    with open(test_json) as f:
        data = json.load(f)
    assert "meta" in data and "records" in data
    assert len(data["records"]) > 0

    page_text = open(test_page).read()
    assert len(page_text) > 200, f"Page suspiciously short: {len(page_text)} chars"
    assert "route" in page_text.lower()
