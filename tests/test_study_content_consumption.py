"""Tests for the content-consumption simulation study.

Fast tests (no estimator run):
    render_page produces a deterministic, non-empty page from a minimal data
    dict; the heterogeneous headline row and the homogeneous-baseline rows are
    present; the page avoids the banned public-prose register.

Slow test (@pytest.mark.slow):
    AIRL-Het recovers the two latent reader segments on a small panel of the
    same DGP the study uses: segment assignment accuracy > 0.7 after label
    matching.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [os.path.join(_ROOT, "src"), _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _minimal_data() -> dict:
    """A minimal valid data dict suitable for render_page (no estimator run)."""
    return {
        "meta": {
            "title": "Content consumption test",
            "date": "2026-01-01",
            "package_version": "test",
            "n_states": 61,
            "n_actions": 3,
            "exit_action": 2,
            "absorbing_state": 60,
            "num_segments": 2,
            "num_chapters": 5,
            "n_individuals": 50,
            "n_periods": 16,
            "n_trajectories": 200,
            "n_observations": 800,
            "true_priors": [0.48, 0.52],
            "segment_names": ["binge reader", "patient reader"],
            "action_names": ["pay", "wait", "exit"],
            "true_segment_gap_tv": 0.31,
            "diagnostics": {"feature_rank": 20, "num_features": 20},
            "discount_factor": 0.92,
            "scale_parameter": 0.85,
            "dgp_kw": {"books_per_user": 4},
            "sim": {"n_individuals": 50, "n_periods": 16, "seed": 1},
        },
        "headline": {
            "name": "AIRL-Het",
            "error": None,
            "assignment_accuracy": 0.88,
            "prior_l1": 0.04,
            "aligned_priors": [0.50, 0.50],
            "segment_reward_nrmse": [0.24, 0.27],
            "max_segment_reward_nrmse": 0.27,
            "segment_policy_tv": [0.04, 0.05],
            "max_segment_policy_tv": 0.05,
            "segment_value_nrmse": [0.09, 0.14],
            "pooled_policy_tv": 0.03,
            "confusion": [[400, 80, 20], [120, 180, 30], [40, 50, 80]],
            "confusion_accuracy": 0.58,
            "confusion_ceiling_accuracy": 0.57,
            "permutation": [1, 0],
            "runtime": 14.0,
            "converged": True,
            "em_iterations": 2,
        },
        "baselines": [
            {"name": "BC", "family": "behavioral", "recovers_segments": False,
             "error": None, "pooled_policy_tv": 0.05,
             "segment_policy_tv": [0.18, 0.15], "max_segment_policy_tv": 0.18,
             "runtime": 0.1, "converged": True},
            {"name": "NFXP", "family": "structural", "recovers_segments": False,
             "error": None, "pooled_policy_tv": 0.04,
             "segment_policy_tv": [0.18, 0.14], "max_segment_policy_tv": 0.18,
             "runtime": 3.0, "converged": True},
        ],
    }


# ---- FAST: render_page is a pure function of its inputs ----

def test_render_page_nonempty_and_deterministic():
    from scripts.study_content_consumption import render_page

    page1 = render_page(_minimal_data())
    page2 = render_page(_minimal_data())
    assert page1 == page2, "render_page is not deterministic"
    assert len(page1) > 400, f"page suspiciously short: {len(page1)} chars"
    assert "AIRL-Het" in page1
    assert "segment" in page1.lower()


def test_render_page_shows_headline_and_baselines():
    from scripts.study_content_consumption import render_page

    page = render_page(_minimal_data())
    # heterogeneous headline metrics present
    assert "assignment accuracy" in page.lower()
    assert "binge reader" in page and "patient reader" in page
    # homogeneous baselines present in the comparison table
    assert "NFXP" in page and "BC" in page
    # confusion matrix present
    assert "pay" in page.lower() and "wait" in page.lower() and "exit" in page.lower()


def test_render_page_records_a_crashed_baseline():
    from scripts.study_content_consumption import render_page

    data = _minimal_data()
    data["baselines"][1] = {
        "name": "NFXP", "family": "structural", "recovers_segments": False,
        "error": "RuntimeError: boom", "runtime": 1.0,
    }
    page = render_page(data)
    assert "crashed" in page.lower()
    assert "boom" in page


def test_render_page_avoids_banned_register():
    """The public page must not use the banned RTD register words."""
    from scripts.study_content_consumption import render_page

    page = render_page(_minimal_data()).lower()
    for word in ["known truth", "known-truth", "artifact", "honest", "frozen",
                 "verbatim", "gauge", "certified as", "release claim"]:
        assert word not in page, f"banned register word in page: {word!r}"


# ---- SLOW: real estimation — segment recovery on the study DGP ----

@pytest.mark.slow
def test_airl_het_recovers_two_segments():
    """AIRL-Het assignment accuracy > 0.7 on a small panel of the study DGP."""
    from validation.known_truth import (
        ContentHeterogeneityKnownTruthConfig,
        SimulationConfig,
        build_known_truth_dgp,
        simulate_known_truth_panel,
        evaluate_segmented_estimator_against_truth,
    )
    from econirl.estimation.adversarial.airl_het import AIRLHetEstimator
    from scripts.study_content_consumption import DGP_KW, _airl_het_config

    dgp = build_known_truth_dgp(ContentHeterogeneityKnownTruthConfig(**DGP_KW))
    # smaller panel than the full study to keep the test fast
    panel = simulate_known_truth_panel(
        dgp, SimulationConfig(n_individuals=350, n_periods=16, seed=11)
    )
    summary = AIRLHetEstimator(_airl_het_config(dgp)).estimate(
        panel, dgp.utility(), dgp.problem, dgp.transitions
    )
    metrics = evaluate_segmented_estimator_against_truth(
        dgp, summary, panel=panel, counterfactual_kinds=()
    )
    acc = metrics["segment_assignment_accuracy"]
    assert acc is not None and acc > 0.7, f"assignment accuracy {acc} <= 0.7"
