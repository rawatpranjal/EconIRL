"""Release contracts for MCE-IRL repeated-run and road validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_ziebart_smoke_run_renders_real_route_comparison(tmp_path: Path) -> None:
    from matplotlib.image import imread

    from validation.estimators.mce_irl.ziebart_road_synthetic import (
        SMOKE_SHAPE,
        run,
    )

    result_path = tmp_path / "result.json"
    figure_path = tmp_path / "road.png"
    payload = run(
        SMOKE_SHAPE,
        output=result_path,
        figure_output=figure_path,
    )

    image = imread(figure_path)
    assert payload["passed"], "smoke study must pass before its figure is trusted"
    assert payload["figure"]["selection_rule"] == (
        "held-out route nearest the median distance match"
    )
    assert 0.0 <= payload["figure"]["distance_match_percent"] <= 100.0
    assert min(image.shape[:2]) >= 900
    assert image.shape[0] > image.shape[1], "stacked panels must remain legible on RTD"
    assert float(image[..., :3].std()) > 0.05, "figure must contain visible route content"


def test_ziebart_smoke_defaults_do_not_target_tracked_figure() -> None:
    from validation.estimators.mce_irl.ziebart_road_synthetic import (
        DEFAULT_FIGURE,
        resolve_figure_output,
    )

    assert resolve_figure_output(smoke=True, requested=None) is None
    assert resolve_figure_output(smoke=False, requested=None) == DEFAULT_FIGURE


def test_mce_irl_repeated_run_receipt_is_ready() -> None:
    payload = json.loads(Path("validation/results/mce_irl_ready.json").read_text(encoding="utf-8"))

    assert payload["status"] == "ready"
    assert payload["n_replications"] >= 300
    repeated = payload["repeated_run_inference"]
    assert repeated["n_successful"] == payload["n_replications"]
    assert repeated["coverage_selfcheck_passed"]
    assert 0.91 <= repeated["coverage"] <= 0.99
    assert repeated["lower_tail_miss_rate"] + repeated["upper_tail_miss_rate"] == pytest.approx(
        1.0 - repeated["coverage"]
    )
    assert abs(repeated["bias"]) <= 0.05
    assert repeated["max_stationarity_residual"] <= 1e-6
    assert repeated["mean_asymptotic_se"] > 0
    assert 0.75 <= repeated["mean_se_to_empirical_sd"] <= 1.25
    assert payload["standard_error_check"]["passed"]
    assert payload["intervention_check"]["passed"]
    assert all(gate["passed"] for gate in payload["gates"])
    assert payload["thresholds"]["coverage_low"] == 0.91
    assert payload["provenance"]["git_commit"]
    assert payload["provenance"]["package_versions"]["econirl"]


def test_mce_irl_bootstrap_calibration_receipt_is_ready() -> None:
    payload = json.loads(
        Path("validation/results/mce_irl_bootstrap_calibration.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "ready"
    assert payload["design"]["resampling_unit"] == "whole individual trajectory"
    summary = payload["summary"]
    assert summary["n_total"] == 50
    assert summary["n_usable"] == 50
    assert summary["total_successful_draws"] == 4_950
    assert summary["total_failed_draws"] == 0
    assert 0.86 <= summary["coverage_95"] <= 1.0
    assert summary["lower_tail_miss_rate"] + summary["upper_tail_miss_rate"] == pytest.approx(
        1.0 - summary["coverage_95"]
    )
    assert summary["minimum_interval_width"] > 0
    assert payload["program_check"]["passed"]
    assert all(gate["passed"] for gate in payload["gates"] if gate["enforced"])


def test_ziebart_synthetic_receipt_separates_shape_from_replication() -> None:
    payload = json.loads(
        Path("validation/results/mce_irl_ziebart_synthetic.json").read_text(encoding="utf-8")
    )

    assert payload["passed"]
    assert "generated-data" in payload["claim"]
    assert "does not reproduce Table 1" in payload["claim"]
    assert payload["network"]["num_states"] > 300_000
    assert payload["network"]["num_action_slots"] > 900_000
    assert payload["network"]["state_definition"] == (
        "directed road segment ending at an intersection"
    )
    assert len(payload["network"]["raw_feature_names"]) == 22
    assert payload["network"]["normalization"]["rule"] == (
        "pivoted full-rank action-contrast basis"
    )
    checks = payload["network"]["shape_checks"]
    assert checks["spatial_segment_topology"]
    assert checks["raw_feature_counts"] == 22
    assert checks["identified_fit_features"] == len(payload["network"]["fit_feature_names"])
    assert checks["filter_reasons"] == {
        "too_short": 1_322,
        "cyclic": 1_322,
        "noisy": 1_322,
    }
    assert payload["data"]["raw_trips"] == 13_220
    assert payload["data"]["discarded_trips"] == 3_966
    assert payload["data"]["train_trips"] == 1_851
    assert payload["data"]["test_trips"] == 7_403
    assert payload["data"]["drivers"] == 25
    assert payload["fit"]["converged"]
    assert not payload["fit"]["road_specific_estimator_logic"]
    assert payload["fit"]["contrast_rank"] == payload["fit"]["num_parameters"]
    assert payload["fit"]["stationarity_residual"] <= 0.02
    assert np.isfinite(list(payload["synthetic_metrics"].values())).all()
    assert payload["metric_semantics"]["average_log_probability"] == (
        "mean fitted log probability on the training paths"
    )
    assert payload["paper"]["reported_table_1_targets"] == {
        "distance_match_percent": 78.79,
        "routes_at_least_90_percent": 52.98,
        "average_log_probability": -6.85,
    }
    assert payload["figure"]["path"] == ("docs/_static/estimators/mce_irl_ziebart_road.png")
    assert payload["figure"]["selection_rule"] == (
        "held-out route nearest the median distance match"
    )
    assert Path(payload["figure"]["path"]).is_file()


def test_ziebart_public_page_uses_receipt_values_and_three_pillars() -> None:
    payload = json.loads(
        Path("validation/results/mce_irl_ziebart_synthetic.json").read_text(encoding="utf-8")
    )
    page = Path("docs/estimators/mce_irl/validation.md").read_text(encoding="utf-8")

    assert "## Three Checks" in page
    assert "## Road Estimation" in page
    assert "## Monte Carlo Inference" in page
    assert "## Counterfactuals" in page
    assert "../../_static/estimators/mce_irl_ziebart_road.png" in page
    assert "The upper panel places all tasks" in page
    assert "The lower panel" in page
    assert "The left panel places all tasks" not in page
    assert "The right panel" not in page
    for value in payload["synthetic_metrics"].values():
        assert f"{value:.2f}" in page, f"missing generated metric {value:.2f}"
    for value in payload["paper_target_gaps"].values():
        assert f"{value:+.2f}" in page, f"missing paper gap {value:+.2f}"
