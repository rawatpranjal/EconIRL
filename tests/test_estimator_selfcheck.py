"""Tests for estimator self-check failure detection."""

from __future__ import annotations

import numpy as np
import pytest

from econirl.evaluation.selfcheck import assert_effect, check_coverage, check_se_ratio


def test_coverage_rejects_intervals_that_are_too_narrow() -> None:
    def make_data(rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=100)

    def estimate(data: np.ndarray) -> tuple[float, float, float]:
        mean = float(data.mean())
        half_width = 0.20 * 1.96 / np.sqrt(data.size)
        return mean, mean - half_width, mean + half_width

    with pytest.raises(AssertionError, match="coverage self-check failed"):
        check_coverage(
            make_data,
            estimate,
            0.0,
            n_sims=300,
            tol=0.04,
            seed=3,
        )


def test_se_ratio_rejects_large_disagreement() -> None:
    with pytest.raises(AssertionError, match="SE self-check failed"):
        check_se_ratio(0.5, 1.0, tol=0.25)


def test_effect_check_rejects_vacuous_demo() -> None:
    with pytest.raises(AssertionError, match="demo self-check failed"):
        assert_effect(0.001, min_abs=0.01)
