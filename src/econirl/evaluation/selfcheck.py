"""Numerical self-checks for estimator inference and demonstrations."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def check_coverage(
    make_data: Callable[[np.random.Generator], Any],
    estimate: Callable[[Any], tuple[float, float, float]],
    truth: float,
    *,
    n_sims: int = 1000,
    level: float = 0.95,
    tol: float = 0.015,
    seed: int = 0,
) -> float:
    """Check both tail-error rates of a claimed confidence interval."""
    rng = np.random.default_rng(seed)
    target = (1.0 - level) / 2.0
    left = 0
    right = 0
    for _ in range(n_sims):
        _point, lower, upper = estimate(make_data(rng))
        if truth < lower:
            left += 1
        elif truth > upper:
            right += 1

    left_rate = left / n_sims
    right_rate = right / n_sims
    coverage = 1.0 - left_rate - right_rate
    problems = []
    if abs(left_rate - target) > tol:
        problems.append(f"left-tail miss {left_rate:.3f} vs {target:.3f}")
    if abs(right_rate - target) > tol:
        problems.append(f"right-tail miss {right_rate:.3f} vs {target:.3f}")
    if problems:
        raise AssertionError(
            "coverage self-check failed "
            f"({'; '.join(problems)}; total coverage {coverage:.3f}, "
            f"nominal {level:.2f}, n_sims {n_sims}, tol {tol})"
        )
    return coverage


def check_se_ratio(
    formula_se: float,
    bootstrap_se: float,
    *,
    tol: float = 0.15,
) -> float:
    """Check that a formula and bootstrap standard error agree."""
    if formula_se <= 0 or bootstrap_se <= 0:
        raise ValueError(
            "standard errors must be positive "
            f"(got {formula_se} and {bootstrap_se})"
        )
    ratio = formula_se / bootstrap_se
    if abs(ratio - 1.0) > tol:
        raise AssertionError(
            "SE self-check failed: "
            f"formula/bootstrap ratio {ratio:.3f} exceeds tolerance {tol}"
        )
    return ratio


def assert_effect(observed: float, *, min_abs: float) -> float:
    """Require a demonstration to have a finite, nontrivial effect."""
    if not math.isfinite(observed):
        raise ValueError(f"effect is not finite: {observed}")
    if abs(observed) < min_abs:
        raise AssertionError(
            "demo self-check failed: "
            f"absolute effect {abs(observed):.4g} is below {min_abs:.4g}"
        )
    return observed
