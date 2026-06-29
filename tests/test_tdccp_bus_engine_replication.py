"""Lock the TD-CCP bus-engine replication (Adusumilli-Eckardt 2025, Table B.1).

Two checks:
  1. The encoder-based logit CCP recovers all three bus-engine structural
     parameters within tolerance (theta0, theta1, theta2) on a single MC draw.
  2. The bug fix: the encoder CCP recovers the bus-type coefficient theta2, while
     the scalar-index CCP (ccp_use_encoder=False) conflates mileage and type and
     badly biases it. Without the fix, theta2 came out near 0.64 against a truth
     of 1.0; with it, near 0.96.

Marked slow because each fit solves a small DGP and runs the semi-gradient + logit
first stage. Reproduce the full Monte Carlo with
``validation/estimators/tdccp/bus_engine_mc.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIR = ROOT / "validation" / "estimators" / "tdccp"
for _p in (str(ROOT), str(ROOT / "src"), str(DRIVER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bus_engine_mc as be  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402


def _fit_thetas(
    ccp_use_encoder: bool, seed: int = 20250, n_buses: int = 600, n_periods: int = 30
) -> dict[str, float]:
    dgp = be.build_dgp()
    operator = SoftBellmanOperator(dgp["problem"], dgp["transitions"])
    true_reward = dgp["utility"].compute(jnp.asarray(be.THETA_TRUE))
    truth = value_iteration(operator, true_reward, tol=1e-12, max_iter=20_000)
    init = be.stationary_initial_distribution(dgp["problem"], dgp["transitions"], truth.policy)
    panel = simulate_panel_from_policy(
        dgp["problem"],
        dgp["transitions"],
        truth.policy,
        init,
        n_individuals=n_buses,
        n_periods=n_periods,
        seed=seed,
    )
    cfg = TDCCPConfig(
        method="semigradient",
        basis_type="encoded",
        basis_dim=3,
        basis_ridge=1e-5,
        ccp_method="logit",
        ccp_poly_degree=3,
        ccp_use_encoder=ccp_use_encoder,
        cross_fitting=False,
        robust_se=False,
        compute_se=False,
        n_policy_iterations=1,
        outer_max_iter=500,
        outer_tol=1e-8,
        verbose=False,
    )
    est = TDCCPEstimator(config=cfg, seed=seed)
    summary = est.estimate(
        panel=panel,
        utility=dgp["utility"],
        problem=dgp["problem"],
        transitions=dgp["transitions"],
    )
    return dict(zip(be.PARAM_NAMES, np.asarray(summary.parameters, dtype=float)))


@pytest.mark.slow
def test_encoder_ccp_recovers_bus_engine_thetas():
    """Encoder CCP recovers (theta0, theta1, theta2) = (2, -0.15, 1)."""
    est = _fit_thetas(ccp_use_encoder=True)
    assert 1.75 <= est["theta0_intercept"] <= 2.30, est
    assert -0.19 <= est["theta1_mileage"] <= -0.11, est
    assert est["theta2_type"] >= 0.85, est  # near true 1.0


@pytest.mark.slow
def test_gbm_avi_recovers_bus_engine_thetas():
    """The any-ML AVI (gradient boosting) recovers the bus-engine thetas.

    ``method="neural"`` with ``avi_functional_class="gbm"`` runs approximate value
    iteration with a sklearn HistGradientBoostingRegressor in place of a neural
    network (the paper's any-ML AVI, eq 3.2). The plug-in AVI carries some
    small-sample bias, so the tolerance is looser than the semigradient's.
    """
    dgp = be.build_dgp()
    operator = SoftBellmanOperator(dgp["problem"], dgp["transitions"])
    true_reward = dgp["utility"].compute(jnp.asarray(be.THETA_TRUE))
    truth = value_iteration(operator, true_reward, tol=1e-12, max_iter=20_000)
    init = be.stationary_initial_distribution(dgp["problem"], dgp["transitions"], truth.policy)
    panel = simulate_panel_from_policy(
        dgp["problem"],
        dgp["transitions"],
        truth.policy,
        init,
        n_individuals=600,
        n_periods=30,
        seed=20250,
    )
    cfg = TDCCPConfig(
        method="neural",
        avi_functional_class="gbm",
        ccp_method="logit",
        ccp_poly_degree=3,
        ccp_use_encoder=True,
        avi_iterations=20,
        cross_fitting=False,
        robust_se=False,
        compute_se=False,
        verbose=False,
    )
    est = TDCCPEstimator(config=cfg, seed=20250)
    summary = est.estimate(
        panel=panel,
        utility=dgp["utility"],
        problem=dgp["problem"],
        transitions=dgp["transitions"],
    )
    th = dict(zip(be.PARAM_NAMES, np.asarray(summary.parameters, dtype=float)))
    assert 1.6 <= th["theta0_intercept"] <= 2.4, th
    assert -0.20 <= th["theta1_mileage"] <= -0.10, th
    assert 0.8 <= th["theta2_type"] <= 1.25, th


@pytest.mark.slow
def test_ccp_design_robust_for_type_coefficient():
    """theta2 recovers under BOTH the scalar-index and the encoder logit CCP.

    Historically the scalar-index CCP appeared to bias theta2 low (~0.64) while the
    encoder CCP "fixed" it (~0.96). That gap was a symptom of a transition-tuple
    misalignment in ``_extract_transitions`` (a_{t+1} misaligned with (s_t, s_{t+1})
    across trajectory boundaries), not the CCP design. With the alignment fixed,
    theta2 recovers regardless of the CCP encoder, so the encoder is no longer
    load-bearing for theta2.
    """
    on = _fit_thetas(ccp_use_encoder=True)["theta2_type"]
    off = _fit_thetas(ccp_use_encoder=False)["theta2_type"]
    assert on >= 0.85, f"encoder CCP should recover theta2, got {on}"
    assert off >= 0.85, f"scalar-index CCP should also recover theta2 post-fix, got {off}"
