"""Tests for the honest quick all-estimator script's reporting machinery.

These lock the anti-lie properties without running estimators (fast): the
printed table is a pure function of the raw records, crashes are surfaced rather
than dropped, and parameter metrics appear only for the structural family.
"""

from __future__ import annotations

import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "quick_all_estimators", os.path.join(_ROOT, "scripts", "quick_all_estimators.py")
)
qae = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qae)


def _fake_data() -> dict:
    return {
        "meta": {
            "date": "2026-06-11", "package_version": "test", "n_replications": 1,
            "mdp": {"num_states": 8, "num_actions": 2, "num_features": 2,
                    "branching": 3, "discount_factor": 0.9},
            "n_individuals": 300, "n_periods": 50, "true_theta": [1.0, 2.0],
            "excluded": [{"name": "GAIL", "reason": "slow"}],
        },
        "records": [
            {"estimator": "NFXP", "family": "structural", "rep": 0,
             "params": [1.1, 2.1], "standard_errors": [0.1, 0.1], "policy_tv": 0.01,
             "value_rmse": 0.0, "runtime": 1.0, "converged": True, "error": None},
            {"estimator": "BC", "family": "behavioral", "rep": 0,
             "params": None, "standard_errors": None, "policy_tv": 0.02,
             "value_rmse": None, "runtime": 0.1, "converged": True, "error": None},
            {"estimator": "BROKEN", "family": "behavioral", "rep": 0,
             "params": None, "standard_errors": None, "policy_tv": None,
             "value_rmse": None, "runtime": 0.0, "converged": None,
             "error": "ValueError: boom"},
        ],
    }


def test_param_rmse_recomputed_from_records():
    # sqrt(mean((1.1-1)^2, (2.1-2)^2)) = sqrt(0.01) = 0.1.
    out = qae.render(_fake_data())
    nfxp_line = next(line for line in out.splitlines() if line.startswith("NFXP"))
    assert "0.1000" in nfxp_line
    assert "1/1" in nfxp_line  # ran once


def test_behavioral_family_gets_no_param_rmse():
    out = qae.render(_fake_data())
    bc_line = next(line for line in out.splitlines() if line.startswith("BC"))
    # ParamRMSE column is n/a for behavioral estimators, but PolicyTV is present.
    assert "n/a" in bc_line
    assert "0.0200" in bc_line


def test_crash_is_surfaced_not_dropped():
    out = qae.render(_fake_data())
    broken_line = next(line for line in out.splitlines() if line.startswith("BROKEN"))
    assert "CRASHED" in broken_line
    assert "ValueError: boom" in broken_line
    assert "0/1" in broken_line  # zero successful runs, recorded honestly


def test_excluded_estimators_listed():
    out = qae.render(_fake_data())
    assert "Excluded from this quick run" in out
    assert "GAIL" in out
