#!/usr/bin/env python3
"""Subprocess entry point: JSON config file → run_single() → JSON result to stdout."""

import json
import math
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on sys.path so 'autolab' package is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run(config_path: str) -> dict:
    """Load config, run one benchmark experiment, return result dict."""
    import os
    import resource

    with open(config_path) as f:
        config = json.load(f)

    # --- Resource guardrails (before heavy imports) ---
    if config.get("single_thread", True):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    max_mem_gb = config.get("max_memory_gb", 4)
    soft = max_mem_gb * 1024**3
    try:
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
    except (ValueError, OSError):
        pass  # Some platforms don't support RLIMIT_AS

    # Lazy imports — keep subprocess startup visible in timing
    import torch
    torch.set_num_threads(1)
    from econirl.estimation import (
        AIRLEstimator,
        CCPEstimator,
        GAILEstimator,
        GCLEstimator,
        GLADIUSEstimator,
        MaxEntIRLEstimator,
        MaxMarginPlanningEstimator,
        MCEIRLEstimator,
        NFXPEstimator,
        TDCCPEstimator,
        TDCCPConfig,
    )
    from econirl.evaluation.benchmark import BenchmarkDGP, EstimatorSpec, run_single

    # Estimator registry: name → (class, can_recover_params)
    ESTIMATORS = {
        "NFXP": (NFXPEstimator, True),
        "CCP": (CCPEstimator, True),
        "MCE IRL": (MCEIRLEstimator, True),
        "MaxEnt IRL": (MaxEntIRLEstimator, False),
        "Max Margin": (MaxMarginPlanningEstimator, True),
        "TD-CCP": (TDCCPEstimator, True),
        "GLADIUS": (GLADIUSEstimator, True),
        "GAIL": (GAILEstimator, False),
        "AIRL": (AIRLEstimator, False),
        "GCL": (GCLEstimator, False),
    }

    estimator_name = config["estimator"]
    if estimator_name not in ESTIMATORS:
        return {
            "experiment_id": config.get("experiment_id", "unknown"),
            "status": "error",
            "error": f"Unknown estimator: {estimator_name}. Valid: {list(ESTIMATORS.keys())}",
        }

    cls, can_recover = ESTIMATORS[estimator_name]

    # Build DGP
    dgp_kwargs = config.get("dgp", {})
    dgp = BenchmarkDGP(**dgp_kwargs)

    # Build estimator kwargs, handling TDCCPConfig special case
    est_kwargs = dict(config.get("estimator_kwargs", {}))
    if estimator_name == "TD-CCP" and "config" in est_kwargs:
        est_kwargs["config"] = TDCCPConfig(**est_kwargs["config"])

    spec = EstimatorSpec(
        estimator_class=cls,
        kwargs=est_kwargs,
        name=estimator_name,
        can_recover_params=can_recover,
    )

    t0 = time.time()
    result = run_single(
        dgp=dgp,
        spec=spec,
        n_agents=config.get("n_agents", 200),
        n_periods=config.get("n_periods", 100),
        seed=config.get("seed", 42),
    )
    wall_time = time.time() - t0

    # Convert BenchmarkResult to dict
    out = {
        "experiment_id": config.get("experiment_id", "unknown"),
        "status": "success",
        "estimator": result.estimator,
        "n_states": result.n_states,
        "n_agents": result.n_agents,
        "seed": result.seed,
        "param_rmse": _sanitize(result.param_rmse),
        "policy_rmse": _sanitize(result.policy_rmse),
        "pct_optimal": _sanitize(result.pct_optimal),
        "pct_optimal_transfer": _sanitize(result.pct_optimal_transfer),
        "time_seconds": round(wall_time, 2),
        "converged": result.converged,
        "estimates": {k: _sanitize(v) for k, v in result.estimates.items()},
        "true_params": {k: _sanitize(v) for k, v in result.true_params.items()},
    }

    # --- Richer output: action agreement ---
    learned_pol = getattr(result, "learned_policy", None)
    true_pol = getattr(result, "true_policy", None)
    if learned_pol is not None and true_pol is not None:
        import numpy as np
        learned = np.asarray(learned_pol)
        true = np.asarray(true_pol)
        if learned.shape == true.shape and learned.ndim == 2:
            agreement = (learned.argmax(axis=1) == true.argmax(axis=1)).mean()
            out["action_agreement"] = round(float(agreement), 4)

    # --- Richer output: per-parameter relative errors ---
    if result.estimates and result.true_params:
        rel_errors = {}
        for k, est_v in result.estimates.items():
            true_v = result.true_params.get(k)
            if est_v is not None and true_v is not None and true_v != 0:
                rel_errors[k] = round(abs(est_v - true_v) / abs(true_v), 4)
        if rel_errors:
            out["param_relative_errors"] = rel_errors

    # --- Validation against ground truth ---
    from autolab.validation import evaluate_against_ground_truth, classify_difficulty

    difficulty = classify_difficulty(config.get("dgp", {}))
    verdict = evaluate_against_ground_truth(out, estimator_name, difficulty)
    out["difficulty"] = difficulty
    out["ground_truth"] = verdict["criteria"]
    out["all_criteria_passed"] = verdict["all_passed"]

    return out


def _sanitize(v):
    """Convert NaN/Inf to None for JSON serialization."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"status": "error", "error": "Usage: run_experiment.py <config.json>"}))
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        result = run(config_path)
    except Exception as e:
        result = {
            "experiment_id": "unknown",
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
