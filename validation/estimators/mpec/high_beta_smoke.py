#!/usr/bin/env python3
"""Generate the local MPEC high-beta smoke artifact.

This is deliberately tiny. It exercises the recommended MPEC ``solver="sqp"``
path at beta=0.9999 without standing in for the Tier 4 high-gamma release run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "mpec_high_beta_smoke.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.environments.rust_bus import RustBusEnvironment  # noqa: E402
from econirl.estimation.mpec import MPECConfig, MPECEstimator  # noqa: E402
from econirl.preferences.linear import LinearUtility  # noqa: E402
from econirl.simulation import simulate_panel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_finite_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


def run_smoke() -> dict[str, Any]:
    env_config = {
        "operating_cost": 0.01,
        "replacement_cost": 1.0,
        "num_mileage_bins": 4,
        "discount_factor": 0.9999,
        "seed": 7,
    }
    simulation_config = {
        "n_individuals": 6,
        "n_periods": 8,
        "seed": 8,
    }
    solver_config = {
        "solver": "sqp",
        "outer_max_iter": 80,
        "constraint_tol": 1e-5,
        "tol": 1e-6,
        "compute_hessian": False,
    }

    env = RustBusEnvironment(**env_config)
    panel = simulate_panel(env, **simulation_config)
    utility = LinearUtility.from_environment(env)
    estimator = MPECEstimator(
        config=MPECConfig(
            solver=solver_config["solver"],
            outer_max_iter=solver_config["outer_max_iter"],
            constraint_tol=solver_config["constraint_tol"],
            tol=solver_config["tol"],
        ),
        compute_hessian=solver_config["compute_hessian"],
        verbose=False,
    )

    wall_start = time.time()
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    wall_time_seconds = time.time() - wall_start

    final_constraint_violation = result.metadata.get("final_constraint_violation")
    passed = (
        bool(result.converged)
        and result.metadata.get("method") == "slsqp"
        and final_constraint_violation is not None
        and float(final_constraint_violation) < 1e-5
        and int(result.num_iterations) <= solver_config["outer_max_iter"]
        and 0 < float(result.estimation_time) < 15.0
        and 0 < wall_time_seconds < 15.0
    )

    return {
        "artifact_name": "mpec_high_beta_smoke",
        "artifact_type": "local_smoke_guard",
        "estimator": "MPEC",
        "generated_by": "validation/estimators/mpec/high_beta_smoke.py",
        "release_status": "local_smoke_only_not_tier4_release_evidence",
        "does_not_replace": "tier4_high_gamma_mpec",
        "purpose": (
            "Exercise the recommended scipy SLSQP MPEC path at beta=0.9999 "
            "without claiming full high-beta release evidence."
        ),
        "environment": "RustBusEnvironment",
        "environment_config": env_config,
        "simulation_config": simulation_config,
        "solver_config": solver_config,
        "thresholds": {
            "discount_factor_min": 0.999,
            "final_constraint_violation_max": 1e-5,
            "num_iterations_max": solver_config["outer_max_iter"],
            "estimation_time_max_seconds": 15.0,
            "wall_time_max_seconds": 15.0,
        },
        "result": {
            "passed": passed,
            "converged": bool(result.converged),
            "method": result.metadata.get("method"),
            "final_constraint_violation": final_constraint_violation,
            "num_iterations": int(result.num_iterations),
            "estimation_time": float(result.estimation_time),
            "wall_time_seconds": float(wall_time_seconds),
            "num_observations": int(panel.num_observations),
            "parameters": np.asarray(result.parameters, dtype=float).tolist(),
            "message": result.convergence_message,
        },
    }


def _finite_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _finite_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
