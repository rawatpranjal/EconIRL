#!/usr/bin/env python3
"""Reproduce Rust (1987) Table IX, group 4, with converged CCP/NPL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_RAW_PATH = ROOT / "downloads" / "nfxp_unzip" / "nfxp" / "dat" / "a530875.asc"
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "ccp_rust_table_ix.json"
EXPECTED_RAW_SHA256 = "5e85a1c33c11632effbec3ffb213c8e4c92501a49dfe388ad28a203f8c732387"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.environments.rust_bus import RustBusEnvironment  # noqa: E402
from econirl.estimation.ccp import CCPEstimator  # noqa: E402
from econirl.preferences.linear import LinearUtility  # noqa: E402
from econirl.replication.rust1987.table_ix import (  # noqa: E402
    PAPER_TABLE_IX_GROUP4,
    load_stordat_group4_panel,
    table_ix_group4,
)
from econirl.replication.rust1987.tables import _df_to_panel  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _gate(actual: float, target: float, tolerance: float) -> dict[str, Any]:
    actual = float(actual)
    target = float(target)
    tolerance = float(tolerance)
    error = float(abs(actual - target))
    return {
        "actual": actual,
        "target": target,
        "absolute_error": error,
        "tolerance": tolerance,
        "passed": bool(error <= tolerance),
    }


def run_profile(raw_path: Path) -> dict[str, Any]:
    """Run CCP and NFXP on the official STORDAT group-4 sample."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing official NFXP data at {raw_path}. Run `make ccp-table-ix`."
        )
    source_sha256 = _sha256(raw_path)
    if source_sha256 != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"Unexpected STORDAT checksum {source_sha256}; expected {EXPECTED_RAW_SHA256}."
        )

    dataframe, metadata = load_stordat_group4_panel(raw_path)
    panel = _df_to_panel(dataframe)
    probabilities = (
        float(metadata["transition_probabilities"][0]),
        float(metadata["transition_probabilities"][1]),
        float(metadata["transition_probabilities"][2]),
    )
    increments = dataframe.sort_values(["bus_id", "period"])["monthly_mileage_increment"].to_numpy(
        dtype=int
    )
    environment = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=int(metadata["n_states"]),
        mileage_transition_probs=probabilities,
        discount_factor=0.9999,
    )
    utility = LinearUtility.from_environment(environment)
    result = CCPEstimator(
        mode="npl",
        ccp_smoothing=0.0,
        convergence_tol=1e-12,
        outer_tol=1e-10,
        outer_max_iter=2000,
        se_method="full_likelihood_bhhh",
        compute_hessian=True,
        verbose=False,
    ).estimate(
        panel,
        utility,
        environment.problem_spec,
        environment.transition_matrices,
        transition_probabilities=probabilities,
        transition_increments=increments,
    )
    nfxp = table_ix_group4(
        raw_path,
        betas=(0.9999,),
        compute_hessian=True,
    ).iloc[0]

    standard_errors = np.asarray(result.standard_errors, dtype=np.float64)
    se_details = result.metadata["se_details"]
    joint_standard_errors = np.asarray(
        se_details["joint_standard_errors"],
        dtype=np.float64,
    )
    joint_covariance = np.asarray(
        se_details["joint_variance_covariance"],
        dtype=np.float64,
    )
    theta_1 = float(result.parameters[0]) * 1000.0
    theta_1_se = standard_errors[0] * 1000.0
    replacement_cost = float(result.parameters[1])
    replacement_cost_se = standard_errors[1]
    if result.log_likelihood is None:
        raise ValueError("CCP profile did not return a log-likelihood")
    choice_log_likelihood = float(result.log_likelihood)
    transition_log_likelihood = float(metadata["transition_log_likelihood"])
    full_log_likelihood = choice_log_likelihood + transition_log_likelihood
    paper = PAPER_TABLE_IX_GROUP4[0.9999]

    gates = {
        "converged": {
            "actual": bool(result.converged),
            "target": True,
            "passed": bool(result.converged),
        },
        "termination_reason": {
            "actual": result.metadata["termination_reason"],
            "target": "fixed_point_converged",
            "passed": result.metadata["termination_reason"] == "fixed_point_converged",
        },
        "npl_parameter_residual": {
            "actual": float(result.metadata["npl_parameter_residual"]),
            "maximum": 1e-12,
            "passed": float(result.metadata["npl_parameter_residual"]) <= 1e-12,
        },
        "npl_policy_residual": {
            "actual": float(result.metadata["npl_policy_residual"]),
            "maximum": 1e-12,
            "passed": float(result.metadata["npl_policy_residual"]) <= 1e-12,
        },
        "bellman_policy_residual": {
            "actual": float(result.metadata["full_likelihood_bhhh"]["bellman_policy_residual"]),
            "maximum": 1e-8,
            "passed": float(result.metadata["full_likelihood_bhhh"]["bellman_policy_residual"])
            <= 1e-8,
        },
        "theta_1_paper": _gate(theta_1, paper["theta_1_paper_units"], 5e-4),
        "theta_1_se_paper": _gate(
            theta_1_se,
            paper["theta_1_se_paper_units"],
            5e-4,
        ),
        "replacement_cost_paper": _gate(replacement_cost, paper["RC"], 5e-4),
        "replacement_cost_se_paper": _gate(
            replacement_cost_se,
            paper["RC_se"],
            5e-4,
        ),
        "p0_paper": _gate(probabilities[0], paper["p0"], 5e-5),
        "p0_se_paper": _gate(joint_standard_errors[2], paper["p0_se"], 5e-5),
        "p1_paper": _gate(probabilities[1], paper["p1"], 5e-5),
        "p1_se_paper": _gate(joint_standard_errors[3], paper["p1_se"], 5e-5),
        "choice_log_likelihood_paper": _gate(
            choice_log_likelihood,
            paper["choice_log_likelihood"],
            5e-3,
        ),
        "full_log_likelihood_paper": _gate(
            full_log_likelihood,
            paper["full_log_likelihood"],
            5e-3,
        ),
        "theta_1_nfxp": _gate(theta_1, float(nfxp["theta_1_paper_units"]), 3e-4),
        "theta_1_se_nfxp": _gate(
            theta_1_se,
            float(nfxp["theta_1_se_paper_units"]),
            3e-4,
        ),
        "replacement_cost_nfxp": _gate(
            replacement_cost,
            float(nfxp["RC"]),
            3e-4,
        ),
        "replacement_cost_se_nfxp": _gate(
            replacement_cost_se,
            float(nfxp["RC_se"]),
            3e-4,
        ),
        "finite_positive_joint_covariance": {
            "actual": bool(
                np.all(np.isfinite(joint_covariance)) and np.all(np.diag(joint_covariance) > 0.0)
            ),
            "target": True,
            "passed": bool(
                np.all(np.isfinite(joint_covariance)) and np.all(np.diag(joint_covariance) > 0.0)
            ),
        },
    }
    status = "pass" if all(gate["passed"] for gate in gates.values()) else "fail"

    return {
        "schema_version": 1,
        "status": status,
        "estimator": "CCP converged NPL",
        "paper_target": "Rust (1987), Table IX, group 4, model 11, beta=0.9999",
        "command": "make ccp-table-ix",
        "source": {
            "path": str(DEFAULT_RAW_PATH.relative_to(ROOT)),
            "archive_url": metadata["source_archive_url"],
            "sha256": source_sha256,
            "sample_observations": int(metadata["sample_observations"]),
            "transition_counts": metadata["transition_counts"],
        },
        "environment": {
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                package: _version(package)
                for package in ("econirl", "jax", "numpy", "pandas", "scipy")
            },
        },
        "specification": {
            "discount_factor": 0.9999,
            "num_states": int(metadata["n_states"]),
            "num_actions": 2,
            "transition_orientation": "(n_actions, n_states, n_states)",
            "transition_model": "Rust residual probability",
            "npl_mode": "until joint parameter and policy convergence",
            "ccp_smoothing": 0.0,
            "convergence_tolerance": 1e-12,
            "outer_tolerance": 1e-10,
            "outer_max_iterations": 2000,
            "standard_error_method": "joint full-likelihood BHHH",
        },
        "ccp": {
            "parameters": {
                "theta_1_paper_units": theta_1,
                "replacement_cost": replacement_cost,
            },
            "standard_errors": {
                "theta_1_paper_units": theta_1_se,
                "replacement_cost": replacement_cost_se,
                "transition_p0": float(joint_standard_errors[2]),
                "transition_p1": float(joint_standard_errors[3]),
            },
            "joint_parameter_names": result.metadata["full_likelihood_bhhh"][
                "joint_parameter_names"
            ],
            "joint_covariance": joint_covariance.tolist(),
            "choice_log_likelihood": choice_log_likelihood,
            "transition_log_likelihood": transition_log_likelihood,
            "full_log_likelihood": full_log_likelihood,
            "num_policy_iterations": int(result.num_iterations),
            "npl_parameter_residual": float(result.metadata["npl_parameter_residual"]),
            "npl_policy_residual": float(result.metadata["npl_policy_residual"]),
            "bellman_policy_residual": float(
                result.metadata["full_likelihood_bhhh"]["bellman_policy_residual"]
            ),
        },
        "nfxp": {
            "theta_1_paper_units": float(nfxp["theta_1_paper_units"]),
            "theta_1_se_paper_units": float(nfxp["theta_1_se_paper_units"]),
            "replacement_cost": float(nfxp["RC"]),
            "replacement_cost_se": float(nfxp["RC_se"]),
            "choice_log_likelihood": float(nfxp["choice_log_likelihood"]),
        },
        "paper": paper,
        "gates": gates,
    }


def verify_receipt(output_path: Path) -> None:
    """Verify that an existing receipt is strict JSON with all gates passing."""
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass":
        raise RuntimeError(f"{output_path} does not record status=pass")
    failed = [
        name for name, gate in receipt.get("gates", {}).items() if not gate.get("passed", False)
    ]
    if failed:
        raise RuntimeError(f"{output_path} has failed gates: {failed}")
    if receipt["source"]["sha256"] != EXPECTED_RAW_SHA256:
        raise RuntimeError(f"{output_path} records the wrong source checksum")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify_receipt(args.output)
        print(f"verified {args.output}")
        return 0

    receipt = run_profile(args.raw_path.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
