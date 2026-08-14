#!/usr/bin/env python3
"""Generate TD-CCP primer results from the known-truth DGP harness.

The TD-CCP primer now uses the shared synthetic DGP instead of a separate
continuous-state demonstration. The low-dimensional cell is gate enforced and
is the current validated path. The high-dimensional cell is run as a diagnostic
stress test because it still fails the same structural recovery gates.

Usage:
    PYTHONPATH=src:. python validation/estimators/tdccp/run.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "tdccp.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_tdccp_primer_known_truth")
DEFAULT_CELL_IDS = ("canonical_low_action", "canonical_high_action")
ESTIMATOR = "TD-CCP"
HARD_CASE_ID = "shapeshifter_encoded_state_locally_robust"
RAW_NEURAL_DIAGNOSTIC_ID = "shapeshifter_neural_neural"
PRIMARY_CELL_ID = HARD_CASE_ID
ENFORCED_SUPPORTING_CELL_IDS = {"canonical_low_action"}
PAPER_TARGET = (
    "Temporal-difference CCP recovery of finite-dimensional structural "
    "reward parameters with Algorithm 2 locally robust standard errors"
)
RELEASE_STATUS = "Certified with caveat"
CELL_LABELS = {
    "canonical_low_action": "Low-dimensional",
    "canonical_high_action": "High-dimensional",
}
CELL_ROLES = {
    "canonical_low_action": "validated",
    "canonical_high_action": "diagnostic",
}
HARD_CASE_SEED = 42
HARD_CASE_N_INDIVIDUALS = 2_000
HARD_CASE_N_PERIODS = 60
MC_REPLICATIONS = 25
MC_N_INDIVIDUALS = 300
MC_N_PERIODS = 35
MC_SEED_START = 10_101
PAPER_HARD_THETA_SCALE = 0.8

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validation_display import (  # noqa: E402
    validation_context,
    validation_display_name,
    validation_role,
)

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.environments.shapeshifter import (  # noqa: E402
    ShapeshifterConfig,
    ShapeshifterEnvironment,
)
from econirl.estimation.td_ccp import (  # noqa: E402
    TDCCPConfig,
    TDCCPEstimator,
    make_state_action_tabular_utility,
)
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402
from validation.known_truth import (  # noqa: E402
    RecoveryGate,
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    stable_hash,
    to_jsonable,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help="Known-truth cell to run. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-hard-case",
        action="store_true",
        help="Skip the shapeshifter neural/neural TD-CCP hard-case run.",
    )
    parser.add_argument(
        "--hard-case-only",
        action="store_true",
        help="Run only the shapeshifter neural/neural TD-CCP hard case.",
    )
    parser.add_argument(
        "--enforce-hard-gates",
        action="store_true",
        help="Raise a nonzero error if the hard-case gates fail.",
    )
    parser.add_argument("--hard-n-individuals", type=int, default=HARD_CASE_N_INDIVIDUALS)
    parser.add_argument("--hard-n-periods", type=int, default=HARD_CASE_N_PERIODS)
    parser.add_argument("--hard-seed", type=int, default=HARD_CASE_SEED)
    parser.add_argument(
        "--skip-monte-carlo",
        action="store_true",
        help="Skip the repeated-seed TD-CCP coverage check.",
    )
    parser.add_argument("--mc-replications", type=int, default=MC_REPLICATIONS)
    parser.add_argument("--mc-n-individuals", type=int, default=MC_N_INDIVIDUALS)
    parser.add_argument("--mc-n-periods", type=int, default=MC_N_PERIODS)
    parser.add_argument("--mc-seed-start", type=int, default=MC_SEED_START)
    args = parser.parse_args()

    cell_ids = args.cell_id if args.cell_id is not None else list(DEFAULT_CELL_IDS)
    print("TD-CCP primer: running known-truth validation")
    if args.hard_case_only:
        cell_ids = []
    print(f"  cells: {', '.join(cell_ids) if cell_ids else '(none)'}")
    print(f"  estimator: {ESTIMATOR}")
    print(f"  output_dir: {args.output_dir}")

    records = [run_validation_cell(cell_id, args) for cell_id in cell_ids]
    hard_record = None
    raw_neural_record = None
    if not args.skip_hard_case:
        hard_record = run_hard_case(args)
        raw_neural_record = run_raw_neural_diagnostic(args)
    monte_carlo_record = None
    if not args.skip_hard_case and not args.skip_monte_carlo:
        monte_carlo_record = run_monte_carlo_coverage(args)

    JSON_OUT.write_text(
        json.dumps(
            to_jsonable(
                compact_payload(
                    records,
                    hard_record,
                    raw_neural_record,
                    monte_carlo_record,
                )
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    total_failed = 0
    for record in records:
        payload = record["payload"]
        failed = [gate for gate in payload["gates"] if not gate.passed]
        total_failed += len(failed)
        print(f"  result ({payload['cell'].cell_id}): {record['run_dir'] / 'result.json'}")
        print(
            f"  hard gates ({payload['cell'].cell_id}): "
            f"{len(payload['gates']) - len(failed)} pass, {len(failed)} fail"
        )
    if hard_record is not None:
        hard_payload = hard_record["payload"]
        failed = [gate for gate in hard_payload["gates"] if not gate.passed]
        total_failed += len(failed)
        print(f"  result ({hard_payload['case_id']}): {hard_record['run_dir'] / 'result.json'}")
        print(
            f"  hard-case gates ({hard_payload['case_id']}): "
            f"{len(hard_payload['gates']) - len(failed)} pass, {len(failed)} fail"
        )
        if failed:
            print("  hard-case status: failure artifact written; no success claim emitted")
    if raw_neural_record is not None:
        raw_payload = raw_neural_record["payload"]
        failed = [gate for gate in raw_payload["gates"] if not gate.passed]
        result_path = raw_neural_record["run_dir"] / "result.json"
        print(f"  diagnostic ({raw_payload['case_id']}): {result_path}")
        print(
            f"  raw-neural diagnostic gates ({raw_payload['case_id']}): "
            f"{len(raw_payload['gates']) - len(failed)} pass, {len(failed)} fail"
        )
        if failed:
            print("  raw-neural diagnostic status: failure artifact retained")
    if monte_carlo_record is not None:
        mc_payload = monte_carlo_record["payload"]
        print(
            "  monte carlo: "
            f"{mc_payload['successful_replications']}/{mc_payload['n_replications']} "
            "replications completed, "
            f"coverage={mc_payload['coverage_95_overall']:.3f}"
        )
    print(f"  wrote: {JSON_OUT}")
    print(f"  failed gates total: {total_failed}")

    if (
        args.enforce_hard_gates
        and hard_record is not None
        and any(not gate.passed for gate in hard_record["payload"]["gates"])
    ):
        failed_names = [gate.name for gate in hard_record["payload"]["gates"] if not gate.passed]
        raise RuntimeError("TD-CCP hard-case gates failed: " + ", ".join(failed_names))


def run_validation_cell(cell_id: str, args: argparse.Namespace) -> dict[str, Any]:
    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = replace(
        cell.simulation_config,
        show_progress=args.show_progress,
    )
    panel = simulate_known_truth_panel(dgp, simulation_config)

    main_run = run_estimator(
        ESTIMATOR,
        dgp,
        panel,
        smoke=False,
        verbose=args.verbose,
        enforce_gates=False,
    )
    failed = [gate for gate in main_run.gates if not gate.passed]
    if cell.cell_id in ENFORCED_SUPPORTING_CELL_IDS and failed:
        details = "; ".join(gate.name for gate in failed)
        raise RuntimeError(f"TD-CCP supporting validation cell failed gates: {details}")

    payload = {
        "cell": cell,
        "simulation": simulation_config,
        "estimator": ESTIMATOR,
        "diagnostics": main_run.diagnostics,
        "compatibility": main_run.compatibility,
        "summary": main_run.summary,
        "metrics": main_run.metrics,
        "gates": main_run.gates,
    }
    run_hash = stable_hash(
        {
            "cell": cell.dgp_config.to_dict(),
            "simulation": simulation_config,
            "estimator": ESTIMATOR,
            "primer": "tdccp",
            "enforce_gates": False,
        }
    )
    run_dir = args.output_dir / f"{cell_id}_tdccp_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "dgp": dgp, "run_dir": run_dir}


def build_paper_hard_case_dgp(seed: int = HARD_CASE_SEED) -> dict[str, Any]:
    """Build an encoded-state finite-theta TD-CCP hard case."""
    env_config = ShapeshifterConfig(
        num_states=9,
        num_actions=3,
        num_features=6,
        reward_type="linear",
        feature_type="linear",
        action_dependent=True,
        stochastic_transitions=True,
        stochastic_rewards=False,
        num_periods=None,
        discount_factor=0.95,
        scale_parameter=1.0,
        state_dim=2,
        transition_branching=4,
        seed=seed,
    )
    env = ShapeshifterEnvironment(env_config)
    coords = np.asarray(
        env.encode_states(jnp.arange(env.num_states, dtype=jnp.int32)),
        dtype=np.float64,
    )
    if coords.shape[1] < 2:
        raise ValueError("encoded TD-CCP hard case requires state_dim >= 2")

    x0 = coords[:, 0]
    x1 = coords[:, 1]
    state_basis = np.column_stack(
        [
            np.ones(env.num_states, dtype=np.float64),
            x0,
            x1,
        ]
    )
    basis_names = ["intercept", "x0", "x1"]

    num_actions = env.num_actions
    rank = state_basis.shape[1]
    feature_matrix = np.zeros(
        (env.num_states, num_actions, rank * (num_actions - 1)),
        dtype=np.float64,
    )
    parameter_names: list[str] = []
    for action in range(1, num_actions):
        start = (action - 1) * rank
        stop = action * rank
        feature_matrix[:, action, start:stop] = state_basis
        parameter_names.extend([f"action_{action}_{name}" for name in basis_names])

    true_params = (
        np.asarray(
            jax.random.normal(
                jax.random.PRNGKey(seed + 4_200),
                shape=(feature_matrix.shape[-1],),
                dtype=jnp.float64,
            )
        )
        * PAPER_HARD_THETA_SCALE
    )
    utility = ActionDependentReward(jnp.asarray(feature_matrix), parameter_names)
    true_reward = utility.compute(jnp.asarray(true_params))

    return {
        "env": env,
        "utility": utility,
        "true_params": jnp.asarray(true_params),
        "true_reward": true_reward,
        "basis_metadata": {
            "basis_source": "encoded state polynomial features",
            "action_normalization": "action 0 reward features fixed to zero",
            "state_basis": basis_names,
            "feature_rank": rank,
            "state_dim": env.state_dim,
            "total_states": env.num_states,
            "theta_scale": PAPER_HARD_THETA_SCALE,
        },
    }


def paper_hard_case_estimator_config(*, verbose: bool = False) -> TDCCPConfig:
    """Estimator settings for the certified encoded-state Algorithm 2 case."""
    return TDCCPConfig(
        method="semigradient",
        basis_type="encoded",
        basis_dim=2,
        basis_ridge=1e-7,
        ccp_method="logit",
        ccp_poly_degree=2,
        cross_fitting=True,
        robust_se=True,
        compute_se=True,
        n_policy_iterations=1,
        outer_max_iter=500,
        outer_tol=1e-7,
        verbose=verbose,
    )


def run_hard_case(args: argparse.Namespace) -> dict[str, Any]:
    """Run the paper-faithful hard case with neural features and linear reward."""
    dgp = build_paper_hard_case_dgp(args.hard_seed)
    env: ShapeshifterEnvironment = dgp["env"]
    utility: ActionDependentReward = dgp["utility"]
    true_params = jnp.asarray(dgp["true_params"])
    true_reward = jnp.asarray(dgp["true_reward"])

    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    truth = value_iteration(
        operator,
        true_reward,
        tol=1e-10,
        max_iter=10_000,
    )
    initial_distribution = jnp.asarray(env._get_initial_state_distribution())
    panel = simulate_panel_from_policy(
        env.problem_spec,
        env.transition_matrices,
        truth.policy,
        initial_distribution,
        n_individuals=args.hard_n_individuals,
        n_periods=args.hard_n_periods,
        seed=args.hard_seed,
    )
    estimator_config = paper_hard_case_estimator_config(verbose=args.verbose)
    estimator = TDCCPEstimator(config=estimator_config, seed=args.hard_seed)
    summary = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    metrics = evaluate_paper_hard_case_summary(
        env,
        utility,
        true_params,
        true_reward,
        summary,
        truth=truth,
    )
    gates = tdccp_paper_hard_case_gates(summary, metrics)
    payload = {
        "case_id": HARD_CASE_ID,
        "estimator": ESTIMATOR,
        "environment_config": env.config,
        "utility_config": dgp["basis_metadata"],
        "simulation": {
            "n_individuals": args.hard_n_individuals,
            "n_periods": args.hard_n_periods,
            "seed": args.hard_seed,
        },
        "estimator_config": {
            "method": estimator_config.method,
            "basis_type": estimator_config.basis_type,
            "basis_dim": estimator_config.basis_dim,
            "basis_ridge": estimator_config.basis_ridge,
            "ccp_method": estimator_config.ccp_method,
            "ccp_poly_degree": estimator_config.ccp_poly_degree,
            "cross_fitting": estimator_config.cross_fitting,
            "robust_se": estimator_config.robust_se,
            "compute_se": estimator_config.compute_se,
            "n_policy_iterations": estimator_config.n_policy_iterations,
            "outer_max_iter": estimator_config.outer_max_iter,
            "outer_tol": estimator_config.outer_tol,
        },
        "truth": {
            "parameter_names": utility.parameter_names,
            "parameters": true_params,
        },
        "summary": summary,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gate.passed for gate in gates),
        "notes": (
            "Paper-faithful hard case: encoded two-dimensional state features, "
            "logit first-stage CCPs, Algorithm 2 cross-fitting, locally robust "
            "standard errors, and action-0 reward normalization."
        ),
    }
    run_hash = stable_hash(
        {
            "case": HARD_CASE_ID,
            "environment": env.config,
            "utility_config": payload["utility_config"],
            "simulation": payload["simulation"],
            "estimator_config": payload["estimator_config"],
        }
    )
    run_dir = args.output_dir / f"{HARD_CASE_ID}_tdccp_{run_hash}"
    write_json(run_dir / "result.json", compact_hard_case_payload(payload, role="certified"))
    return {"payload": payload, "run_dir": run_dir}


def run_monte_carlo_coverage(args: argparse.Namespace) -> dict[str, Any]:
    """Repeated-seed coverage check for the certified TD-CCP hard case."""
    if args.mc_replications <= 0:
        raise ValueError("--mc-replications must be positive")

    dgp = build_paper_hard_case_dgp(args.hard_seed)
    env: ShapeshifterEnvironment = dgp["env"]
    utility: ActionDependentReward = dgp["utility"]
    true_params = np.asarray(dgp["true_params"], dtype=np.float64)
    true_reward = jnp.asarray(dgp["true_reward"])

    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    truth = value_iteration(
        operator,
        true_reward,
        tol=1e-10,
        max_iter=10_000,
    )
    initial_distribution = jnp.asarray(env._get_initial_state_distribution())
    estimator_config = paper_hard_case_estimator_config(verbose=args.verbose)
    seeds = list(range(args.mc_seed_start, args.mc_seed_start + args.mc_replications))

    estimates: list[np.ndarray] = []
    standard_errors: list[np.ndarray] = []
    coverage: list[np.ndarray] = []
    relative_rmse: list[float] = []
    gate_pass_counts: list[int] = []
    gate_fail_counts: list[int] = []
    moment_norms: list[float] = []
    lambda_residual_norms: list[float] = []
    lambda_residual_rms: list[float] = []
    lambda_residual_max_abs: list[float] = []
    estimation_times: list[float] = []
    preliminary_success: list[list[bool]] = []
    preliminary_stationary: list[list[bool]] = []
    preliminary_messages: list[str] = []
    robust_success: list[list[bool]] = []
    robust_stationary: list[list[bool]] = []
    preliminary_projected_gradient_norms: list[float] = []
    robust_projected_gradient_norms: list[float] = []
    covariance_units: list[str] = []
    per_replication: list[dict[str, Any]] = []

    for seed in seeds:
        panel = simulate_panel_from_policy(
            env.problem_spec,
            env.transition_matrices,
            truth.policy,
            initial_distribution,
            n_individuals=args.mc_n_individuals,
            n_periods=args.mc_n_periods,
            seed=seed,
        )
        estimator = TDCCPEstimator(config=estimator_config, seed=seed)
        summary = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )
        metrics = evaluate_paper_hard_case_summary(
            env,
            utility,
            jnp.asarray(true_params),
            true_reward,
            summary,
            truth=truth,
        )
        gates = tdccp_paper_hard_case_gates(summary, metrics)
        ci_low, ci_high = summary.confidence_interval()
        ci_low = np.asarray(ci_low, dtype=np.float64)
        ci_high = np.asarray(ci_high, dtype=np.float64)
        theta_hat = np.asarray(summary.parameters, dtype=np.float64)
        se_hat = np.asarray(summary.standard_errors, dtype=np.float64)
        covered = (ci_low <= true_params) & (true_params <= ci_high)
        paper_inference = summary.metadata.get("paper_inference") or {}
        prelim = [bool(value) for value in paper_inference.get("preliminary_optimizer_success", [])]
        robust = [bool(value) for value in paper_inference.get("robust_optimizer_success", [])]
        prelim_stationary = [
            bool(value) for value in paper_inference.get("preliminary_optimizer_stationary", [])
        ]
        robust_is_stationary = [
            bool(value) for value in paper_inference.get("robust_optimizer_stationary", [])
        ]
        preliminary_success.append(prelim)
        preliminary_stationary.append(prelim_stationary)
        robust_success.append(robust)
        robust_stationary.append(robust_is_stationary)
        preliminary_messages.extend(
            str(message) for message in paper_inference.get("preliminary_optimizer_messages", [])
        )
        covariance_units.append(str(paper_inference.get("covariance_unit", "unknown")))
        preliminary_projected_gradient_norms.append(
            float(
                paper_inference.get(
                    "preliminary_projected_gradient_norm_max",
                    float("nan"),
                )
            )
        )
        robust_projected_gradient_norms.append(
            float(
                paper_inference.get(
                    "robust_projected_gradient_norm_max",
                    float("nan"),
                )
            )
        )

        estimates.append(theta_hat)
        standard_errors.append(se_hat)
        coverage.append(covered.astype(np.float64))
        relative_rmse.append(float(metrics["parameters"]["relative_rmse"]))
        gate_pass_count = sum(gate.passed for gate in gates)
        gate_pass_counts.append(gate_pass_count)
        gate_fail_counts.append(len(gates) - gate_pass_count)
        moment_norm = float(paper_inference.get("moment_norm_max", float("nan")))
        lambda_norm = float(
            paper_inference.get(
                "lambda_fixed_point_residual_norm_max",
                float("nan"),
            )
        )
        lambda_rms = float(
            paper_inference.get(
                "lambda_fixed_point_residual_rms_max",
                float("nan"),
            )
        )
        lambda_max_abs = float(
            paper_inference.get(
                "lambda_fixed_point_residual_max_abs",
                float("nan"),
            )
        )
        moment_norms.append(moment_norm)
        lambda_residual_norms.append(lambda_norm)
        lambda_residual_rms.append(lambda_rms)
        lambda_residual_max_abs.append(lambda_max_abs)
        estimation_times.append(float(summary.estimation_time))
        per_replication.append(
            {
                "seed": seed,
                "parameter_relative_rmse": metrics["parameters"]["relative_rmse"],
                "coverage_95": float(np.mean(covered)),
                "mean_standard_error": float(np.mean(se_hat)),
                "gate_pass_count": gate_pass_count,
                "gate_fail_count": len(gates) - gate_pass_count,
                "zeta_moment_norm": moment_norm,
                "lambda_fixed_point_residual_norm": lambda_norm,
                "lambda_fixed_point_residual_rms": lambda_rms,
                "lambda_fixed_point_residual_max_abs": lambda_max_abs,
                "preliminary_optimizer_success": prelim,
                "preliminary_optimizer_stationary": prelim_stationary,
                "robust_optimizer_success": robust,
                "robust_optimizer_stationary": robust_is_stationary,
                "preliminary_projected_gradient_norm": paper_inference.get(
                    "preliminary_projected_gradient_norm_max"
                ),
                "robust_projected_gradient_norm": paper_inference.get(
                    "robust_projected_gradient_norm_max"
                ),
                "estimation_time": summary.estimation_time,
            }
        )

    theta_array = np.vstack(estimates)
    se_array = np.vstack(standard_errors)
    coverage_array = np.vstack(coverage)
    errors = theta_array - true_params[None, :]
    empirical_sd = (
        np.std(theta_array, axis=0, ddof=1)
        if args.mc_replications > 1
        else np.zeros_like(true_params)
    )
    preliminary_all_success = [bool(values) and all(values) for values in preliminary_success]
    preliminary_all_stationary = [bool(values) and all(values) for values in preliminary_stationary]
    robust_all_success = [bool(values) and all(values) for values in robust_success]
    robust_all_stationary = [bool(values) and all(values) for values in robust_stationary]
    message_counts = {
        message: preliminary_messages.count(message)
        for message in sorted(set(preliminary_messages))
    }

    payload = {
        "case_id": HARD_CASE_ID,
        "estimator": ESTIMATOR,
        "purpose": "repeated_seed_algorithm2_standard_error_coverage",
        "n_replications": args.mc_replications,
        "successful_replications": len(per_replication),
        "simulation": {
            "n_individuals": args.mc_n_individuals,
            "n_periods": args.mc_n_periods,
            "seeds": seeds,
        },
        "estimator_config": {
            "method": estimator_config.method,
            "basis_type": estimator_config.basis_type,
            "basis_dim": estimator_config.basis_dim,
            "basis_ridge": estimator_config.basis_ridge,
            "ccp_method": estimator_config.ccp_method,
            "ccp_poly_degree": estimator_config.ccp_poly_degree,
            "cross_fitting": estimator_config.cross_fitting,
            "robust_se": estimator_config.robust_se,
            "compute_se": estimator_config.compute_se,
            "outer_max_iter": estimator_config.outer_max_iter,
            "outer_tol": estimator_config.outer_tol,
        },
        "parameter_names": utility.parameter_names,
        "true_parameters": true_params,
        "mean_estimate": theta_array.mean(axis=0),
        "bias": errors.mean(axis=0),
        "rmse": np.sqrt(np.mean(errors**2, axis=0)),
        "empirical_sd": empirical_sd,
        "mean_standard_error": se_array.mean(axis=0),
        "median_standard_error": np.median(se_array, axis=0),
        "coverage_95_by_parameter": coverage_array.mean(axis=0),
        "coverage_95_overall": float(np.mean(coverage_array)),
        "mean_parameter_relative_rmse": float(np.mean(relative_rmse)),
        "median_parameter_relative_rmse": float(np.median(relative_rmse)),
        "gate_passing_replications": int(sum(count == 0 for count in gate_fail_counts)),
        "mean_gate_pass_count": float(np.mean(gate_pass_counts)),
        "zeta_moment_norm_max": float(np.nanmax(moment_norms)),
        "lambda_fixed_point_residual_norm_max": float(np.nanmax(lambda_residual_norms)),
        "lambda_fixed_point_residual_rms_max": float(np.nanmax(lambda_residual_rms)),
        "lambda_fixed_point_residual_max_abs": float(np.nanmax(lambda_residual_max_abs)),
        "preliminary_optimizer_all_success_count": int(sum(preliminary_all_success)),
        "preliminary_optimizer_all_stationary_count": int(sum(preliminary_all_stationary)),
        "preliminary_projected_gradient_norm_max": float(
            np.nanmax(preliminary_projected_gradient_norms)
        ),
        "preliminary_optimizer_message_counts": message_counts,
        "robust_optimizer_all_success_count": int(sum(robust_all_success)),
        "robust_optimizer_all_stationary_count": int(sum(robust_all_stationary)),
        "robust_projected_gradient_norm_max": float(np.nanmax(robust_projected_gradient_norms)),
        "covariance_units": sorted(set(covariance_units)),
        "runtime_seconds": {
            "mean": float(np.mean(estimation_times)),
            "median": float(np.median(estimation_times)),
            "max": float(np.max(estimation_times)),
            "sum": float(np.sum(estimation_times)),
        },
        "replications": per_replication,
        "notes": (
            "Coverage uses individual-clustered Algorithm 2 sandwich standard "
            "errors on repeated panels from the certified encoded-state DGP. "
            "Preliminary plug-in optimizer success is reported separately "
            "from the final robust zeta solve."
        ),
    }
    run_hash = stable_hash(
        {
            "case": HARD_CASE_ID,
            "purpose": payload["purpose"],
            "simulation": payload["simulation"],
            "estimator_config": payload["estimator_config"],
        }
    )
    run_dir = args.output_dir / f"{HARD_CASE_ID}_mc_coverage_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "run_dir": run_dir}


def run_raw_neural_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run the raw neural/neural shapeshifter diagnostic.

    This is separate from the shared known-truth harness because the truth is a
    frozen reward matrix, not a finite structural theta. The estimator receives
    a saturated state-action utility, starts from its ordinary zero vector, and
    uses known transitions only for final evaluation.
    """
    env_config = ShapeshifterConfig(
        num_states=32,
        num_actions=3,
        num_features=4,
        reward_type="neural",
        feature_type="neural",
        action_dependent=True,
        stochastic_transitions=True,
        stochastic_rewards=False,
        num_periods=None,
        discount_factor=0.95,
        seed=args.hard_seed,
    )
    env = ShapeshifterEnvironment(env_config)
    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    truth = value_iteration(
        operator,
        env.true_reward_matrix,
        tol=1e-10,
        max_iter=10_000,
    )
    initial_distribution = jnp.asarray(env._get_initial_state_distribution())
    panel = simulate_panel_from_policy(
        env.problem_spec,
        env.transition_matrices,
        truth.policy,
        initial_distribution,
        n_individuals=args.hard_n_individuals,
        n_periods=args.hard_n_periods,
        seed=args.hard_seed,
    )
    utility = make_state_action_tabular_utility(
        env.num_states,
        env.num_actions,
        parameter_prefix="r",
    )
    estimator_config = TDCCPConfig(
        method="semigradient",
        basis_type="tabular",
        cross_fitting=False,
        robust_se=False,
        compute_se=False,
        n_policy_iterations=1,
        outer_max_iter=500,
        outer_tol=1e-8,
        verbose=args.verbose,
    )
    estimator = TDCCPEstimator(config=estimator_config, seed=args.hard_seed)
    summary = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    metrics = evaluate_hard_case_summary(env, summary, truth=truth)
    gates = tdccp_hard_case_gates(summary, metrics)
    payload = {
        "case_id": RAW_NEURAL_DIAGNOSTIC_ID,
        "estimator": ESTIMATOR,
        "environment_config": env_config,
        "simulation": {
            "n_individuals": args.hard_n_individuals,
            "n_periods": args.hard_n_periods,
            "seed": args.hard_seed,
        },
        "estimator_config": {
            "method": estimator_config.method,
            "basis_type": estimator_config.basis_type,
            "cross_fitting": estimator_config.cross_fitting,
            "robust_se": estimator_config.robust_se,
            "compute_se": estimator_config.compute_se,
            "n_policy_iterations": estimator_config.n_policy_iterations,
            "outer_max_iter": estimator_config.outer_max_iter,
            "outer_tol": estimator_config.outer_tol,
        },
        "summary": summary,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gate.passed for gate in gates),
        "notes": (
            "Neural reward has no finite true theta; parameter-cosine gates "
            "are intentionally omitted."
        ),
    }
    run_hash = stable_hash(
        {
            "case": RAW_NEURAL_DIAGNOSTIC_ID,
            "environment": env_config,
            "simulation": payload["simulation"],
            "estimator_config": payload["estimator_config"],
        }
    )
    run_dir = args.output_dir / f"{RAW_NEURAL_DIAGNOSTIC_ID}_tdccp_{run_hash}"
    write_json(run_dir / "result.json", compact_hard_case_payload(payload, role="diagnostic"))
    return {"payload": payload, "run_dir": run_dir}


def evaluate_hard_case_summary(
    env: ShapeshifterEnvironment,
    summary: Any,
    *,
    truth: Any | None = None,
) -> dict[str, Any]:
    """Compare a TD-CCP hard-case estimate with shapeshifter solver truth."""
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_reward = jnp.asarray(env.true_reward_matrix)
    if truth is None:
        truth = value_iteration(
            SoftBellmanOperator(problem, transitions),
            true_reward,
            tol=1e-10,
            max_iter=10_000,
        )

    estimated_reward = jnp.asarray(summary.parameters).reshape(
        env.num_states,
        env.num_actions,
    )
    estimated_policy = jnp.asarray(summary.policy)
    estimated_value = jnp.asarray(summary.value_function)
    estimated_q = q_from_value(
        estimated_reward,
        estimated_value,
        transitions,
        problem.discount_factor,
    )

    metrics: dict[str, Any] = {
        "reward_rmse": rmse(estimated_reward, true_reward),
        "reward_normalized_rmse": normalized_rmse(estimated_reward, true_reward),
        "reward_truth_rms": rms(true_reward),
        "policy_tv": policy_tv(truth.policy, estimated_policy),
        "value_rmse": rmse(estimated_value, truth.V),
        "value_normalized_rmse": normalized_rmse(estimated_value, truth.V),
        "value_truth_rms": rms(truth.V),
        "q_rmse": rmse(estimated_q, truth.Q),
        "q_normalized_rmse": normalized_rmse(estimated_q, truth.Q),
        "q_truth_rms": rms(truth.Q),
        "counterfactuals": {},
    }

    initial_distribution = jnp.asarray(env._get_initial_state_distribution())
    for kind, cf in hard_case_counterfactuals(env).items():
        cf_reward = cf["reward"]
        cf_transitions = cf["transitions"]
        cf_operator = SoftBellmanOperator(problem, cf_transitions)
        oracle = value_iteration(cf_operator, cf_reward, tol=1e-10, max_iter=10_000)
        estimated_cf_reward = estimated_reward + (cf_reward - true_reward)
        estimated_cf = value_iteration(
            cf_operator,
            estimated_cf_reward,
            tol=1e-10,
            max_iter=10_000,
        )
        estimated_cf_value = evaluate_policy_value(
            reward=cf_reward,
            transitions=cf_transitions,
            policy=estimated_cf.policy,
            discount_factor=problem.discount_factor,
            initial_distribution=initial_distribution,
            scale_parameter=problem.scale_parameter,
        )
        regret = float(jnp.dot(initial_distribution, oracle.V - estimated_cf_value))
        metrics["counterfactuals"][kind] = {
            "description": cf["description"],
            "policy_tv": policy_tv(oracle.policy, estimated_cf.policy),
            "value_rmse": rmse(estimated_cf_value, oracle.V),
            "regret": regret,
        }

    return metrics


def evaluate_paper_hard_case_summary(
    env: ShapeshifterEnvironment,
    utility: ActionDependentReward,
    true_params: jnp.ndarray,
    true_reward: jnp.ndarray,
    summary: Any,
    *,
    truth: Any | None = None,
) -> dict[str, Any]:
    """Compare the finite-theta hard case against structural truth."""
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = jnp.asarray(true_params)
    true_reward = jnp.asarray(true_reward)
    if truth is None:
        truth = value_iteration(
            SoftBellmanOperator(problem, transitions),
            true_reward,
            tol=1e-10,
            max_iter=10_000,
        )

    estimated_params = jnp.asarray(summary.parameters)
    estimated_reward = utility.compute(estimated_params)
    estimated_policy = jnp.asarray(summary.policy)
    estimated_value = jnp.asarray(summary.value_function)
    estimated_q = q_from_value(
        estimated_reward,
        estimated_value,
        transitions,
        problem.discount_factor,
    )

    metrics: dict[str, Any] = {
        "parameters": parameter_metric_dict(true_params, estimated_params),
        "reward_rmse": rmse(estimated_reward, true_reward),
        "reward_normalized_rmse": normalized_rmse(estimated_reward, true_reward),
        "reward_truth_rms": rms(true_reward),
        "policy_tv": policy_tv(truth.policy, estimated_policy),
        "value_rmse": rmse(estimated_value, truth.V),
        "value_normalized_rmse": normalized_rmse(estimated_value, truth.V),
        "value_truth_rms": rms(truth.V),
        "q_rmse": rmse(estimated_q, truth.Q),
        "q_normalized_rmse": normalized_rmse(estimated_q, truth.Q),
        "q_truth_rms": rms(truth.Q),
        "counterfactuals": {},
    }

    initial_distribution = jnp.asarray(env._get_initial_state_distribution())
    for kind, cf in hard_case_counterfactuals(env, true_reward=true_reward).items():
        cf_reward = cf["reward"]
        cf_transitions = cf["transitions"]
        cf_operator = SoftBellmanOperator(problem, cf_transitions)
        oracle = value_iteration(cf_operator, cf_reward, tol=1e-10, max_iter=10_000)
        estimated_cf_reward = estimated_reward + (cf_reward - true_reward)
        estimated_cf = value_iteration(
            cf_operator,
            estimated_cf_reward,
            tol=1e-10,
            max_iter=10_000,
        )
        estimated_cf_value = evaluate_policy_value(
            reward=cf_reward,
            transitions=cf_transitions,
            policy=estimated_cf.policy,
            discount_factor=problem.discount_factor,
            initial_distribution=initial_distribution,
            scale_parameter=problem.scale_parameter,
        )
        regret = float(jnp.dot(initial_distribution, oracle.V - estimated_cf_value))
        metrics["counterfactuals"][kind] = {
            "description": cf["description"],
            "policy_tv": policy_tv(oracle.policy, estimated_cf.policy),
            "value_rmse": rmse(estimated_cf_value, oracle.V),
            "regret": regret,
        }

    return metrics


def parameter_metric_dict(truth: jnp.ndarray, estimated: jnp.ndarray) -> dict[str, float]:
    truth = jnp.asarray(truth)
    estimated = jnp.asarray(estimated)
    error = estimated - truth
    error_rmse = float(jnp.sqrt(jnp.mean(error**2)))
    truth_rms = max(float(jnp.sqrt(jnp.mean(truth**2))), 1e-12)
    denom = float(jnp.linalg.norm(truth) * jnp.linalg.norm(estimated))
    cosine = float(jnp.dot(truth, estimated) / denom) if denom > 1e-12 else float("nan")
    return {
        "rmse": error_rmse,
        "relative_rmse": error_rmse / truth_rms,
        "max_abs_error": float(jnp.max(jnp.abs(error))),
        "cosine_similarity": cosine,
    }


def tdccp_paper_hard_case_gates(
    summary: Any,
    metrics: dict[str, Any],
) -> list[RecoveryGate]:
    """Strict gates for the finite-theta encoded-state hard case."""
    paper_inference = summary.metadata.get("paper_inference") or {}
    se_array = np.asarray(summary.standard_errors, dtype=float)
    cov_array = (
        np.asarray(summary.variance_covariance, dtype=float)
        if summary.variance_covariance is not None
        else np.full((len(se_array), len(se_array)), np.nan)
    )
    if np.all(np.isfinite(cov_array)):
        eig_min = float(np.linalg.eigvalsh(0.5 * (cov_array + cov_array.T)).min())
    else:
        eig_min = float("nan")

    gates = [
        bool_gate("converged", bool(summary.converged), True),
        bool_gate(
            "algorithm2_locally_robust_path",
            summary.metadata.get("se_method_detail") == "tdccp_algorithm2_locally_robust",
            True,
        ),
        bool_gate(
            "finite_positive_standard_errors",
            bool(np.all(np.isfinite(se_array)) and np.all(se_array > 0.0)),
            True,
        ),
        numeric_gate(
            "zeta_moment_norm",
            float(paper_inference.get("moment_norm_max", float("inf"))),
            "<=",
            1e-4,
        ),
        numeric_gate("covariance_min_eigenvalue", eig_min, ">=", -1e-10),
        numeric_gate("parameter_cosine", metrics["parameters"]["cosine_similarity"], ">=", 0.99),
        numeric_gate("parameter_relative_rmse", metrics["parameters"]["relative_rmse"], "<=", 0.15),
        numeric_gate("reward_normalized_rmse", metrics["reward_normalized_rmse"], "<=", 0.08),
        numeric_gate("policy_tv", metrics["policy_tv"], "<=", 0.03),
        numeric_gate("value_normalized_rmse", metrics["value_normalized_rmse"], "<=", 0.10),
        numeric_gate("q_normalized_rmse", metrics["q_normalized_rmse"], "<=", 0.10),
    ]
    for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
        gates.append(numeric_gate(f"{kind}_regret", cf_metrics["regret"], "<=", 0.05))
    return gates


def tdccp_hard_case_gates(summary: Any, metrics: dict[str, Any]) -> list[RecoveryGate]:
    """Strict hard-case gates; no parameter-cosine check for neural reward."""
    gates = [
        bool_gate("converged", bool(summary.converged), True),
        numeric_gate("reward_normalized_rmse", metrics["reward_normalized_rmse"], "<=", 0.10),
        numeric_gate("policy_tv", metrics["policy_tv"], "<=", 0.03),
        numeric_gate("value_normalized_rmse", metrics["value_normalized_rmse"], "<=", 0.10),
        numeric_gate("q_normalized_rmse", metrics["q_normalized_rmse"], "<=", 0.10),
    ]
    for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
        gates.append(numeric_gate(f"{kind}_regret", cf_metrics["regret"], "<=", 0.05))
    return gates


def hard_case_counterfactuals(
    env: ShapeshifterEnvironment,
    *,
    true_reward: jnp.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    if true_reward is None:
        true_reward = jnp.asarray(env.true_reward_matrix)
    else:
        true_reward = jnp.asarray(true_reward)
    transitions = jnp.asarray(env.transition_matrices)
    coords = env.encode_states(jnp.arange(env.num_states))
    progress = coords[:, 0]

    type_a_reward = true_reward - 0.25 * progress[:, None]
    type_b_transitions = skip_action_transitions(env, action=0, skip=2)
    type_c_reward = true_reward.at[:, 1].add(-1_000.0)

    return {
        "type_a": {
            "description": "state-progress reward shift with baseline transitions",
            "reward": type_a_reward,
            "transitions": transitions,
        },
        "type_b": {
            "description": "action-0 transition skip with baseline reward",
            "reward": true_reward,
            "transitions": type_b_transitions,
        },
        "type_c": {
            "description": "disable action 1 with a large reward penalty",
            "reward": type_c_reward,
            "transitions": transitions,
        },
    }


def skip_action_transitions(
    env: ShapeshifterEnvironment,
    *,
    action: int,
    skip: int,
) -> jnp.ndarray:
    transitions = np.asarray(env.transition_matrices).copy()
    transitions[action, :, :] = 0.0
    for state in range(env.num_states):
        target = min(state + skip, env.num_states - 1)
        transitions[action, state, target] = 1.0
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    return jnp.asarray(transitions, dtype=jnp.float32)


def evaluate_policy_value(
    *,
    reward: jnp.ndarray,
    transitions: jnp.ndarray,
    policy: jnp.ndarray,
    discount_factor: float,
    initial_distribution: jnp.ndarray,
    scale_parameter: float,
) -> jnp.ndarray:
    del initial_distribution
    clipped_policy = jnp.clip(policy, 1e-12, 1.0)
    entropy_flow = -scale_parameter * jnp.sum(policy * jnp.log(clipped_policy), axis=1)
    reward_pi = jnp.sum(policy * reward, axis=1) + entropy_flow
    transition_pi = jnp.einsum("sa,ast->st", policy, transitions)
    lhs = jnp.eye(reward.shape[0]) - discount_factor * transition_pi
    return jnp.linalg.solve(lhs, reward_pi)


def q_from_value(
    reward: jnp.ndarray,
    value: jnp.ndarray,
    transitions: jnp.ndarray,
    discount_factor: float,
) -> jnp.ndarray:
    continuation = jnp.einsum("ast,t->as", transitions, value).T
    return reward + discount_factor * continuation


def rmse(estimated: jnp.ndarray, truth: jnp.ndarray) -> float:
    return float(jnp.sqrt(jnp.mean((jnp.asarray(estimated) - jnp.asarray(truth)) ** 2)))


def rms(values: jnp.ndarray) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.asarray(values) ** 2)))


def normalized_rmse(estimated: jnp.ndarray, truth: jnp.ndarray) -> float:
    return rmse(estimated, truth) / max(rms(truth), 1e-12)


def policy_tv(truth: jnp.ndarray, estimated: jnp.ndarray) -> float:
    l1_by_state = jnp.abs(jnp.asarray(truth) - jnp.asarray(estimated)).sum(axis=1)
    return float(0.5 * jnp.mean(l1_by_state))


def numeric_gate(
    name: str,
    value: float,
    operator: str,
    threshold: float,
) -> RecoveryGate:
    if operator == "<=":
        passed = float(value) <= threshold
    elif operator == ">=":
        passed = float(value) >= threshold
    else:
        raise ValueError(f"unknown gate operator {operator!r}")
    return RecoveryGate(
        name=name,
        value=float(value),
        operator=operator,
        threshold=float(threshold),
        passed=bool(passed),
    )


def bool_gate(name: str, value: bool, threshold: bool) -> RecoveryGate:
    return RecoveryGate(
        name=name,
        value=bool(value),
        operator="is",
        threshold=bool(threshold),
        passed=bool(value) == bool(threshold),
    )


def render_results_tex(
    records: list[dict[str, Any]],
    hard_record: dict[str, Any] | None = None,
    raw_neural_record: dict[str, Any] | None = None,
    monte_carlo_record: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("% Auto-generated by tdccp_run.py. Do not edit by hand.")
    add("% Known-truth DGP cells: " + ", ".join(r["payload"]["cell"].cell_id for r in records))
    add("")
    add(r"\section{Known-Truth Validation Results}")
    add(
        "TD-CCP is transition-free "
        "for structural parameter estimation: the estimator uses observed "
        "state-action-next-state tuples to estimate the recursive terms "
        "\\(h(a,x)\\) and \\(g(a,x)\\). Known transitions enter only after "
        "estimation, when the recovered reward is evaluated for policies, values, "
        "and counterfactuals."
    )
    add("")
    add(
        "The paper-faithful hard flexible DGP is the certified release path: "
        "it uses two-dimensional encoded state coordinates, logit first-stage "
        "CCPs, Algorithm 2 cross-fitting, locally robust standard errors, and "
        "a finite linear structural reward parameter. The low-dimensional "
        "action-dependent cell is retained as a sanity check. The "
        "high-dimensional encoded-state cell and raw neural reward cell are "
        "diagnostic stress tests, not release certification."
    )
    add("")

    context_ids = [r["payload"]["cell"].cell_id for r in records]
    if hard_record is not None:
        context_ids.append(hard_record["payload"]["case_id"])
    if raw_neural_record is not None:
        context_ids.append(raw_neural_record["payload"]["case_id"])
    add(r"\subsection{Validation Design Context}")
    add(validation_context(context_ids))
    add("")

    add(r"\subsection{Validation Cells}")
    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP known-truth cells.}")
    add(r"\begin{tabular}{@{}llrrrrrr@{}}")
    add(r"\toprule")
    add(r"Cell & Role & States & State dim. & Reward params & Iter. & Gates pass & Gates fail \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        dgp = record["dgp"]
        gates = payload["gates"]
        passed = sum(gate.passed for gate in gates)
        role = validation_role(
            payload["cell"].cell_id,
            CELL_ROLES.get(payload["cell"].cell_id, "diagnostic"),
        )
        add(
            f"{tex_text(validation_display_name(payload['cell'].cell_id))} & "
            f"{tex_text(role)} & "
            f"{dgp.problem.num_states} & {dgp.problem.state_dim} & "
            f"{dgp.feature_matrix.shape[-1]} & {int(payload['summary'].num_iterations)} & "
            f"{passed} & {len(gates) - passed} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\subsection{Estimator Settings}")
    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP settings used by the shared harness.}")
    add(r"\begin{tabular}{lrrrrr}")
    add(r"\toprule")
    add(r"Cell & Basis & CCP smoothing & L2 penalty & Cross-fit & Transitions for theta \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        metadata = payload["summary"].metadata
        add(
            f"{tex_text(validation_display_name(payload['cell'].cell_id))} & "
            f"{tex_text(str(metadata.get('basis_type')))}-{metadata.get('basis_dim')} & "
            f"{fmt(metadata.get('ccp_smoothing'), 2)} & "
            f"{fmt(metadata.get('theta_l2_penalty'), 0)} & "
            f"{tf(metadata.get('cross_fitting'))} & no \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\subsection{Recovery Metrics}")
    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Core structural recovery metrics.}")
    add(r"\begin{tabular}{lrrrrrr}")
    add(r"\toprule")
    add(r"Cell & Param. cos. & Param. rel. RMSE & Reward RMSE & Policy TV & Value RMSE & Q RMSE \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        metrics = payload["metrics"]
        add(
            f"{tex_text(validation_display_name(payload['cell'].cell_id))} & "
            f"{fmt(metrics['parameters'].cosine_similarity, 3)} & "
            f"{fmt(metrics['parameters'].relative_rmse, 3)} & "
            f"{fmt(metrics['reward_rmse'], 3)} & "
            f"{fmt(metrics['policy'].tv, 3)} & "
            f"{fmt(metrics['value_rmse'], 3)} & "
            f"{fmt(metrics['q_rmse'], 3)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\subsection{Gate Audit}")
    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Hard-gate audit.}")
    add(r"\begin{tabular}{llrrr}")
    add(r"\toprule")
    add(r"Cell & Gate & Value & Threshold & Pass \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        for gate in payload["gates"]:
            threshold = f"{gate.operator} {gate_text(gate.threshold)}"
            add(
                f"{tex_text(validation_display_name(payload['cell'].cell_id))} & "
                f"{tex_text(gate.name)} & "
                f"{gate_text(gate.value)} & "
                f"{tex_text(threshold)} & {tf(gate.passed)} \\\\"
            )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    if hard_record is not None:
        add_hard_case_tex(lines, hard_record["payload"])
    if monte_carlo_record is not None:
        add_monte_carlo_tex(lines, monte_carlo_record["payload"])
    if raw_neural_record is not None:
        add_raw_neural_diagnostic_tex(lines, raw_neural_record["payload"])
    return "\n".join(lines) + "\n"


def add_hard_case_tex(lines: list[str], payload: dict[str, Any]) -> None:
    add = lines.append
    metrics = payload["metrics"]
    gates = payload["gates"]
    failed = [gate for gate in gates if not gate.passed]
    passed = not failed
    summary = payload["summary"]
    paper_inference = summary.metadata.get("paper_inference") or {}

    add(r"\subsection{Encoded-State Locally Robust Showcase}")
    add(
        "The certified hard DGP uses stochastic transitions and "
        "two-dimensional encoded state coordinates. The reward is linear in "
        "an intercept and the two state coordinates, interacted with "
        "nonbaseline actions; action 0 is the utility normalization. The "
        "estimator uses logit CCPs, an encoded semi-gradient basis, "
        "Algorithm 2 cross-fitting, and locally robust standard errors."
    )
    add("")
    if passed:
        add(
            "The encoded-state locally robust DGP passes all gates. This is "
            "the condition under which the RTD page may describe the "
            "finite-theta TD-CCP hard-case showcase as validated."
        )
    else:
        failed_names = ", ".join(tex_text(gate.name) for gate in failed)
        add(
            "This encoded-state locally robust DGP is not gate-certified. "
            "The record is kept as a failure case, and no RTD success "
            "claim should be made. "
            f"Failed gates: {failed_names}."
        )
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP encoded-state locally robust DGP settings and gate summary.}")
    add(r"\begin{tabular}{llrrrr}")
    add(r"\toprule")
    add(r"Case & Reward / features & Obs. & Params & Gates pass & Gates fail \\")
    add(r"\midrule")
    simulation = payload["simulation"]
    observations = int(simulation["n_individuals"]) * int(simulation["n_periods"])
    reward_params = len(payload["summary"].parameters)
    add(
        f"{tex_text(validation_display_name(payload['case_id']))} & linear / encoded & "
        f"{observations} & {reward_params} & "
        f"{sum(gate.passed for gate in gates)} & {len(failed)} \\\\"
    )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP encoded-state locally robust DGP fit summary.}")
    add(r"\begin{tabular}{lrrrrr}")
    add(r"\toprule")
    add(r"Case & Converged & Iter. & Log likelihood & Est. time & Obs. \\")
    add(r"\midrule")
    add(
        f"{tex_text(validation_display_name(payload['case_id']))} & "
        f"{tf(summary.converged)} & {int(summary.num_iterations)} & "
        f"{fmt(summary.log_likelihood, 4)} & "
        f"{fmt(summary.estimation_time, 2)} seconds & "
        f"{int(summary.num_observations)} \\\\"
    )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP locally robust inference diagnostics.}")
    add(r"\begin{tabular}{lrrrrrrr}")
    add(r"\toprule")
    add(
        r"Case & Cov. unit & Prelim. stat. & Prelim. pgrad & Robust OK & "
        r"Max. zeta norm & Lambda max abs & Max SE \\"
    )
    add(r"\midrule")
    standard_errors = np.asarray(summary.standard_errors, dtype=float)
    prelim = paper_inference.get("preliminary_optimizer_stationary", [])
    robust = paper_inference.get("robust_optimizer_success", [])
    add(
        f"{tex_text(validation_display_name(payload['case_id']))} & "
        f"{tex_text(str(paper_inference.get('covariance_unit', 'unknown')))} & "
        f"{sum(bool(value) for value in prelim)}/{len(prelim)} & "
        f"{fmt(paper_inference.get('preliminary_projected_gradient_norm_max'), 3)} & "
        f"{sum(bool(value) for value in robust)}/{len(robust)} & "
        f"{fmt(paper_inference.get('moment_norm_max'), 3)} & "
        f"{fmt(paper_inference.get('lambda_fixed_point_residual_max_abs'), 3)} & "
        f"{fmt(np.max(standard_errors), 3)} \\\\"
    )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Hard flexible DGP recovery metrics. Normalized RMSE divides by the truth RMS.}")
    add(r"\begin{tabular}{lrrrrrrr}")
    add(r"\toprule")
    add(
        r"Case & Param. cos. & Param. rel. RMSE & Reward nRMSE & Policy TV & "
        r"Value nRMSE & Q nRMSE \\"
    )
    add(r"\midrule")
    param_metrics = metrics["parameters"]
    add(
        f"{tex_text(validation_display_name(payload['case_id']))} & "
        f"{fmt(param_metrics['cosine_similarity'], 3)} & "
        f"{fmt(param_metrics['relative_rmse'], 3)} & "
        f"{fmt(metrics['reward_normalized_rmse'], 3)} & "
        f"{fmt(metrics['policy_tv'], 3)} & "
        f"{fmt(metrics['value_normalized_rmse'], 3)} & "
        f"{fmt(metrics['q_normalized_rmse'], 3)} \\\\"
    )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Hard flexible DGP counterfactual regret checks.}")
    add(r"\begin{tabular}{lrrr}")
    add(r"\toprule")
    add(r"Counterfactual & Policy TV & Value RMSE & Regret \\")
    add(r"\midrule")
    for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
        add(
            f"{tex_text(counterfactual_label(kind))} & "
            f"{fmt(cf_metrics['policy_tv'], 3)} & "
            f"{fmt(cf_metrics['value_rmse'], 3)} & "
            f"{fmt(cf_metrics['regret'], 3)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Hard flexible DGP gate audit.}")
    add(r"\begin{tabular}{lrrr}")
    add(r"\toprule")
    add(r"Gate & Value & Threshold & Pass \\")
    add(r"\midrule")
    for gate in gates:
        threshold = f"{gate.operator} {gate_text(gate.threshold)}"
        add(
            f"{tex_text(gate.name)} & {gate_text(gate.value)} & "
            f"{tex_text(threshold)} & {tf(gate.passed)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")


def add_monte_carlo_tex(lines: list[str], payload: dict[str, Any]) -> None:
    add = lines.append
    simulation = payload["simulation"]
    n_obs = int(simulation["n_individuals"]) * int(simulation["n_periods"])

    add("")
    add(r"\subsection{Repeated-Seed Standard Error Coverage}")
    add(
        "The coverage check repeatedly simulates the certified encoded-state "
        "DGP and re-estimates TD-CCP with the same Algorithm 2 settings. "
        "Reported standard errors use individual-clustered fold covariances. "
        "Preliminary plug-in optimizer status is reported separately because "
        "the final robust zeta solve is the binding convergence check."
    )
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP repeated-seed coverage summary.}")
    add(r"\begin{tabular}{lrrrrrr}")
    add(r"\toprule")
    add(r"Case & Reps. & Obs./rep. & Coverage & Mean rel. RMSE & Gate pass reps. & Robust OK \\")
    add(r"\midrule")
    add(
        f"{tex_text(validation_display_name(payload['case_id']))} & "
        f"{int(payload['n_replications'])} & {n_obs} & "
        f"{fmt(payload['coverage_95_overall'], 3)} & "
        f"{fmt(payload['mean_parameter_relative_rmse'], 3)} & "
        f"{int(payload['gate_passing_replications'])} & "
        f"{int(payload['robust_optimizer_all_success_count'])} \\\\"
    )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP repeated-seed parameter uncertainty diagnostics.}")
    add(r"\begin{tabular}{lrrrrr}")
    add(r"\toprule")
    add(r"Parameter & Bias & RMSE & Emp. SD & Mean SE & 95\% cov. \\")
    add(r"\midrule")
    for name, bias, rmse_value, empirical_sd, mean_se, coverage in zip(
        payload["parameter_names"],
        payload["bias"],
        payload["rmse"],
        payload["empirical_sd"],
        payload["mean_standard_error"],
        payload["coverage_95_by_parameter"],
    ):
        add(
            f"{tex_text(name)} & "
            f"{fmt(bias, 3)} & {fmt(rmse_value, 3)} & "
            f"{fmt(empirical_sd, 3)} & {fmt(mean_se, 3)} & "
            f"{fmt(coverage, 3)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{TD-CCP repeated-seed inference diagnostics.}")
    add(r"\begin{tabular}{lrrrrr}")
    add(r"\toprule")
    add(r"Case & Cov. unit & Max zeta norm & Max lambda abs & Prelim. stat. & Prelim. pgrad \\")
    add(r"\midrule")
    covariance_units = ", ".join(str(unit) for unit in payload["covariance_units"])
    add(
        f"{tex_text(validation_display_name(payload['case_id']))} & "
        f"{tex_text(covariance_units)} & "
        f"{fmt(payload['zeta_moment_norm_max'], 3)} & "
        f"{fmt(payload['lambda_fixed_point_residual_max_abs'], 3)} & "
        f"{int(payload['preliminary_optimizer_all_stationary_count'])} & "
        f"{fmt(payload['preliminary_projected_gradient_norm_max'], 3)} \\\\"
    )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")


def add_raw_neural_diagnostic_tex(lines: list[str], payload: dict[str, Any]) -> None:
    add = lines.append
    metrics = payload["metrics"]
    gates = payload["gates"]
    failed = [gate for gate in gates if not gate.passed]

    add("")
    add(r"\subsection{Raw Neural Reward Diagnostic}")
    add(
        "The raw neural flexible DGP is retained as a diagnostic "
        "failure record. It uses a frozen neural reward matrix with no "
        "finite true \\(\\theta\\). TD-CCP is therefore not claimed to recover "
        "the raw reward matrix in this case; policy and counterfactual gates "
        "are reported separately from the finite-theta showcase."
    )
    add("")
    failed_names = ", ".join(tex_text(gate.name) for gate in failed)
    add(f"Failed diagnostic gates: {failed_names}.")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Raw neural reward diagnostic metrics.}")
    add(r"\begin{tabular}{lrrrrr}")
    add(r"\toprule")
    add(r"Case & Reward nRMSE & Policy TV & Value nRMSE & Q nRMSE & Gates fail \\")
    add(r"\midrule")
    add(
        f"{tex_text(validation_display_name(payload['case_id']))} & "
        f"{fmt(metrics['reward_normalized_rmse'], 3)} & "
        f"{fmt(metrics['policy_tv'], 3)} & "
        f"{fmt(metrics['value_normalized_rmse'], 3)} & "
        f"{fmt(metrics['q_normalized_rmse'], 3)} & "
        f"{len(failed)} \\\\"
    )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Raw neural diagnostic counterfactual regret checks.}")
    add(r"\begin{tabular}{lrrr}")
    add(r"\toprule")
    add(r"Counterfactual & Policy TV & Value RMSE & Regret \\")
    add(r"\midrule")
    for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
        add(
            f"{tex_text(counterfactual_label(kind))} & "
            f"{fmt(cf_metrics['policy_tv'], 3)} & "
            f"{fmt(cf_metrics['value_rmse'], 3)} & "
            f"{fmt(cf_metrics['regret'], 3)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\scriptsize")
    add(r"\caption{Raw neural diagnostic gate audit.}")
    add(r"\begin{tabular}{lrrr}")
    add(r"\toprule")
    add(r"Gate & Value & Threshold & Pass \\")
    add(r"\midrule")
    for gate in gates:
        threshold = f"{gate.operator} {gate_text(gate.threshold)}"
        add(
            f"{tex_text(gate.name)} & {gate_text(gate.value)} & "
            f"{tex_text(threshold)} & {tf(gate.passed)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")


def compact_payload(
    records: list[dict[str, Any]],
    hard_record: dict[str, Any] | None = None,
    raw_neural_record: dict[str, Any] | None = None,
    monte_carlo_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supporting_results = [compact_cell_payload(record["payload"]) for record in records]
    hard_case = (
        None
        if hard_record is None
        else compact_hard_case_payload(hard_record["payload"], role="certified")
    )
    raw_neural_diagnostic = (
        None
        if raw_neural_record is None
        else compact_hard_case_payload(raw_neural_record["payload"], role="diagnostic")
    )
    monte_carlo = (
        None
        if monte_carlo_record is None
        else compact_monte_carlo_payload(monte_carlo_record["payload"])
    )
    primary_result = hard_case
    primary_cell_id = PRIMARY_CELL_ID
    release_status = RELEASE_STATUS
    if primary_result is None:
        primary_result = supporting_results[0] if supporting_results else None
        primary_cell_id = primary_result["cell_id"] if primary_result is not None else "none"
        release_status = "Diagnostic only"

    return {
        "estimator": ESTIMATOR,
        "paper_target": PAPER_TARGET,
        "primary_cell_id": primary_cell_id,
        "release_status": release_status,
        "result": primary_result,
        "results": supporting_results,
        "hard_case": hard_case,
        "monte_carlo": monte_carlo,
        "raw_neural_diagnostic": raw_neural_diagnostic,
    }


def compact_cell_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    metadata = summary.metadata
    return {
        "cell_id": payload["cell"].cell_id,
        "estimator": payload["estimator"],
        "simulation": payload["simulation"],
        "compatibility": payload["compatibility"],
        "diagnostics": payload["diagnostics"],
        "summary": {
            "parameter_names": summary.parameter_names,
            "parameters": summary.parameters,
            "standard_errors": finite_list(summary.standard_errors),
            "log_likelihood": summary.log_likelihood,
            "converged": summary.converged,
            "num_iterations": summary.num_iterations,
            "num_observations": summary.num_observations,
            "estimation_time": summary.estimation_time,
            "convergence_message": summary.convergence_message,
            "metadata": {
                "method": metadata.get("method"),
                "basis_type": metadata.get("basis_type"),
                "basis_dim": metadata.get("basis_dim"),
                "basis_include_rewards": metadata.get("basis_include_rewards"),
                "basis_ridge": metadata.get("basis_ridge"),
                "basis_pinv_rcond": metadata.get("basis_pinv_rcond"),
                "ccp_method": metadata.get("ccp_method"),
                "ccp_poly_degree": metadata.get("ccp_poly_degree"),
                "ccp_smoothing": metadata.get("ccp_smoothing"),
                "theta_l2_penalty": metadata.get("theta_l2_penalty"),
                "cross_fitting": metadata.get("cross_fitting"),
                "split_unit": metadata.get("split_unit"),
                "robust_se": metadata.get("robust_se"),
                "se_method_detail": metadata.get("se_method_detail"),
            },
        },
        "metrics": payload["metrics"],
        "gates": payload["gates"],
    }


def compact_monte_carlo_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": payload["case_id"],
        "estimator": payload["estimator"],
        "purpose": payload["purpose"],
        "n_replications": payload["n_replications"],
        "successful_replications": payload["successful_replications"],
        "simulation": payload["simulation"],
        "estimator_config": payload["estimator_config"],
        "parameter_names": payload["parameter_names"],
        "true_parameters": finite_list(payload["true_parameters"]),
        "mean_estimate": finite_list(payload["mean_estimate"]),
        "bias": finite_list(payload["bias"]),
        "rmse": finite_list(payload["rmse"]),
        "empirical_sd": finite_list(payload["empirical_sd"]),
        "mean_standard_error": finite_list(payload["mean_standard_error"]),
        "median_standard_error": finite_list(payload["median_standard_error"]),
        "coverage_95_by_parameter": finite_list(payload["coverage_95_by_parameter"]),
        "coverage_95_overall": payload["coverage_95_overall"],
        "mean_parameter_relative_rmse": payload["mean_parameter_relative_rmse"],
        "median_parameter_relative_rmse": payload["median_parameter_relative_rmse"],
        "gate_passing_replications": payload["gate_passing_replications"],
        "mean_gate_pass_count": payload["mean_gate_pass_count"],
        "zeta_moment_norm_max": payload["zeta_moment_norm_max"],
        "lambda_fixed_point_residual_norm_max": payload["lambda_fixed_point_residual_norm_max"],
        "lambda_fixed_point_residual_rms_max": payload["lambda_fixed_point_residual_rms_max"],
        "lambda_fixed_point_residual_max_abs": payload["lambda_fixed_point_residual_max_abs"],
        "preliminary_optimizer_all_success_count": payload[
            "preliminary_optimizer_all_success_count"
        ],
        "preliminary_optimizer_all_stationary_count": payload[
            "preliminary_optimizer_all_stationary_count"
        ],
        "preliminary_projected_gradient_norm_max": payload[
            "preliminary_projected_gradient_norm_max"
        ],
        "preliminary_optimizer_message_counts": payload["preliminary_optimizer_message_counts"],
        "robust_optimizer_all_success_count": payload["robust_optimizer_all_success_count"],
        "robust_optimizer_all_stationary_count": payload["robust_optimizer_all_stationary_count"],
        "robust_projected_gradient_norm_max": payload["robust_projected_gradient_norm_max"],
        "covariance_units": payload["covariance_units"],
        "runtime_seconds": payload["runtime_seconds"],
        "replications": payload["replications"],
        "notes": payload["notes"],
    }


def compact_hard_case_payload(
    payload: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    summary = payload["summary"]
    gates = payload["gates"]
    gate_pass_count = sum(gate.passed for gate in gates)
    ci_low, ci_high = summary.confidence_interval()
    paper_inference = summary.metadata.get("paper_inference") or {}
    fold_summaries = []
    for fold in paper_inference.get("folds", []):
        fold_summaries.append(
            {
                "name": fold.get("name"),
                "source_fold": fold.get("source_fold"),
                "n_observations": fold.get("n_observations"),
                "n_effective_units": fold.get("n_effective_units"),
                "covariance_unit": fold.get("covariance_unit"),
                "theta": fold.get("theta"),
                "tilde_theta": fold.get("tilde_theta"),
                "zeta_mean": fold.get("zeta_mean"),
                "zeta_norm": fold.get("zeta_norm"),
                "lambda_fixed_point_residual_norm": fold.get("lambda_fixed_point_residual_norm"),
                "lambda_fixed_point_residual_rms": fold.get("lambda_fixed_point_residual_rms"),
                "lambda_fixed_point_residual_max_abs": fold.get(
                    "lambda_fixed_point_residual_max_abs"
                ),
                "covariance_diag": np.diag(np.asarray(fold.get("V_asymptotic"))).tolist()
                if fold.get("V_asymptotic") is not None
                else None,
            }
        )
    return {
        "case_id": payload["case_id"],
        "estimator": payload["estimator"],
        "environment_config": payload["environment_config"],
        "utility_config": payload.get("utility_config"),
        "simulation": payload["simulation"],
        "estimator_config": payload["estimator_config"],
        "truth": payload.get("truth"),
        "summary": {
            "parameter_names": summary.parameter_names,
            "parameters": summary.parameters,
            "standard_errors": finite_list(summary.standard_errors),
            "confidence_intervals_95": {
                name: [finite_float(lo), finite_float(hi)]
                for name, lo, hi in zip(summary.parameter_names, ci_low, ci_high)
            },
            "log_likelihood": summary.log_likelihood,
            "converged": summary.converged,
            "num_iterations": summary.num_iterations,
            "num_observations": summary.num_observations,
            "estimation_time": summary.estimation_time,
            "convergence_message": summary.convergence_message,
            "metadata": {
                "method": summary.metadata.get("method"),
                "basis_type": summary.metadata.get("basis_type"),
                "basis_dim": summary.metadata.get("basis_dim"),
                "cross_fitting": summary.metadata.get("cross_fitting"),
                "robust_se": summary.metadata.get("robust_se"),
                "se_method_detail": summary.metadata.get("se_method_detail"),
                "ccp_method": summary.metadata.get("ccp_method"),
                "ccp_poly_degree": summary.metadata.get("ccp_poly_degree"),
                "ccp_smoothing": summary.metadata.get("ccp_smoothing"),
            },
            "paper_inference": {
                "method": paper_inference.get("method"),
                "split_unit": paper_inference.get("split_unit"),
                "covariance_unit": paper_inference.get("covariance_unit"),
                "preliminary_stationarity_tol": paper_inference.get("preliminary_stationarity_tol"),
                "robust_stationarity_tol": paper_inference.get("robust_stationarity_tol"),
                "moment_norm_max": paper_inference.get("moment_norm_max"),
                "preliminary_projected_gradient_norm_max": paper_inference.get(
                    "preliminary_projected_gradient_norm_max"
                ),
                "robust_projected_gradient_norm_max": paper_inference.get(
                    "robust_projected_gradient_norm_max"
                ),
                "lambda_fixed_point_residual_norm_max": paper_inference.get(
                    "lambda_fixed_point_residual_norm_max"
                ),
                "lambda_fixed_point_residual_rms_max": paper_inference.get(
                    "lambda_fixed_point_residual_rms_max"
                ),
                "lambda_fixed_point_residual_max_abs": paper_inference.get(
                    "lambda_fixed_point_residual_max_abs"
                ),
                "preliminary_optimizer_success": paper_inference.get(
                    "preliminary_optimizer_success"
                ),
                "preliminary_optimizer_messages": paper_inference.get(
                    "preliminary_optimizer_messages"
                ),
                "preliminary_optimizer_diagnostics": paper_inference.get(
                    "preliminary_optimizer_diagnostics"
                ),
                "preliminary_optimizer_stationary": paper_inference.get(
                    "preliminary_optimizer_stationary"
                ),
                "robust_optimizer_success": paper_inference.get("robust_optimizer_success"),
                "robust_optimizer_messages": paper_inference.get("robust_optimizer_messages"),
                "robust_optimizer_diagnostics": paper_inference.get("robust_optimizer_diagnostics"),
                "robust_optimizer_stationary": paper_inference.get("robust_optimizer_stationary"),
                "standard_errors": finite_list(paper_inference.get("standard_errors", [])),
                "sample_covariance_diag": np.diag(
                    np.asarray(paper_inference.get("sample_covariance_pd"))
                ).tolist()
                if paper_inference.get("sample_covariance_pd") is not None
                else None,
                "asymptotic_covariance_diag": np.diag(
                    np.asarray(paper_inference.get("asymptotic_covariance"))
                ).tolist()
                if paper_inference.get("asymptotic_covariance") is not None
                else None,
                "folds": fold_summaries,
            },
        },
        "diagnostics": {
            "role": role,
            "passed": payload["passed"],
            "gate_pass_count": gate_pass_count,
            "gate_fail_count": len(gates) - gate_pass_count,
            "estimation_uses_transition_density": False,
            "evaluation_uses_supplied_transitions": True,
        },
        "metrics": payload["metrics"],
        "gates": gates,
        "passed": payload["passed"],
        "notes": payload["notes"],
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "---"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "---"
    return f"{number:.{digits}f}"


def finite_list(values: Any) -> list[float | None]:
    array = np.asarray(values, dtype=float)
    return [float(value) if math.isfinite(float(value)) else None for value in array]


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def tf(value: Any) -> str:
    return "true" if bool(value) else "false"


def gate_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return fmt(value, 3)


def counterfactual_label(kind: str) -> str:
    return {
        "type_a": "Type A",
        "type_b": "Type B",
        "type_c": "Type C",
    }.get(kind, kind.replace("_", " ").title())


def tt(value: str) -> str:
    return r"\texttt{" + tex_text(value) + "}"


def tex_text(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


if __name__ == "__main__":
    main()
