"""Multi-estimator benchmark runner.

Runs each estimator on each cell over R replication seeds, capturing per-fit
parameters, standard errors, policy, value, runtime, and convergence. Estimator
specs are built generically from the cell's environment (so any ArrayMDP works,
not just Rust bus), reusing the same low-level ``.estimate(panel, spec, problem,
transitions)`` path as ``scripts/small_benchmark.py``.

Output is a JSON-serializable dict mirroring the ``validation/results`` schema,
with a freshness stamp and the segmented metrics from ``metrics.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.environments.base import DDCEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

from validation.benchmark import metrics as M
from validation.benchmark.cells import CELLS, BenchmarkCell


# ---------------------------------------------------------------------------
# Estimator specs (generic over any DDCEnvironment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimatorSpec:
    """A benchmarkable estimator.

    Attributes:
        name: Display name.
        family: ``"structural"`` (gets parameter bias/coverage) or
            ``"behavioral"`` (behavioral metrics only). The benchmark never
            puts a parameter-bias number on a behavioral estimator.
        run: ``run(env, panel)`` builds the spec from ``env`` and returns an
            object exposing ``parameters``, ``standard_errors``, ``policy``,
            ``value_function``, ``converged``.
    """

    name: str
    family: str
    run: Callable[[DDCEnvironment, object], object]


def _spec_names(env) -> list[str]:
    """Parameter names for a linear spec built from the environment.

    Uses the environment's own names when it has a finite-dimensional theta.
    When it does not (e.g. a nonlinear-reward DGP whose ``parameter_names`` is
    empty), fall back to one name per feature column so a linear-utility
    estimator can still fit a misspecified linear approximation. Such cells
    carry ``has_finite_theta=False``, so the runner reports only behavioral
    metrics for them and never a (meaningless) parameter bias.
    """
    names = list(env.parameter_names)
    if names:
        return names
    K = int(np.asarray(env.feature_matrix).shape[2])
    return [f"theta_{i}" for i in range(K)]


def _linear_utility(env) -> LinearUtility:
    return LinearUtility(feature_matrix=env.feature_matrix, parameter_names=_spec_names(env))


def _action_reward(env) -> ActionDependentReward:
    return ActionDependentReward(env.feature_matrix, _spec_names(env))


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(
        inner_solver="hybrid", inner_tol=1e-10, inner_max_iter=100000,
        compute_hessian=True, verbose=False,
    )
    return est.estimate(panel, _linear_utility(env),
                        env.problem_spec, env.transition_matrices)


def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=1, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env),
                        env.problem_spec, env.transition_matrices)


def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECEstimator, MPECConfig

    est = MPECEstimator(
        config=MPECConfig(solver="slsqp", max_iter=200, constraint_tol=1e-6),
        compute_hessian=True, verbose=False,
    )
    return est.estimate(panel, _linear_utility(env),
                        env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(
        config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                            inner_max_iter=2000, compute_se=False, verbose=False)
    )
    return est.estimate(panel, _action_reward(env),
                        env.problem_spec, env.transition_matrices)


DEFAULT_ESTIMATORS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec("NFXP", "structural", _run_nfxp),
    EstimatorSpec("CCP", "structural", _run_ccp),
    EstimatorSpec("MPEC", "structural", _run_mpec),
    EstimatorSpec("MCE-IRL", "behavioral", _run_mce_irl),
)


# ---------------------------------------------------------------------------
# Oracle objects
# ---------------------------------------------------------------------------


def _oracle(env: DDCEnvironment) -> tuple[np.ndarray, np.ndarray]:
    """Return the oracle (true-parameter) policy and value function."""
    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    result = value_iteration(operator, env.compute_utility_matrix())
    return np.asarray(result.policy), np.asarray(result.V)


def _as_array(x) -> np.ndarray | None:
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_cell(
    cell: BenchmarkCell,
    estimators: tuple[EstimatorSpec, ...],
    n_replications: int,
    verbose: bool = True,
) -> dict:
    """Run all estimators on one cell over R replications; return its result dict."""
    env = cell.build()
    oracle_policy, oracle_value = _oracle(env)
    has_theta = len(env.parameter_names) > 0
    true_theta = (
        np.asarray(env.get_true_parameter_vector(), dtype=np.float64)
        if has_theta else None
    )
    diagnostics = M.feature_diagnostics(np.asarray(env.feature_matrix))

    # Per-estimator accumulators.
    acc: dict[str, dict] = {
        e.name: {"policy_tv": [], "value_rmse": [], "runtime": [],
                 "converged": 0, "crashed": 0,
                 "estimates": [], "ses": []}
        for e in estimators
    }

    for rep in range(n_replications):
        seed = cell.seed + 1000 + rep
        panel = simulate_panel(env, n_individuals=cell.n_individuals,
                               n_periods=cell.n_periods, seed=seed)
        for spec in estimators:
            t0 = time.time()
            try:
                res = spec.run(env, panel)
                dt = time.time() - t0
                a = acc[spec.name]
                a["runtime"].append(dt)
                if getattr(res, "converged", True):
                    a["converged"] += 1
                pol = _as_array(getattr(res, "policy", None))
                if pol is not None and pol.shape == oracle_policy.shape:
                    a["policy_tv"].append(M.policy_tv(pol, oracle_policy))
                a["value_rmse"].append(
                    M.value_rmse(_as_array(getattr(res, "value_function", None)), oracle_value)
                )
                if spec.family == "structural" and has_theta:
                    params = _as_array(getattr(res, "parameters", None))
                    ses = _as_array(getattr(res, "standard_errors", None))
                    if params is not None and params.shape[0] == true_theta.shape[0]:
                        a["estimates"].append(params)
                        a["ses"].append(ses if (ses is not None and ses.shape == params.shape)
                                        else np.full_like(params, np.nan))
            except Exception as exc:  # noqa: BLE001 - failure is data for the failure map
                acc[spec.name]["crashed"] += 1
                if verbose:
                    print(f"    [{cell.cell_id}/{spec.name}] rep {rep} crashed: {exc}")
        if verbose:
            print(f"  {cell.cell_id}: replication {rep + 1}/{n_replications} done")

    # Aggregate.
    estimator_results = []
    for spec in estimators:
        a = acc[spec.name]
        n_done = len(a["runtime"])
        runtime = np.asarray(a["runtime"], dtype=np.float64)
        entry: dict = {
            "estimator": spec.name,
            "family": spec.family,
            "n_replications": n_replications,
            "n_completed": n_done,
            "n_crashed": a["crashed"],
            "convergence_rate": (a["converged"] / n_done) if n_done else 0.0,
            "runtime_mean": float(runtime.mean()) if runtime.size else None,
            "runtime_sd": float(runtime.std(ddof=1)) if runtime.size > 1 else None,
            "behavioral": M.behavioral_summary(a["policy_tv"], a["value_rmse"]),
            "parameters": None,
        }
        if spec.family == "structural" and has_theta and a["estimates"]:
            entry["parameters"] = M.parameter_summary(
                np.vstack(a["estimates"]), np.vstack(a["ses"]),
                true_theta, list(env.parameter_names),
            )
        estimator_results.append(entry)

    return {
        "cell_id": cell.cell_id,
        "label": cell.label,
        "difficulty": cell.difficulty,
        "stresses": cell.stresses,
        "description": cell.description,
        "num_states": int(env.num_states),
        "num_actions": int(env.num_actions),
        "has_finite_theta": has_theta,
        "discount_factor": float(env.problem_spec.discount_factor),
        "simulation": {"n_individuals": cell.n_individuals,
                       "n_periods": cell.n_periods, "base_seed": cell.seed},
        "diagnostics": diagnostics,
        "estimators": estimator_results,
    }


def run_benchmark(
    cells: tuple[BenchmarkCell, ...] = CELLS,
    estimators: tuple[EstimatorSpec, ...] = DEFAULT_ESTIMATORS,
    n_replications: int = 30,
    date: str = "unknown",
    package_version: str = "unknown",
    verbose: bool = True,
) -> dict:
    """Run the full benchmark and return a JSON-serializable result dict.

    Args:
        cells: Difficulty-ladder cells to run.
        estimators: Estimator specs to evaluate.
        n_replications: Monte Carlo replications per cell.
        date: Generation date stamp (pass in; the runtime cannot read the clock).
        package_version: econirl version stamp.
        verbose: Print progress.

    Returns:
        ``{"meta": {...}, "cells": [...]}`` with a freshness stamp and segmented
        metrics. Behavioral metrics for every estimator; parameter bias/coverage
        only for the structural family.
    """
    results = []
    for cell in cells:
        if verbose:
            print(f"=== cell {cell.cell_id} ({cell.label}) ===")
        results.append(run_cell(cell, estimators, n_replications, verbose=verbose))

    return {
        "meta": {
            "title": "Cross-estimator simulation studies",
            "date": date,
            "package_version": package_version,
            "n_replications": n_replications,
            "estimators": [{"name": e.name, "family": e.family} for e in estimators],
            "coverage_convention": "normal approximation, z=1.96, matches run_monte_carlo",
            "note": (
                "Parameter bias and coverage are reported only for the structural "
                "family with finite-dimensional theta and real standard errors. "
                "Behavioral metrics (policy TV, value RMSE) are comparable across "
                "all estimators."
            ),
        },
        "cells": results,
    }
