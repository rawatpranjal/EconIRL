#!/usr/bin/env python3
"""Implementation-readiness checks for finite-horizon MCE-IRL.

The runner exercises the public estimator entry point on a closed-form binary
choice problem. It checkpoints each fit, checks repeated-run recovery and
confidence-interval coverage, compares asymptotic and bootstrap standard
errors, and verifies that a reward intervention changes behavior.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "mce_irl_ready.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.tasks import MCEIRLTask  # noqa: E402
from econirl.core.types import Panel, Trajectory  # noqa: E402
from econirl.estimators import MCEIRL  # noqa: E402
from econirl.evaluation.selfcheck import (  # noqa: E402
    assert_effect,
    check_coverage,
    check_se_ratio,
)
from econirl.transitions import DeterministicTransitions  # noqa: E402
from validation.estimators.nfxp.ready import (  # noqa: E402
    _git_commit,
    _package_versions,
    _strict_json,
)

TRUE_THETA = 0.4
N_OBSERVATIONS = 400
FULL_REPLICATIONS = 300
FULL_BOOTSTRAPS = 200
THRESHOLDS = {
    "usable_rate_min": 1.0,
    "coverage_low": 0.91,
    "coverage_high": 0.99,
    "absolute_bias_max": 0.05,
    "rmse_max": 0.15,
    "stationarity_max": 1e-6,
    "se_ratio_tolerance": 0.25,
    "counterfactual_effect_min": 0.1,
}


def make_panel(*, seed: int, n_observations: int = N_OBSERVATIONS) -> Panel:
    """Draw a one-step MCE panel with theta equal to ``TRUE_THETA``."""
    rng = np.random.default_rng(seed)
    probability = 1.0 / (1.0 + np.exp(-TRUE_THETA))
    actions = rng.binomial(1, probability, size=n_observations)
    return Panel(
        [
            Trajectory(
                states=jnp.array([0]),
                actions=jnp.array([action]),
                next_states=jnp.array([1]),
                individual_id=index,
                metadata={"task_id": "binary"},
            )
            for index, action in enumerate(actions)
        ]
    )


def model_spec(
    *,
    se_method: str,
    n_bootstrap: int = 0,
    se_seed: int | None = None,
) -> tuple[MCEIRL, DeterministicTransitions, list[MCEIRLTask]]:
    """Construct the public estimator and its fixed deterministic task."""
    transitions = DeterministicTransitions(
        next_state=np.array([[1, 1], [1, -1]]),
        valid_action=np.array([[True, True], [True, False]]),
    )
    features: np.ndarray = np.zeros((2, 2, 1), dtype=np.float32)
    features[0, 1, 0] = 1.0
    tasks = [
        MCEIRLTask(
            task_id="binary",
            initial_state=0,
            terminal_states=np.array([1]),
            horizon=1,
        )
    ]
    model = MCEIRL(
        n_states=2,
        n_actions=2,
        discount=1.0,
        horizon=1,
        feature_matrix=features,
        feature_names=["action_one"],
        se_method=se_method,
        n_bootstrap=n_bootstrap,
        se_seed=se_seed,
        compute_se=True,
    )
    return model, transitions, tasks


def fit_once(replication: int) -> dict[str, Any]:
    """Fit one independently simulated panel."""
    seed = 61_000 + replication
    panel = make_panel(seed=seed)
    model, transitions, tasks = model_spec(se_method="asymptotic")
    started = time.perf_counter()
    try:
        model.fit(panel, transitions=transitions, tasks=tasks)
        estimate = float(model.params_["action_one"])
        standard_error = float(model.se_["action_one"])
        return {
            "replication": replication,
            "seed": seed,
            "estimate": estimate,
            "standard_error": standard_error,
            "lower": estimate - 1.96 * standard_error,
            "upper": estimate + 1.96 * standard_error,
            "converged": bool(model.converged_),
            "termination_reason": model.termination_reason_,
            "stationarity_residual": float(model.feature_residual_),
            "runtime_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are validation evidence
        return {
            "replication": replication,
            "seed": seed,
            "converged": False,
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    """Read completed JSONL checkpoint records."""
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["replication"])] = record
    return records


def run_replications(path: Path, n_replications: int, *, quiet: bool) -> list[dict[str, Any]]:
    """Run or resume the repeated-fit experiment."""
    completed = load_checkpoint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for replication in range(n_replications):
        if replication in completed:
            records.append(completed[replication])
            continue
        record = fit_once(replication)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        if not quiet:
            print(
                f"{replication + 1}/{n_replications}: "
                f"{record.get('error') or record['estimate']:.6}"
            )
    return records


def coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the shared per-tail coverage self-check to completed fits."""
    successful = [record for record in records if record.get("error") is None]
    iterator = iter(successful)
    passed = True
    error = None
    try:
        coverage = check_coverage(
            lambda _rng: next(iterator),
            lambda record: (
                record["estimate"],
                record["lower"],
                record["upper"],
            ),
            TRUE_THETA,
            n_sims=len(successful),
            tol=0.04,
            seed=0,
        )
    except AssertionError as exc:
        coverage = float(
            np.mean([record["lower"] <= TRUE_THETA <= record["upper"] for record in successful])
        )
        passed = False
        error = str(exc)

    estimates = np.asarray([record["estimate"] for record in successful])
    standard_errors = np.asarray([record["standard_error"] for record in successful])
    lower_tail = np.asarray([TRUE_THETA < record["lower"] for record in successful])
    upper_tail = np.asarray([TRUE_THETA > record["upper"] for record in successful])
    return {
        "n_successful": len(successful),
        "coverage": coverage,
        "coverage_selfcheck_passed": passed,
        "coverage_error": error,
        "mean_estimate": float(estimates.mean()),
        "bias": float(estimates.mean() - TRUE_THETA),
        "rmse": float(np.sqrt(np.mean((estimates - TRUE_THETA) ** 2))),
        "monte_carlo_sd": float(estimates.std(ddof=1)),
        "mean_asymptotic_se": float(standard_errors.mean()),
        "mean_se_to_empirical_sd": float(standard_errors.mean() / estimates.std(ddof=1)),
        "lower_tail_miss_rate": float(lower_tail.mean()),
        "upper_tail_miss_rate": float(upper_tail.mean()),
        "convergence_rate": len(successful) / len(records),
        "max_stationarity_residual": float(
            max(record["stationarity_residual"] for record in successful)
        ),
    }


def bootstrap_check(n_bootstrap: int) -> dict[str, Any]:
    """Compare public asymptotic and trajectory-bootstrap standard errors."""
    panel = make_panel(seed=77_001, n_observations=1_000)
    asymptotic, transitions, tasks = model_spec(se_method="asymptotic")
    asymptotic.fit(panel, transitions=transitions, tasks=tasks)
    bootstrap, transitions, tasks = model_spec(
        se_method="bootstrap",
        n_bootstrap=n_bootstrap,
        se_seed=93,
    )
    bootstrap.fit(panel, transitions=transitions, tasks=tasks)
    formula_se = float(asymptotic.se_["action_one"])
    bootstrap_se = float(bootstrap.se_["action_one"])
    passed = True
    error = None
    try:
        ratio = check_se_ratio(formula_se, bootstrap_se, tol=0.25)
    except AssertionError as exc:
        ratio = formula_se / bootstrap_se
        passed = False
        error = str(exc)
    return {
        "n_bootstrap": n_bootstrap,
        "asymptotic_se": formula_se,
        "bootstrap_se": bootstrap_se,
        "ratio": ratio,
        "passed": passed,
        "error": error,
    }


def intervention_check() -> dict[str, Any]:
    """Require a reward change to move the fitted action probability."""
    panel = make_panel(seed=88_001, n_observations=1_000)
    model, transitions, tasks = model_spec(se_method="asymptotic")
    model.fit(panel, transitions=transitions, tasks=tasks)
    result = model.counterfactual(
        params={"action_one": model.params_["action_one"] + 1.0},
        description="increase the reward of action one",
    )
    effect = float(np.max(np.abs(result.policy_change)))
    passed = True
    error = None
    try:
        assert_effect(effect, min_abs=0.1)
    except AssertionError as exc:
        passed = False
        error = str(exc)
    return {"max_policy_change": effect, "passed": passed, "error": error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--n-reps", type=int, default=FULL_REPLICATIONS)
    parser.add_argument("--n-bootstrap", type=int, default=FULL_BOOTSTRAPS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    n_reps = min(args.n_reps, 8) if args.smoke else args.n_reps
    n_bootstrap = min(args.n_bootstrap, 10) if args.smoke else args.n_bootstrap
    output = args.output or (
        Path("/tmp/econirl_mce_irl_ready_smoke.json") if args.smoke else DEFAULT_OUTPUT
    )
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    records = run_replications(checkpoint, n_reps, quiet=args.quiet)
    repeated = coverage_summary(records)
    bootstrap = bootstrap_check(n_bootstrap)
    intervention = intervention_check()
    final_run = n_reps >= FULL_REPLICATIONS and n_bootstrap >= FULL_BOOTSTRAPS
    gates = [
        {
            "name": "all_fits_converged",
            "value": repeated["convergence_rate"],
            "operator": ">=",
            "threshold": THRESHOLDS["usable_rate_min"],
            "passed": repeated["convergence_rate"] >= THRESHOLDS["usable_rate_min"],
        },
        {
            "name": "coverage_selfcheck",
            "value": repeated["coverage"],
            "operator": "between",
            "threshold": [THRESHOLDS["coverage_low"], THRESHOLDS["coverage_high"]],
            "passed": repeated["coverage_selfcheck_passed"],
        },
        {
            "name": "absolute_bias",
            "value": abs(repeated["bias"]),
            "operator": "<=",
            "threshold": THRESHOLDS["absolute_bias_max"],
            "passed": abs(repeated["bias"]) <= THRESHOLDS["absolute_bias_max"],
        },
        {
            "name": "rmse",
            "value": repeated["rmse"],
            "operator": "<=",
            "threshold": THRESHOLDS["rmse_max"],
            "passed": repeated["rmse"] <= THRESHOLDS["rmse_max"],
        },
        {
            "name": "stationarity",
            "value": repeated["max_stationarity_residual"],
            "operator": "<=",
            "threshold": THRESHOLDS["stationarity_max"],
            "passed": repeated["max_stationarity_residual"] <= THRESHOLDS["stationarity_max"],
        },
        {
            "name": "asymptotic_bootstrap_se_ratio",
            "value": bootstrap["ratio"],
            "operator": "within",
            "threshold": THRESHOLDS["se_ratio_tolerance"],
            "passed": bootstrap["passed"],
        },
        {
            "name": "counterfactual_effect",
            "value": intervention["max_policy_change"],
            "operator": ">=",
            "threshold": THRESHOLDS["counterfactual_effect_min"],
            "passed": intervention["passed"],
        },
    ]
    status = "ready" if final_run and all(gate["passed"] for gate in gates) else "incomplete"
    payload = {
        "estimator": "MCE-IRL",
        "status": status,
        "truth": TRUE_THETA,
        "n_observations_per_fit": N_OBSERVATIONS,
        "n_replications": n_reps,
        "checkpoint": (
            str(checkpoint.relative_to(ROOT))
            if checkpoint.is_relative_to(ROOT)
            else str(checkpoint)
        ),
        "repeated_run_inference": repeated,
        "standard_error_check": bootstrap,
        "intervention_check": intervention,
        "gates": gates,
        "thresholds": THRESHOLDS,
        "provenance": {
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    display_output = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(f"wrote {display_output}")
    print(f"status: {status}")
    if final_run and status != "ready":
        print("failed gates: " + ", ".join(gate["name"] for gate in gates if not gate["passed"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
