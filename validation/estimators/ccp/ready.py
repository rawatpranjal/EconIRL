#!/usr/bin/env python3
"""CCP implementation-readiness validation.

This runner uses package-owned structural problems rather than a paper target.
It checkpoints every fit and grades two layers:

1. one-step Hotz-Miller inference over 1,000 dense 20-state panels;
2. three-stage NPL recovery and counterfactuals on a 100-state problem.

The thresholds are the same frozen structural-estimator thresholds used for
NFXP. Smoke runs compute them without enforcing them.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "ccp_ready.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.reward_spec import RewardSpec  # noqa: E402
from econirl.estimators import CCP  # noqa: E402
from econirl.evaluation.selfcheck import check_se_ratio  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402
from validation.benchmark.metrics import policy_tv, value_rmse  # noqa: E402
from validation.estimators.nfxp.ready import (  # noqa: E402
    ProblemConfig,
    _oracle,
    _out_of_sample_scores,
    _regret,
    _strict_json,
    build_problem,
    hard_problem_summary,
    inference_summary,
    readiness_gates,
    slower_deterioration,
    validate_intervention_effects,
)

INFERENCE_CONFIG = ProblemConfig(
    name="inference_20_state",
    num_states=20,
    n_individuals=250,
    n_periods=40,
    n_replications=1000,
    base_seed=41000,
    theta=(0.35, -0.25, 0.20),
    fit_timeout_seconds=60.0,
)

HARD_CONFIG = ProblemConfig(
    name="npl_100_state",
    num_states=100,
    n_individuals=4000,
    n_periods=40,
    n_replications=20,
    base_seed=51000,
    theta=(0.35, -0.25, 0.20),
    fit_timeout_seconds=60.0,
)


def fit_once(
    config: ProblemConfig,
    rep: int,
    *,
    policy_iterations: int,
    include_counterfactuals: bool,
) -> dict[str, Any]:
    """Fit the public CCP wrapper and return one checkpoint record."""
    env = build_problem(config)
    seed = config.base_seed + rep
    panel = simulate_panel(
        env,
        n_individuals=config.n_individuals,
        n_periods=config.n_periods,
        seed=seed,
    )
    transitions = np.asarray(env.transition_matrices, dtype=np.float64)
    feature_matrix = np.asarray(env.feature_matrix, dtype=np.float64)
    names = list(env.parameter_names)
    true_theta = np.asarray([env.true_parameters[name] for name in names])
    reward = RewardSpec(feature_matrix, names=names)

    started = time.perf_counter()
    try:
        model = CCP(
            n_states=env.num_states,
            n_actions=env.num_actions,
            discount=float(env.problem_spec.discount_factor),
            utility=reward,
            se_method="robust",
            num_policy_iterations=policy_iterations,
        )
        model.fit(panel, transitions=transitions)
        runtime = time.perf_counter() - started
        if runtime > config.fit_timeout_seconds:
            raise TimeoutError(
                f"fit took {runtime:.2f}s, exceeding {config.fit_timeout_seconds:.0f}s"
            )

        support = model._result.metadata["ccp_support"]
        support_errors = list(support["warnings"])
        if support["state_action_coverage"] < 1.0:
            support_errors.append(
                f"state-action coverage is {support['state_action_coverage']:.6f}, expected 1.0"
            )
        if support_errors:
            raise RuntimeError("CCP support check failed: " + "; ".join(support_errors))

        estimate = np.asarray(model.coef_, dtype=np.float64)
        standard_errors = np.asarray(
            [model.se_[name] for name in names],
            dtype=np.float64,
        )
        oracle_policy, oracle_value, _true_reward = _oracle(
            env,
            true_theta,
            transitions,
        )
        record: dict[str, Any] = {
            "problem": config.name,
            "rep": rep,
            "seed": seed,
            "parameters": estimate.tolist(),
            "standard_errors": standard_errors.tolist(),
            "converged": bool(model.converged_),
            "npl_converged": bool(model.npl_converged_),
            "termination_reason": model.termination_reason_,
            "runtime_seconds": runtime,
            "policy_tv": policy_tv(model.policy_, oracle_policy),
            "value_rmse": value_rmse(model.value_, oracle_value),
            "support": support,
            "counterfactuals": None,
            "summary": model.summary() if rep == 0 else None,
            "error": None,
        }
        if not include_counterfactuals:
            return record

        heldout = simulate_panel(
            env,
            n_individuals=100,
            n_periods=25,
            seed=config.base_seed + 500_000 + rep,
        )
        record["out_of_sample"] = _out_of_sample_scores(
            heldout,
            model.policy_,
            oracle_policy,
        )

        reward_theta = true_theta.copy()
        reward_theta[0] += 1.0
        reward_policy, reward_value, reward_matrix = _oracle(
            env,
            reward_theta,
            transitions,
        )
        reward_cf = model.counterfactual(theta_0=float(model.params_["theta_0"] + 1.0))

        transition_cf_tensor = slower_deterioration(transitions)
        transition_policy, transition_value, transition_reward = _oracle(
            env,
            true_theta,
            transition_cf_tensor,
        )
        transition_cf = model.counterfactual(transitions=transition_cf_tensor)
        record["counterfactuals"] = {
            "reward": {
                "policy_tv": policy_tv(reward_cf.policy, reward_policy),
                "value_rmse": value_rmse(
                    reward_cf.value_function,
                    reward_value,
                ),
                "regret": _regret(
                    env=env,
                    estimated_policy=reward_cf.policy,
                    oracle_policy=reward_policy,
                    oracle_value=reward_value,
                    reward=reward_matrix,
                    transitions=transitions,
                ),
            },
            "transition": {
                "policy_tv": policy_tv(
                    transition_cf.policy,
                    transition_policy,
                ),
                "value_rmse": value_rmse(
                    transition_cf.value_function,
                    transition_value,
                ),
                "regret": _regret(
                    env=env,
                    estimated_policy=transition_cf.policy,
                    oracle_policy=transition_policy,
                    oracle_value=transition_value,
                    reward=transition_reward,
                    transitions=transition_cf_tensor,
                ),
            },
        }
        return record
    except Exception as exc:  # noqa: BLE001 - failures are validation data
        return {
            "problem": config.name,
            "rep": rep,
            "seed": seed,
            "runtime_seconds": time.perf_counter() - started,
            "converged": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def checkpoint_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read valid completed checkpoint rows."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[(record["problem"], int(record["rep"]))] = record
    return records


def run_problem(
    config: ProblemConfig,
    *,
    checkpoint: Path,
    policy_iterations: int,
    include_counterfactuals: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Run or resume one readiness problem."""
    validate_intervention_effects(build_problem(config))
    done = checkpoint_records(checkpoint)
    records: list[dict[str, Any]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(config.n_replications):
        key = (config.name, rep)
        if key in done:
            records.append(done[key])
            continue
        record = fit_once(
            config,
            rep,
            policy_iterations=policy_iterations,
            include_counterfactuals=include_counterfactuals,
        )
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_strict_json(record), allow_nan=False) + "\n")
        records.append(record)
        if verbose:
            outcome = record["error"] or (
                f"converged={record['converged']} "
                f"tv={record['policy_tv']:.4f} "
                f"time={record['runtime_seconds']:.2f}s"
            )
            print(f"{config.name} {rep + 1}/{config.n_replications}: {outcome}")
    return records


def alternate_se_checks(*, smoke: bool) -> dict[str, Any]:
    """Exercise every advertised CCP standard-error mode."""
    config = ProblemConfig(
        name="alternate_se_40_state",
        num_states=40,
        n_individuals=400,
        n_periods=20,
        n_replications=1,
        base_seed=61000,
        theta=INFERENCE_CONFIG.theta,
        fit_timeout_seconds=60.0,
    )
    env = build_problem(config)
    panel = simulate_panel(
        env,
        n_individuals=config.n_individuals,
        n_periods=config.n_periods,
        seed=config.base_seed,
    )
    transitions = np.asarray(env.transition_matrices)
    reward = RewardSpec(np.asarray(env.feature_matrix), names=env.parameter_names)
    outputs: dict[str, Any] = {}
    methods = ("asymptotic", "robust", "clustered", "bootstrap")
    for method in methods:
        model = CCP(
            n_states=env.num_states,
            n_actions=env.num_actions,
            discount=float(env.problem_spec.discount_factor),
            utility=reward,
            se_method=method,
            n_bootstrap=10 if smoke else 100,
            se_seed=62001,
            num_policy_iterations=1,
        )
        model.fit(panel, transitions=transitions)
        outputs[method] = {
            "converged": bool(model.converged_),
            "standard_errors": [model.se_[name] for name in env.parameter_names],
        }

    ratios = [
        clustered / bootstrap
        for clustered, bootstrap in zip(
            outputs["clustered"]["standard_errors"],
            outputs["bootstrap"]["standard_errors"],
            strict=True,
        )
    ]
    outputs["clustered_to_bootstrap_ratio"] = ratios
    ratio_passed = True
    if not smoke:
        try:
            for clustered, bootstrap in zip(
                outputs["clustered"]["standard_errors"],
                outputs["bootstrap"]["standard_errors"],
                strict=True,
            ):
                check_se_ratio(clustered, bootstrap, tol=0.25)
        except AssertionError:
            ratio_passed = False
    outputs["ratio_check_passed"] = ratio_passed
    outputs["passed"] = ratio_passed and all(
        result["converged"]
        and np.isfinite(result["standard_errors"]).all()
        and (np.asarray(result["standard_errors"]) > 0).all()
        for key, result in outputs.items()
        if key in methods
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--n-reps", type=int, default=INFERENCE_CONFIG.n_replications)
    parser.add_argument("--hard-reps", type=int, default=HARD_CONFIG.n_replications)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--alternate-se-results",
        type=Path,
        help="Reuse a completed alternate-standard-error JSON result.",
    )
    args = parser.parse_args()

    n_reps = min(args.n_reps, 3) if args.smoke else args.n_reps
    hard_reps = min(args.hard_reps, 2) if args.smoke else args.hard_reps
    inference_config = ProblemConfig(**{**asdict(INFERENCE_CONFIG), "n_replications": n_reps})
    hard_config = ProblemConfig(**{**asdict(HARD_CONFIG), "n_replications": hard_reps})
    output = args.output or (
        Path("/tmp/econirl_ccp_ready_smoke.json") if args.smoke else DEFAULT_OUTPUT
    )
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")

    inference_records = run_problem(
        inference_config,
        checkpoint=checkpoint,
        policy_iterations=1,
        include_counterfactuals=False,
        verbose=not args.quiet,
    )
    hard_records = run_problem(
        hard_config,
        checkpoint=checkpoint,
        policy_iterations=3,
        include_counterfactuals=True,
        verbose=not args.quiet,
    )

    inference_env = build_problem(inference_config)
    hard_env = build_problem(hard_config)
    names = list(inference_env.parameter_names)
    inference_truth = np.asarray([inference_env.true_parameters[name] for name in names])
    hard_truth = np.asarray([hard_env.true_parameters[name] for name in names])
    inference = inference_summary(inference_records, inference_truth, names)
    hard = hard_problem_summary(hard_records, hard_truth, names)
    alternate = (
        json.loads(args.alternate_se_results.read_text(encoding="utf-8"))
        if args.alternate_se_results is not None
        else alternate_se_checks(smoke=args.smoke)
    )
    final_run = (
        n_reps >= INFERENCE_CONFIG.n_replications and hard_reps >= HARD_CONFIG.n_replications
    )
    gates = readiness_gates(inference, hard, final_run=final_run)
    gates.append(
        {
            "name": "alternate_se_methods",
            "value": alternate["passed"],
            "operator": "is",
            "threshold": True,
            "passed": bool(alternate["passed"]),
            "enforced": final_run,
        }
    )

    first_summary = next(
        (record["summary"] for record in inference_records if record.get("summary") is not None),
        None,
    )
    support_example = next(
        (
            record["support"]
            for record in inference_records
            if record.get("error") is None and record.get("support") is not None
        ),
        None,
    )
    enforced = [gate for gate in gates if gate["enforced"]]
    status = "ready" if enforced and all(gate["passed"] for gate in enforced) else "incomplete"
    payload = {
        "estimator": "CCP",
        "paper_target": None,
        "status": status,
        "checkpoint": str(checkpoint),
        "configs": {
            "inference": asdict(inference_config),
            "hard_problem": asdict(hard_config),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "inference": inference,
        "hard_problem": hard,
        "alternate_standard_errors": alternate,
        "support_example": support_example,
        "gates": gates,
        "summary_report": first_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"status: {status}")
    if final_run and status != "ready":
        failed = [gate["name"] for gate in enforced if not gate["passed"]]
        print("failed gates: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
