#!/usr/bin/env python3
"""Grade the public TD-CCP workflow on repeated encoded-state panels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from econirl import TDCCP  # noqa: E402
from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.reward_spec import RewardSpec  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402
from validation.benchmark.metrics import policy_tv  # noqa: E402
from validation.estimators.nfxp.ready import (  # noqa: E402
    _out_of_sample_scores,
    _regret,
    _strict_json,
)
from validation.estimators.tdccp.run import (  # noqa: E402
    build_paper_hard_case_dgp,
    skip_action_transitions,
)

DEFAULT_OUTPUT = ROOT / "validation" / "results" / "tdccp_ready.json"
FINAL_REPLICATIONS = 20
N_INDIVIDUALS = 2_000
N_PERIODS = 60
BASE_SEED = 91_000


def _oracle(env: Any, reward: np.ndarray, transitions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    solved = value_iteration(
        SoftBellmanOperator(env.problem_spec, jnp.asarray(transitions)),
        jnp.asarray(reward),
        tol=1e-10,
        max_iter=10_000,
    )
    return np.asarray(solved.policy, dtype=float), np.asarray(solved.V, dtype=float)


def _fit_one(rep: int, *, n_individuals: int, n_periods: int) -> dict[str, Any]:
    dgp = build_paper_hard_case_dgp()
    env = dgp["env"]
    utility = dgp["utility"]
    truth = np.asarray(dgp["true_params"], dtype=float)
    true_reward = np.asarray(dgp["true_reward"], dtype=float)
    transitions = np.asarray(env.transition_matrices, dtype=float)
    oracle_policy, oracle_value = _oracle(env, true_reward, transitions)
    seed = BASE_SEED + rep
    panel = simulate_panel_from_policy(
        env.problem_spec,
        env.transition_matrices,
        jnp.asarray(oracle_policy),
        jnp.asarray(env._get_initial_state_distribution()),
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )
    reward = RewardSpec(
        np.asarray(utility.feature_matrix, dtype=float),
        names=list(utility.parameter_names),
    )
    state_features = np.asarray(
        env.encode_states(jnp.arange(env.num_states, dtype=jnp.int32)),
        dtype=float,
    )
    started = time.perf_counter()
    try:
        model = TDCCP(
            n_states=env.num_states,
            n_actions=env.num_actions,
            discount=float(env.problem_spec.discount_factor),
            utility=reward,
            se_method="robust",
            seed=seed,
            method="semigradient",
            basis_type="encoded",
            basis_dim=2,
            basis_ridge=1e-7,
            ccp_method="logit",
            ccp_poly_degree=2,
            cross_fitting=True,
            robust_se=True,
            outer_max_iter=500,
            outer_tol=1e-7,
            state_features=state_features,
        )
        model.fit(panel, transitions=transitions)
        runtime = time.perf_counter() - started
        estimate = np.asarray(model.coef_, dtype=float)

        heldout = simulate_panel_from_policy(
            env.problem_spec,
            env.transition_matrices,
            jnp.asarray(oracle_policy),
            jnp.asarray(env._get_initial_state_distribution()),
            n_individuals=100,
            n_periods=n_periods,
            seed=BASE_SEED + 100_000 + rep,
        )
        reward_truth = truth.copy()
        reward_truth[0] += 1.0
        reward_matrix = np.asarray(utility.compute(jnp.asarray(reward_truth)), dtype=float)
        reward_policy, reward_value = _oracle(env, reward_matrix, transitions)
        reward_cf = model.counterfactual(
            **{utility.parameter_names[0]: float(model.coef_[0] + 1.0)}
        )

        changed_transitions = np.asarray(
            skip_action_transitions(env, action=0, skip=2), dtype=float
        )
        transition_policy, transition_value = _oracle(env, true_reward, changed_transitions)
        transition_cf = model.counterfactual(transitions=changed_transitions)
        return {
            "rep": rep,
            "seed": seed,
            "converged": bool(model.converged_),
            "runtime_seconds": runtime,
            "parameters": estimate.tolist(),
            "standard_errors": [float(model.se_[name]) for name in utility.parameter_names],
            "relative_parameter_error": (
                np.abs(estimate - truth) / np.maximum(np.abs(truth), 1e-12)
            ).tolist(),
            "policy_tv": policy_tv(model.policy_, oracle_policy),
            "out_of_sample": _out_of_sample_scores(heldout, model.policy_, oracle_policy),
            "counterfactuals": {
                "reward": {
                    "oracle_effect_policy_tv": policy_tv(reward_policy, oracle_policy),
                    "policy_tv": policy_tv(reward_cf.policy, reward_policy),
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
                    "oracle_effect_policy_tv": policy_tv(transition_policy, oracle_policy),
                    "policy_tv": policy_tv(transition_cf.policy, transition_policy),
                    "regret": _regret(
                        env=env,
                        estimated_policy=transition_cf.policy,
                        oracle_policy=transition_policy,
                        oracle_value=transition_value,
                        reward=true_reward,
                        transitions=changed_transitions,
                    ),
                },
            },
            "summary": model.summary() if rep == 0 else None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are readiness evidence
        return {
            "rep": rep,
            "seed": seed,
            "converged": False,
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["rep"])] = record
    return records


def _merge_checkpoints(paths: list[Path]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in paths:
        for rep, record in _read_checkpoint(path).items():
            if rep in records and records[rep] != record:
                raise ValueError(f"conflicting readiness record for rep={rep}")
            records[rep] = record
    return records


def _gate(name: str, value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _summarize(records: list[dict[str, Any]], *, final_run: bool) -> dict[str, Any]:
    usable = [
        record
        for record in records
        if record.get("error") is None
        and record.get("converged")
        and np.all(np.isfinite(record["parameters"]))
    ]
    truth = np.asarray(build_paper_hard_case_dgp()["true_params"], dtype=float)
    estimates = np.asarray([record["parameters"] for record in usable], dtype=float)
    relative = np.linalg.norm(estimates - truth[None, :], axis=1) / np.linalg.norm(truth)
    policy_errors = np.asarray([record["policy_tv"] for record in usable])
    runtimes = np.asarray([record["runtime_seconds"] for record in usable])
    nll = np.asarray(
        [record["out_of_sample"]["excess_negative_log_likelihood"] for record in usable]
    )
    brier = np.asarray([record["out_of_sample"]["excess_brier_score"] for record in usable])
    counterfactuals = {
        kind: {
            field: float(np.mean([record["counterfactuals"][kind][field] for record in usable]))
            for field in ("oracle_effect_policy_tv", "policy_tv", "regret")
        }
        for kind in ("reward", "transition")
    }
    usable_rate = len(usable) / max(len(records), 1)
    median_relative = float(np.median(relative)) if len(usable) else float("inf")
    p90_relative = float(np.quantile(relative, 0.9)) if len(usable) else float("inf")
    mean_policy_tv = float(np.mean(policy_errors)) if len(usable) else float("inf")
    max_runtime = float(np.max(runtimes)) if len(usable) else float("inf")
    mean_nll = float(np.mean(nll)) if len(usable) else float("inf")
    mean_brier = float(np.mean(brier)) if len(usable) else float("inf")
    gates = [
        _gate(
            "replications",
            len(records),
            ">=",
            FINAL_REPLICATIONS,
            len(records) >= FINAL_REPLICATIONS,
        ),
        _gate("usable_rate", usable_rate, ">=", 0.95, usable_rate >= 0.95),
        _gate(
            "median_relative_parameter_error",
            median_relative,
            "<=",
            0.10,
            median_relative <= 0.10,
        ),
        _gate(
            "p90_relative_parameter_error",
            p90_relative,
            "<=",
            0.25,
            p90_relative <= 0.25,
        ),
        _gate("mean_policy_tv", mean_policy_tv, "<=", 0.03, mean_policy_tv <= 0.03),
        _gate("max_runtime_seconds", max_runtime, "<=", 60.0, max_runtime <= 60.0),
        _gate("mean_excess_nll", mean_nll, "<=", 0.02, mean_nll <= 0.02),
        _gate("mean_excess_brier", mean_brier, "<=", 0.02, mean_brier <= 0.02),
    ]
    for kind, values in counterfactuals.items():
        gates.extend(
            [
                _gate(
                    f"{kind}_oracle_effect_policy_tv",
                    values["oracle_effect_policy_tv"],
                    ">=",
                    0.01,
                    values["oracle_effect_policy_tv"] >= 0.01,
                ),
                _gate(
                    f"{kind}_policy_tv",
                    values["policy_tv"],
                    "<=",
                    0.03,
                    values["policy_tv"] <= 0.03,
                ),
                _gate(
                    f"{kind}_regret",
                    values["regret"],
                    "<=",
                    0.05,
                    values["regret"] <= 0.05,
                ),
            ]
        )
    passed = bool(final_run and all(gate["passed"] for gate in gates))
    return {
        "status": "ready" if passed else "smoke_only" if not final_run else "not_ready",
        "completed_replications": len(records),
        "usable_replications": len(usable),
        "usable_rate": usable_rate,
        "median_relative_parameter_error": median_relative,
        "p90_relative_parameter_error": p90_relative,
        "mean_policy_tv": mean_policy_tv,
        "max_runtime_seconds": max_runtime,
        "mean_excess_negative_log_likelihood": mean_nll,
        "mean_excess_brier_score": mean_brier,
        "counterfactuals": counterfactuals,
        "gates": gates,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=FINAL_REPLICATIONS)
    parser.add_argument("--n-individuals", type=int, default=N_INDIVIDUALS)
    parser.add_argument("--n-periods", type=int, default=N_PERIODS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.merge:
        record_map = _merge_checkpoints(args.merge)
        records = [record_map[index] for index in sorted(record_map)]
        summary = _summarize(records, final_run=len(records) >= FINAL_REPLICATIONS)
        payload = {
            "estimator": "TD-CCP",
            "design": {
                "problem": "encoded-state three-action structural DDC",
                "n_replications": len(records),
                "n_individuals": N_INDIVIDUALS,
                "n_periods": N_PERIODS,
                "base_seed": BASE_SEED,
                "method": "semigradient with Algorithm 2 locally robust inference",
            },
            "summary": summary,
            "records": records,
            "provenance": {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
        return 0 if summary["passed"] else 1
    n_reps = min(args.n_reps, 2) if args.smoke else args.n_reps
    n_individuals = min(args.n_individuals, 300) if args.smoke else args.n_individuals
    n_periods = min(args.n_periods, 30) if args.smoke else args.n_periods
    final_run = (
        n_reps >= FINAL_REPLICATIONS and n_individuals >= N_INDIVIDUALS and n_periods >= N_PERIODS
    )
    output = Path("/tmp/econirl_tdccp_ready_smoke.json") if args.smoke else args.output
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    done = _read_checkpoint(checkpoint)
    records = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(args.start_rep, args.start_rep + n_reps):
        record = done.get(rep)
        if record is None:
            record = _fit_one(rep, n_individuals=n_individuals, n_periods=n_periods)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        status = record.get("error") or f"{record['runtime_seconds']:.2f}s"
        print(
            f"ready {rep + 1}/{n_reps}: {status}",
            flush=True,
        )
    summary = _summarize(records, final_run=final_run)
    payload = {
        "estimator": "TD-CCP",
        "design": {
            "problem": "encoded-state three-action structural DDC",
            "n_replications": n_reps,
            "n_individuals": n_individuals,
            "n_periods": n_periods,
            "base_seed": BASE_SEED,
            "method": "semigradient with Algorithm 2 locally robust inference",
        },
        "summary": summary,
        "records": records,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0 if not final_run or summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
