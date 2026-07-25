#!/usr/bin/env python3
"""NFXP implementation-readiness validation.

This runner uses package-owned known-truth problems. It does not use a paper
target. The two proof layers are:

1. repeated-sample inference on an identified 40-state problem;
2. recovery and counterfactual transfer on a sparse 200-state problem.

Every completed fit is checkpointed to JSONL. The compact JSON result is a pure
aggregate of those records and contains every threshold used for sign-off.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "nfxp_ready.json"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.reward_spec import RewardSpec  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.environments import ArrayMDP  # noqa: E402
from econirl.estimators import NFXP  # noqa: E402
from econirl.evaluation.selfcheck import assert_effect, check_se_ratio  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402
from validation.benchmark.metrics import policy_tv, value_rmse  # noqa: E402
from validation.known_truth import counterfactual_metrics  # noqa: E402

Z95 = 1.959963984540054


@dataclass(frozen=True)
class ProblemConfig:
    """Frozen inputs for one NFXP readiness problem."""

    name: str
    num_states: int
    n_individuals: int
    n_periods: int
    n_replications: int
    base_seed: int
    theta: tuple[float, float, float]
    fit_timeout_seconds: float


MC_CONFIG = ProblemConfig(
    name="inference_40_state",
    num_states=40,
    n_individuals=250,
    n_periods=40,
    n_replications=1000,
    base_seed=21000,
    theta=(0.35, -0.25, 0.20),
    fit_timeout_seconds=60.0,
)

HARD_CONFIG = ProblemConfig(
    name="sparse_200_state",
    num_states=200,
    n_individuals=300,
    n_periods=25,
    n_replications=20,
    base_seed=31000,
    theta=(1.0, -0.8, 0.6),
    fit_timeout_seconds=60.0,
)


def build_problem(config: ProblemConfig) -> ArrayMDP:
    """Build the frozen, identified action-dependent DGP."""
    states = config.num_states
    transitions = np.zeros((2, states, states), dtype=np.float64)
    drift = np.array([0.60, 0.30, 0.10])
    for state in range(states):
        for increment, probability in enumerate(drift):
            transitions[0, state, min(state + increment, states - 1)] += probability
    transitions[1, :, 0] = 1.0

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(states, 3))
    raw -= raw.mean(axis=0)
    orthogonal, _ = np.linalg.qr(raw)
    contrasts = orthogonal / orthogonal.std(axis=0, keepdims=True)
    features = np.zeros((states, 2, 3), dtype=np.float64)
    features[:, 1, :] = contrasts

    initial = np.ones(states, dtype=np.float64) / states
    return ArrayMDP(
        transitions,
        features,
        theta=np.asarray(config.theta),
        discount_factor=0.95,
        scale_parameter=1.0,
        parameter_names=["theta_0", "theta_1", "theta_2"],
        initial_distribution=initial,
        seed=config.base_seed,
    )


def slower_deterioration(transitions: np.ndarray) -> np.ndarray:
    """Return a frozen transition intervention with less upward drift."""
    changed = np.asarray(transitions, dtype=np.float64).copy()
    states = changed.shape[1]
    changed[0] = 0.0
    drift = np.array([0.80, 0.15, 0.05])
    for state in range(states):
        for increment, probability in enumerate(drift):
            changed[0, state, min(state + increment, states - 1)] += probability
    return changed


def _oracle(
    env: ArrayMDP,
    theta: np.ndarray,
    transitions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reward = np.einsum(
        "sak,k->sa",
        np.asarray(env.feature_matrix, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
    )
    solved = value_iteration(
        SoftBellmanOperator(env.problem_spec, jnp.asarray(transitions)),
        jnp.asarray(reward),
    )
    return (
        np.asarray(solved.policy, dtype=np.float64),
        np.asarray(solved.V, dtype=np.float64),
        reward,
    )


def _regret(
    *,
    env: ArrayMDP,
    estimated_policy: np.ndarray,
    oracle_policy: np.ndarray,
    oracle_value: np.ndarray,
    reward: np.ndarray,
    transitions: np.ndarray,
) -> float:
    metrics = counterfactual_metrics(
        oracle_policy=jnp.asarray(oracle_policy),
        oracle_value=jnp.asarray(oracle_value),
        estimated_policy=jnp.asarray(estimated_policy),
        reward=jnp.asarray(reward),
        transitions=jnp.asarray(transitions),
        discount_factor=float(env.problem_spec.discount_factor),
        initial_distribution=jnp.asarray(env._get_initial_state_distribution()),
        scale_parameter=float(env.problem_spec.scale_parameter),
    )
    return float(metrics.regret)


def validate_intervention_effects(env: ArrayMDP) -> dict[str, float]:
    """Refuse to run if either counterfactual is vacuous under truth."""
    true_theta = np.asarray(list(env.true_parameters.values()), dtype=np.float64)
    transitions = np.asarray(env.transition_matrices, dtype=np.float64)
    baseline_policy, _baseline_value, _baseline_reward = _oracle(
        env, true_theta, transitions
    )

    reward_theta = true_theta.copy()
    reward_theta[0] += 1.0
    reward_policy, _reward_value, _reward = _oracle(
        env, reward_theta, transitions
    )
    transition_cf = slower_deterioration(transitions)
    transition_policy, _transition_value, _transition_reward = _oracle(
        env, true_theta, transition_cf
    )

    reward_effect = policy_tv(reward_policy, baseline_policy)
    transition_effect = policy_tv(transition_policy, baseline_policy)
    assert_effect(reward_effect, min_abs=0.01)
    assert_effect(transition_effect, min_abs=0.01)
    return {
        "reward_policy_tv": reward_effect,
        "transition_policy_tv": transition_effect,
    }


def fit_once(env: ArrayMDP, config: ProblemConfig, rep: int) -> dict[str, Any]:
    """Fit the public NFXP wrapper and grade estimation plus counterfactuals."""
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
        model = NFXP(
            n_states=env.num_states,
            n_actions=env.num_actions,
            discount=float(env.problem_spec.discount_factor),
            utility=reward,
            se_method="robust",
        )
        model.fit(panel, transitions=transitions)
        runtime = time.perf_counter() - started
        if runtime > config.fit_timeout_seconds:
            raise TimeoutError(
                f"fit took {runtime:.2f}s, exceeding {config.fit_timeout_seconds:.0f}s"
            )

        estimate = np.asarray(model.coef_, dtype=np.float64)
        standard_errors = np.asarray(
            [model.se_[name] for name in names],
            dtype=np.float64,
        )
        oracle_policy, oracle_value, _true_reward = _oracle(
            env, true_theta, transitions
        )
        record = {
            "problem": config.name,
            "rep": rep,
            "seed": seed,
            "parameters": estimate.tolist(),
            "standard_errors": standard_errors.tolist(),
            "converged": bool(model.converged_),
            "runtime_seconds": runtime,
            "policy_tv": policy_tv(model.policy_, oracle_policy),
            "value_rmse": value_rmse(model.value_, oracle_value),
            "counterfactuals": None,
            "summary": model.summary() if rep == 0 else None,
            "error": None,
        }
        if config.name != HARD_CONFIG.name:
            return record

        reward_theta = true_theta.copy()
        reward_theta[0] += 1.0
        reward_oracle_policy, reward_oracle_value, reward_oracle_reward = _oracle(
            env, reward_theta, transitions
        )
        reward_cf = model.counterfactual(
            theta_0=float(model.params_["theta_0"] + 1.0)
        )

        transition_cf_tensor = slower_deterioration(transitions)
        transition_oracle_policy, transition_oracle_value, transition_oracle_reward = (
            _oracle(env, true_theta, transition_cf_tensor)
        )
        transition_cf = model.counterfactual(transitions=transition_cf_tensor)
        record["counterfactuals"] = {
            "reward": {
                "policy_tv": policy_tv(
                    reward_cf.policy,
                    reward_oracle_policy,
                ),
                "value_rmse": value_rmse(
                    reward_cf.value_function,
                    reward_oracle_value,
                ),
                "regret": _regret(
                    env=env,
                    estimated_policy=reward_cf.policy,
                    oracle_policy=reward_oracle_policy,
                    oracle_value=reward_oracle_value,
                    reward=reward_oracle_reward,
                    transitions=transitions,
                ),
            },
            "transition": {
                "policy_tv": policy_tv(
                    transition_cf.policy,
                    transition_oracle_policy,
                ),
                "value_rmse": value_rmse(
                    transition_cf.value_function,
                    transition_oracle_value,
                ),
                "regret": _regret(
                    env=env,
                    estimated_policy=transition_cf.policy,
                    oracle_policy=transition_oracle_policy,
                    oracle_value=transition_oracle_value,
                    reward=transition_oracle_reward,
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


def _checkpoint_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
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
    verbose: bool,
) -> list[dict[str, Any]]:
    """Run or resume all replications for one problem."""
    env = build_problem(config)
    validate_intervention_effects(env)
    done = _checkpoint_records(checkpoint)
    records: list[dict[str, Any]] = []

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(config.n_replications):
        key = (config.name, rep)
        if key in done:
            records.append(done[key])
            continue
        record = fit_once(env, config, rep)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        if verbose:
            status = record["error"] or (
                f"converged={record['converged']} "
                f"tv={record['policy_tv']:.4f}"
            )
            print(
                f"[{config.name}] {rep + 1}/{config.n_replications} "
                f"{record['runtime_seconds']:.2f}s {status}",
                flush=True,
            )
    return records


def inference_summary(
    records: list[dict[str, Any]],
    true_theta: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    """Aggregate repeated-sample inference, including both tail errors."""
    usable = [
        record
        for record in records
        if record.get("error") is None
        and record.get("converged") is True
        and np.isfinite(record["parameters"]).all()
        and np.isfinite(record["standard_errors"]).all()
    ]
    estimates = np.asarray([record["parameters"] for record in usable], dtype=np.float64)
    standard_errors = np.asarray(
        [record["standard_errors"] for record in usable],
        dtype=np.float64,
    )
    if estimates.size == 0:
        return {
            "n_total": len(records),
            "n_usable": 0,
            "usable_rate": 0.0,
            "parameters": {},
        }

    empirical_sd = (
        estimates.std(axis=0, ddof=1)
        if estimates.shape[0] > 1
        else np.full(estimates.shape[1], np.nan)
    )
    mean_estimate = estimates.mean(axis=0)
    mean_reported_se = standard_errors.mean(axis=0)
    lower = estimates - Z95 * standard_errors
    upper = estimates + Z95 * standard_errors

    parameters: dict[str, Any] = {}
    for index, name in enumerate(names):
        left_miss = float(np.mean(true_theta[index] < lower[:, index]))
        right_miss = float(np.mean(true_theta[index] > upper[:, index]))
        bias = float(mean_estimate[index] - true_theta[index])
        parameters[name] = {
            "true": float(true_theta[index]),
            "mean_estimate": float(mean_estimate[index]),
            "bias": bias,
            "empirical_sd": float(empirical_sd[index]),
            "mean_reported_se": float(mean_reported_se[index]),
            "se_ratio": float(mean_reported_se[index] / empirical_sd[index]),
            "standardized_bias": float(abs(bias) / empirical_sd[index]),
            "coverage_95": float(1.0 - left_miss - right_miss),
            "left_miss": left_miss,
            "right_miss": right_miss,
        }

    return {
        "n_total": len(records),
        "n_usable": len(usable),
        "usable_rate": len(usable) / len(records),
        "parameters": parameters,
    }


def hard_problem_summary(
    records: list[dict[str, Any]],
    true_theta: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    """Aggregate recovery, counterfactual, and runtime metrics."""
    usable = [
        record
        for record in records
        if record.get("error") is None and record.get("converged") is True
    ]
    if not usable:
        return {
            "n_total": len(records),
            "n_usable": 0,
            "usable_rate": 0.0,
        }

    estimates = np.asarray([record["parameters"] for record in usable])
    relative_errors = np.abs(estimates - true_theta) / np.abs(true_theta)
    summary: dict[str, Any] = {
        "n_total": len(records),
        "n_usable": len(usable),
        "usable_rate": len(usable) / len(records),
        "median_relative_error": {
            name: float(np.median(relative_errors[:, index]))
            for index, name in enumerate(names)
        },
        "p90_relative_error": {
            name: float(np.percentile(relative_errors[:, index], 90))
            for index, name in enumerate(names)
        },
        "policy_tv_mean": float(np.mean([record["policy_tv"] for record in usable])),
        "runtime_seconds_max": float(
            np.max([record["runtime_seconds"] for record in usable])
        ),
        "counterfactuals": {},
    }
    for kind in ("reward", "transition"):
        summary["counterfactuals"][kind] = {
            "policy_tv_mean": float(
                np.mean(
                    [
                        record["counterfactuals"][kind]["policy_tv"]
                        for record in usable
                    ]
                )
            ),
            "regret_mean": float(
                np.mean(
                    [
                        record["counterfactuals"][kind]["regret"]
                        for record in usable
                    ]
                )
            ),
        }
    return summary


def readiness_gates(
    inference: dict[str, Any],
    hard: dict[str, Any],
    *,
    final_run: bool,
) -> list[dict[str, Any]]:
    """Evaluate frozen thresholds without hiding individual failures."""
    gates: list[dict[str, Any]] = []

    def add(name: str, value: float, operator: str, threshold: float) -> None:
        if operator == ">=":
            passed = value >= threshold
        elif operator == "<=":
            passed = value <= threshold
        else:
            raise ValueError(f"unknown operator {operator}")
        gates.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    add("inference_usable_rate", inference["usable_rate"], ">=", 0.99)
    for name, result in inference.get("parameters", {}).items():
        add(f"{name}_standardized_bias", result["standardized_bias"], "<=", 0.20)
        add(f"{name}_se_ratio_low", result["se_ratio"], ">=", 0.80)
        add(f"{name}_se_ratio_high", result["se_ratio"], "<=", 1.20)
        add(f"{name}_coverage_low", result["coverage_95"], ">=", 0.91)
        add(f"{name}_coverage_high", result["coverage_95"], "<=", 0.99)
        add(f"{name}_left_tail_low", result["left_miss"], ">=", 0.01)
        add(f"{name}_left_tail_high", result["left_miss"], "<=", 0.04)
        add(f"{name}_right_tail_low", result["right_miss"], ">=", 0.01)
        add(f"{name}_right_tail_high", result["right_miss"], "<=", 0.04)

    add("hard_usable_rate", hard["usable_rate"], ">=", 0.95)
    if hard.get("n_usable", 0):
        for name, value in hard["median_relative_error"].items():
            add(f"hard_{name}_median_relative_error", value, "<=", 0.10)
        for name, value in hard["p90_relative_error"].items():
            add(f"hard_{name}_p90_relative_error", value, "<=", 0.25)
        add("hard_policy_tv_mean", hard["policy_tv_mean"], "<=", 0.03)
        add(
            "hard_runtime_seconds_max",
            hard["runtime_seconds_max"],
            "<=",
            HARD_CONFIG.fit_timeout_seconds,
        )
        for kind, values in hard["counterfactuals"].items():
            add(
                f"hard_{kind}_counterfactual_policy_tv",
                values["policy_tv_mean"],
                "<=",
                0.03,
            )
            add(
                f"hard_{kind}_counterfactual_regret",
                values["regret_mean"],
                "<=",
                0.05,
            )

    if not final_run:
        for gate in gates:
            gate["enforced"] = False
    else:
        for gate in gates:
            gate["enforced"] = True
    return gates


def alternate_se_checks(
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Exercise every advertised NFXP SE mode on one frozen panel."""
    config = ProblemConfig(
        **{
            **asdict(MC_CONFIG),
            "n_individuals": 80,
            "n_periods": 20,
            "n_replications": 1,
        }
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
        model = NFXP(
            n_states=env.num_states,
            discount=float(env.problem_spec.discount_factor),
            utility=reward,
            se_method=method,
            n_bootstrap=20 if smoke else 100,
            se_seed=9901,
        )
        model.fit(panel, transitions=transitions)
        outputs[method] = {
            "converged": bool(model.converged_),
            "standard_errors": [model.se_[name] for name in env.parameter_names],
        }

    ratios = []
    for clustered, bootstrap in zip(
        outputs["clustered"]["standard_errors"],
        outputs["bootstrap"]["standard_errors"],
        strict=True,
    ):
        ratios.append(check_se_ratio(clustered, bootstrap, tol=0.25))
    outputs["clustered_to_bootstrap_ratio"] = ratios

    full_model = NFXP(
        n_states=env.num_states,
        discount=float(env.problem_spec.discount_factor),
        utility=reward,
        se_method="full_likelihood_bhhh",
    )
    full_model.fit(panel)
    outputs["full_likelihood_bhhh"] = {
        "converged": bool(full_model.converged_),
        "standard_errors": [
            full_model.se_[name] for name in env.parameter_names
        ],
    }
    outputs["passed"] = all(
        result["converged"]
        and np.isfinite(result["standard_errors"]).all()
        and (np.asarray(result["standard_errors"]) > 0).all()
        for key, result in outputs.items()
        if key in (*methods, "full_likelihood_bhhh")
    )
    return outputs


def _strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--n-reps", type=int, default=MC_CONFIG.n_replications)
    parser.add_argument("--hard-reps", type=int, default=HARD_CONFIG.n_replications)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        n_reps = min(args.n_reps, 3)
        hard_reps = min(args.hard_reps, 2)
    else:
        n_reps = args.n_reps
        hard_reps = args.hard_reps

    mc_config = ProblemConfig(**{**asdict(MC_CONFIG), "n_replications": n_reps})
    hard_config = ProblemConfig(
        **{**asdict(HARD_CONFIG), "n_replications": hard_reps}
    )
    output = args.output or (
        Path("/tmp/econirl_nfxp_ready_smoke.json")
        if args.smoke
        else DEFAULT_OUTPUT
    )
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")

    mc_records = run_problem(
        mc_config,
        checkpoint=checkpoint,
        verbose=not args.quiet,
    )
    hard_records = run_problem(
        hard_config,
        checkpoint=checkpoint,
        verbose=not args.quiet,
    )

    mc_env = build_problem(mc_config)
    hard_env = build_problem(hard_config)
    names = list(mc_env.parameter_names)
    mc_truth = np.asarray([mc_env.true_parameters[name] for name in names])
    hard_truth = np.asarray([hard_env.true_parameters[name] for name in names])
    inference = inference_summary(mc_records, mc_truth, names)
    hard = hard_problem_summary(hard_records, hard_truth, names)
    alternate = alternate_se_checks(smoke=args.smoke)
    final_run = n_reps >= MC_CONFIG.n_replications and hard_reps >= HARD_CONFIG.n_replications
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

    # Render the report with the current implementation even when the numerical
    # records were resumed from an older checkpoint.
    summary_report = fit_once(hard_env, hard_config, 0)["summary"]
    payload = {
        "estimator": "NFXP",
        "status": (
            "ready"
            if final_run and all(gate["passed"] for gate in gates)
            else "smoke_only"
            if not final_run
            else "not_ready"
        ),
        "implementation_target": (
            "public summary, repeated-sample inference, and oracle-backed "
            "counterfactual recovery"
        ),
        "paper_target": None,
        "configs": {
            "inference": asdict(mc_config),
            "hard_problem": asdict(hard_config),
        },
        "intervention_effects": {
            "inference": validate_intervention_effects(mc_env),
            "hard_problem": validate_intervention_effects(hard_env),
        },
        "inference": inference,
        "hard_problem": hard,
        "alternate_standard_errors": alternate,
        "gates": gates,
        "summary_report": summary_report,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "pid": os.getpid(),
        },
        "checkpoint": str(checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    failed = [gate["name"] for gate in gates if gate["enforced"] and not gate["passed"]]
    if failed:
        print("failed gates:")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
