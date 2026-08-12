#!/usr/bin/env python3
"""Run EconIRL on all official Table E.1 simulation panels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from econirl.core.types import DDCProblem, Panel, Trajectory  # noqa: E402
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator  # noqa: E402
from validation.estimators.tdccp.paper_table_e1 import (  # noqa: E402
    N_STATES,
    _reward,
    _state_features,
)

PUBLISHED = {
    "nonrobust": {
        "mean": np.array([1.97858874754578, -0.149203001688111, 1.00444781264231]),
        "empirical_sd": np.array([0.0868797349657598, 0.00334166404087212, 0.0583148856423492]),
    },
    "robust": {
        "mean": np.array([1.97751250141684, -0.148897244208649, 1.0037238962982]),
        "empirical_sd": np.array([0.0875936750784004, 0.00338730305981525, 0.0586844243236338]),
    },
}
TRUE_PARAMETERS = np.array([2.0, -0.15, 1.0])


def _problem() -> DDCProblem:
    encoded = _state_features()

    def state_encoder(states: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(encoded)[jnp.asarray(states, dtype=jnp.int32)]

    return DDCProblem(
        num_states=N_STATES,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
        state_dim=7,
        state_encoder=state_encoder,
    )


def _config(robust: bool) -> TDCCPConfig:
    return TDCCPConfig(
        method="semigradient",
        basis_type="encoded",
        basis_dim=1,
        basis_ridge=0.0,
        basis_action_coding="reference",
        ccp_method="logit",
        ccp_poly_degree=1,
        ccp_use_encoder=True,
        cross_fitting=robust,
        cross_fit_shuffle=False,
        cross_fit_ccp=False,
        robust_se=robust,
        linear_robust_correction="sensitivity",
        n_policy_iterations=1,
        outer_max_iter=1_000,
        outer_tol=1e-10,
        compute_se=False,
        compute_policy=False,
    )


def _panel(states: np.ndarray, actions: np.ndarray, folds: np.ndarray) -> Panel:
    order = np.argsort(folds, kind="stable")
    trajectories = []
    for panel_id, individual in enumerate(order):
        state = states[individual]
        action = actions[individual]
        next_state = np.empty_like(state)
        next_state[:-1] = state[1:]
        next_state[-1] = state[-1]
        trajectories.append(
            Trajectory(
                states=jnp.asarray(state, dtype=jnp.int32),
                actions=jnp.asarray(action, dtype=jnp.int32),
                next_states=jnp.asarray(next_state, dtype=jnp.int32),
                individual_id=panel_id,
            )
        )
    return Panel(trajectories)


def _fit(panel: Panel, problem: DDCProblem, *, robust: bool) -> dict[str, object]:
    started = time.perf_counter()
    estimator = TDCCPEstimator(
        config=_config(robust),
        se_method="asymptotic",
        seed=0,
    )
    result = estimator.estimate(
        panel=panel,
        utility=_reward(),
        problem=problem,
        transitions=jnp.ones((1, 1, 1)),
        initial_params=jnp.array([0.5, 0.5, 0.5]),
        transition_source="not used by the TD parameter stage",
    )
    record: dict[str, object] = {
        "estimates": np.asarray(result.parameters, dtype=float).tolist(),
        "converged": bool(result.converged),
        "fit_time_seconds": time.perf_counter() - started,
    }
    if robust:
        paper_inference = result.metadata.get("paper_inference") or {}
        standard_errors = np.asarray(
            paper_inference.get("standard_errors", np.full(3, np.nan)),
            dtype=float,
        )
        record.update(
            {
                "standard_errors": standard_errors.tolist(),
                "moment_norm_max": float(paper_inference.get("moment_norm_max", float("nan"))),
                "preliminary_optimizer_stationary": [
                    bool(value)
                    for value in paper_inference.get("preliminary_optimizer_stationary", [])
                ],
                "robust_optimizer_stationary": [
                    bool(value) for value in paper_inference.get("robust_optimizer_stationary", [])
                ],
            }
        )
    return record


def _load_checkpoint(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["k"])] = record
    return records


def _summary(records: dict[int, dict[str, object]], requested: int) -> dict[str, object]:
    modes: dict[str, object] = {}
    for mode in ("nonrobust", "robust"):
        estimates = np.asarray(
            [record[mode]["estimates"] for _, record in sorted(records.items())],
            dtype=float,
        )
        if estimates.size == 0:
            continue
        mean = estimates.mean(axis=0)
        empirical_sd = estimates.std(axis=0, ddof=1) if len(estimates) > 1 else np.full(3, np.nan)
        target = PUBLISHED[mode]
        complete = len(estimates) == requested
        modes[mode] = {
            "n_replications": len(estimates),
            "mean": mean.tolist(),
            "empirical_sd": empirical_sd.tolist(),
            "published_mean": target["mean"].tolist(),
            "published_empirical_sd": target["empirical_sd"].tolist(),
            "mean_absolute_error": np.abs(mean - target["mean"]).tolist(),
            "empirical_sd_absolute_error": np.abs(empirical_sd - target["empirical_sd"]).tolist(),
            "matches_four_significant_figures": bool(
                complete
                and np.allclose(mean, target["mean"], rtol=5e-5, atol=5e-7)
                and np.allclose(
                    empirical_sd,
                    target["empirical_sd"],
                    rtol=5e-5,
                    atol=5e-7,
                )
            ),
            "all_converged": all(bool(record[mode]["converged"]) for record in records.values()),
            "total_fit_time_seconds": float(
                sum(float(record[mode]["fit_time_seconds"]) for record in records.values())
            ),
        }
        if mode == "robust":
            standard_errors = np.asarray(
                [
                    record[mode].get("standard_errors", [np.nan] * 3)
                    for _, record in sorted(records.items())
                ],
                dtype=float,
            )
            usable = np.all(np.isfinite(estimates), axis=1) & np.all(
                np.isfinite(standard_errors) & (standard_errors > 0.0), axis=1
            )
            usable_estimates = estimates[usable]
            usable_se = standard_errors[usable]
            if len(usable_estimates) > 1:
                usable_mean = usable_estimates.mean(axis=0)
                usable_sd = usable_estimates.std(axis=0, ddof=1)
                mean_se = usable_se.mean(axis=0)
                lower = usable_estimates - 1.96 * usable_se
                upper = usable_estimates + 1.96 * usable_se
                coverage = ((lower <= TRUE_PARAMETERS) & (TRUE_PARAMETERS <= upper)).mean(axis=0)
                lower_tail_miss = (lower > TRUE_PARAMETERS).mean(axis=0)
                upper_tail_miss = (upper < TRUE_PARAMETERS).mean(axis=0)
                standardized_bias = np.abs(usable_mean - TRUE_PARAMETERS) / usable_sd
                mean_se_to_empirical_sd = mean_se / usable_sd
            else:
                coverage = np.full(3, np.nan)
                lower_tail_miss = np.full(3, np.nan)
                upper_tail_miss = np.full(3, np.nan)
                standardized_bias = np.full(3, np.nan)
                mean_se_to_empirical_sd = np.full(3, np.nan)
                mean_se = np.full(3, np.nan)
            inference_passed = bool(
                complete
                and usable.mean() >= 0.99
                and np.all(standardized_bias <= 0.20)
                and np.all((mean_se_to_empirical_sd >= 0.80) & (mean_se_to_empirical_sd <= 1.20))
                and np.all((coverage >= 0.91) & (coverage <= 0.99))
                and np.all((lower_tail_miss >= 0.01) & (lower_tail_miss <= 0.04))
                and np.all((upper_tail_miss >= 0.01) & (upper_tail_miss <= 0.04))
            )
            modes[mode]["inference"] = {
                "true_parameters": TRUE_PARAMETERS.tolist(),
                "usable_replications": int(usable.sum()),
                "usable_rate": float(usable.mean()),
                "mean_standard_error": mean_se.tolist(),
                "standardized_bias": standardized_bias.tolist(),
                "mean_se_to_empirical_sd": mean_se_to_empirical_sd.tolist(),
                "coverage_95": coverage.tolist(),
                "lower_tail_miss_rate": lower_tail_miss.tolist(),
                "upper_tail_miss_rate": upper_tail_miss.tolist(),
                "passed": inference_passed,
            }
    exact = bool(
        len(records) == requested
        and all(bool(value["matches_four_significant_figures"]) for value in modes.values())
    )
    return {
        "schema_version": 1,
        "claim": "official_table_e1_exact_replication" if exact else "exact_replication_attempt",
        "requested_replications": requested,
        "completed_replications": len(records),
        "parameter_names": ["theta_0", "theta_1", "theta_2"],
        "modes": modes,
        "exact_replication_passed": exact,
        "inference_calibration_passed": bool(
            modes.get("robust", {}).get("inference", {}).get("passed", False)
        ),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()

    records = _load_checkpoint(args.checkpoint)
    problem = _problem()
    with args.batch.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=3)
        if len(header) != 3:
            raise ValueError("batch file has an incomplete header")
        total, n_individuals, n_periods = map(int, header)
        start = max(args.start_index, 0)
        stop = total if args.count is None else min(total, start + args.count)
        requested = stop - start
        for replication in range(total):
            k_raw = np.fromfile(stream, dtype="<i4", count=1)
            folds = np.fromfile(stream, dtype="<i4", count=n_individuals)
            states = np.fromfile(stream, dtype="<i4", count=n_individuals * n_periods).reshape(
                n_individuals, n_periods
            )
            actions = np.fromfile(stream, dtype="<i4", count=n_individuals * n_periods).reshape(
                n_individuals, n_periods
            )
            if len(k_raw) != 1 or states.size != n_individuals * n_periods:
                raise ValueError(f"batch file ended during replication {replication + 1}")
            k = int(k_raw[0])
            if replication >= stop:
                break
            if replication < start:
                continue
            if k in records:
                continue
            panel = _panel(states, actions, folds)
            record = {
                "k": k,
                "nonrobust": _fit(panel, problem, robust=False),
                "robust": _fit(panel, problem, robust=True),
            }
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with args.checkpoint.open("a", encoding="utf-8") as checkpoint:
                checkpoint.write(json.dumps(record, sort_keys=True) + "\n")
            records[k] = record
            payload = _summary(records, requested)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"completed {len(records)}/{requested}: k={k}, "
                f"nonrobust={record['nonrobust']['fit_time_seconds']:.2f}s, "
                f"robust={record['robust']['fit_time_seconds']:.2f}s",
                flush=True,
            )

    payload = _summary(records, requested)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
