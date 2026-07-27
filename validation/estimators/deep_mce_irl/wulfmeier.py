#!/usr/bin/env python3
"""Wulfmeier-shaped Objectworld and Binaryworld comparison study."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "deep_mce_irl_wulfmeier.json"
EXAMPLE = ROOT / "examples" / "wulfmeier-deep-maxent" / "replicate.py"
FULL_GRID = 32
FULL_DEMO_COUNTS = (8, 16, 32, 64, 128)
FULL_PANEL_SEEDS = 5
FULL_TRAINING_SEEDS = 3

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import hybrid_iteration  # noqa: E402
from econirl.core.types import DDCProblem  # noqa: E402
from econirl.environments.binaryworld import BinaryworldEnvironment  # noqa: E402
from econirl.environments.objectworld import ObjectworldEnvironment  # noqa: E402

spec = importlib.util.spec_from_file_location("wulfmeier_example", EXAMPLE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {EXAMPLE}")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def package_versions() -> dict[str, str]:
    names = ("econirl", "jax", "jaxlib", "equinox", "optax", "numpy")
    return {name: importlib.metadata.version(name) for name in names}


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def make_environment(name: str, grid_size: int, seed: int):
    if name == "objectworld":
        return ObjectworldEnvironment(
            grid_size=grid_size,
            n_colors=2,
            n_objects_per_color=3,
            discount_factor=0.9,
            feature_type="continuous",
            seed=seed,
        )
    return BinaryworldEnvironment(
        grid_size=grid_size,
        discount_factor=0.9,
        seed=seed,
    )


def solve_policy(reward: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    problem = DDCProblem(
        num_states=reward.shape[0],
        num_actions=reward.shape[1],
        discount_factor=0.9,
        scale_parameter=1.0,
    )
    result = hybrid_iteration(
        SoftBellmanOperator(problem, jnp.asarray(transitions)),
        jnp.asarray(reward),
        tol=1e-8,
        max_iter=5_000,
    )
    if not result.converged:
        raise RuntimeError("transfer Bellman solve did not converge")
    return np.asarray(result.policy)


def neural_transfer_policy(model, environment) -> np.ndarray:
    state_features = jnp.asarray(environment.feature_matrix)[:, 0, :]
    reward = model._compute_reward_matrix(  # noqa: SLF001 - validation probe
        model._reward_net,  # noqa: SLF001
        state_features,
        environment.num_states,
        environment.num_actions,
    )
    return solve_policy(np.asarray(reward), np.asarray(environment.transition_matrices))


def linear_transfer_policy(result, environment) -> np.ndarray:
    reward = np.einsum(
        "sak,k->sa",
        np.asarray(environment.feature_matrix),
        np.asarray(result.parameters),
    )
    return solve_policy(reward, np.asarray(environment.transition_matrices))


def load_checkpoint(path: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            key = (
                record["environment"],
                int(record["n_demos"]),
                int(record["panel_seed"]),
                int(record["training_seed"]),
            )
            records[key] = record
    return records


def fit_cell(
    environment_name: str,
    n_demos: int,
    panel_seed: int,
    training_seed: int,
    *,
    grid_size: int,
    max_epochs: int,
) -> dict[str, Any]:
    environment = make_environment(environment_name, grid_size, panel_seed)
    panel = environment.simulate_demonstrations(
        n_demos=n_demos,
        max_steps=50,
        noise_fraction=0.3,
        seed=10_000 + panel_seed,
    )
    transfer = make_environment(environment_name, grid_size, 50_000 + panel_seed)
    started = time.perf_counter()
    try:
        if training_seed >= 0:
            model = benchmark.run_estimator_mce_neural(
                environment,
                panel,
                grid_size,
                0.9,
                max_epochs=max_epochs,
                lr=0.01,
                training_seed=70_000 + training_seed,
                return_model=True,
            )
            policy = np.asarray(model.policy_)
            transfer_policy = neural_transfer_policy(model, transfer)
            learned_reward = np.asarray(model.reward_, dtype=float).tolist()
            method = "neural"
            converged = model.converged_ is True
            optimizer_converged = None
            feature_converged = None
            occupancy_converged = None
            bellman_converged = None
            termination_reason = model.termination_reason_
            n_iterations = model.n_epochs_
        else:
            result = benchmark.run_estimator_mce_linear(
                environment,
                panel,
                0.9,
                return_result=True,
            )
            policy = np.asarray(result.policy)
            transfer_policy = linear_transfer_policy(result, transfer)
            learned_reward = None
            method = "linear"
            converged = bool(result.converged)
            optimizer_converged = bool(result.metadata.get("optimizer_converged", False))
            feature_converged = bool(result.metadata.get("feature_converged", False))
            occupancy_converged = bool(result.metadata.get("occupancy_converged", False))
            bellman_converged = bool(result.metadata.get("bellman_converged", False))
            termination_reason = str(
                result.metadata.get("termination_reason", getattr(result, "message", ""))
            )
            n_iterations = int(result.num_iterations)
        evd = benchmark.compute_evd(
            environment.true_reward,
            environment.transition_matrices,
            policy,
            0.9,
        )
        transfer_evd = benchmark.compute_evd(
            transfer.true_reward,
            transfer.transition_matrices,
            transfer_policy,
            0.9,
        )
        return {
            "environment": environment_name,
            "method": method,
            "n_demos": n_demos,
            "panel_seed": panel_seed,
            "training_seed": training_seed,
            "evd": float(evd),
            "transfer_evd": float(transfer_evd),
            "true_reward": np.asarray(environment.true_reward, dtype=float).tolist(),
            "learned_reward": learned_reward,
            "runtime_seconds": time.perf_counter() - started,
            "converged": converged,
            "optimizer_converged": optimizer_converged,
            "feature_converged": feature_converged,
            "occupancy_converged": occupancy_converged,
            "bellman_converged": bellman_converged,
            "termination_reason": termination_reason,
            "n_iterations": n_iterations,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - failures are evidence
        return {
            "environment": environment_name,
            "method": "neural" if training_seed >= 0 else "linear",
            "n_demos": n_demos,
            "panel_seed": panel_seed,
            "training_seed": training_seed,
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_study(
    checkpoint: Path,
    *,
    grid_size: int,
    demo_counts: tuple[int, ...],
    n_panel_seeds: int,
    n_training_seeds: int,
    max_epochs: int,
    quiet: bool,
) -> list[dict[str, Any]]:
    completed = load_checkpoint(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for environment in ("objectworld", "binaryworld"):
        for n_demos in demo_counts:
            for panel_seed in range(n_panel_seeds):
                for training_seed in (-1, *range(n_training_seeds)):
                    key = (environment, n_demos, panel_seed, training_seed)
                    if key in completed:
                        records.append(completed[key])
                        continue
                    record = fit_cell(
                        environment,
                        n_demos,
                        panel_seed,
                        training_seed,
                        grid_size=grid_size,
                        max_epochs=max_epochs,
                    )
                    with checkpoint.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    records.append(record)
                    if not quiet:
                        print(
                            environment,
                            n_demos,
                            panel_seed,
                            training_seed,
                            record.get("error") or f"EVD={record['evd']:.4f}",
                        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("error") is None]

    def median(items: list[dict[str, Any]], key: str) -> float | None:
        values = [float(item[key]) for item in items]
        return float(np.median(values)) if values else None

    def percentile(items: list[dict[str, Any]], key: str, q: float) -> float | None:
        values = [float(item[key]) for item in items]
        return float(np.percentile(values, q)) if values else None

    cells: dict[str, Any] = {}
    for environment in ("objectworld", "binaryworld"):
        for n_demos in sorted({int(record["n_demos"]) for record in records}):
            subset = [
                record
                for record in successful
                if record["environment"] == environment and int(record["n_demos"]) == n_demos
            ]
            linear = [record for record in subset if record["method"] == "linear"]
            neural = [record for record in subset if record["method"] == "neural"]
            cells[f"{environment}:{n_demos}"] = {
                "neural_evd_median": median(neural, "evd"),
                "neural_evd_p95": percentile(neural, "evd", 95),
                "linear_evd_median": median(linear, "evd"),
                "neural_transfer_evd_median": median(neural, "transfer_evd"),
                "linear_transfer_evd_median": median(linear, "transfer_evd"),
                "linear_optimizer_converged": sum(
                    item.get("optimizer_converged") is True for item in linear
                ),
                "linear_solution_residuals_pass": sum(
                    item.get("occupancy_converged") is True
                    and item.get("bellman_converged") is True
                    for item in linear
                ),
                "linear_fits": len(linear),
            }
    selected: dict[str, Any] = {}
    for environment in ("objectworld", "binaryworld"):
        candidates = sorted(
            (
                record
                for record in successful
                if record["environment"] == environment
                and int(record["n_demos"]) == max(int(item["n_demos"]) for item in successful)
                and record["method"] == "neural"
            ),
            key=lambda record: float(record["evd"]),
        )
        if not candidates:
            continue
        chosen = candidates[len(candidates) // 2]
        selected[environment] = {
            "n_demos": int(chosen["n_demos"]),
            "panel_seed": int(chosen["panel_seed"]),
            "training_seed": int(chosen["training_seed"]),
            "evd": float(chosen["evd"]),
            "true_reward": chosen["true_reward"],
            "learned_reward": chosen["learned_reward"],
        }
    termination_counts: dict[str, int] = {}
    for record in successful:
        key = f"{record['method']}:{record.get('termination_reason', 'unknown')}"
        termination_counts[key] = termination_counts.get(key, 0) + 1

    return {
        "n_requested": len(records),
        "n_successful": len(successful),
        "n_converged": sum(record.get("converged") is True for record in successful),
        "termination_counts": termination_counts,
        "cells": cells,
        "selected_median_fits": selected,
    }


def render_figure(summary: dict[str, Any], grid_size: int, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    demo_counts = sorted(
        int(key.split(":")[1]) for key in summary["cells"] if key.startswith("objectworld:")
    )
    for row, environment in enumerate(("objectworld", "binaryworld")):
        selected = summary["selected_median_fits"][environment]
        true_reward = np.asarray(selected["true_reward"]).reshape(grid_size, grid_size)
        learned_reward = np.asarray(selected["learned_reward"]).reshape(grid_size, grid_size)
        vmin = min(float(true_reward.min()), float(learned_reward.min()))
        vmax = max(float(true_reward.max()), float(learned_reward.max()))
        axes[row, 0].imshow(true_reward, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"{environment.title()} true reward")
        axes[row, 1].imshow(learned_reward, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Median fit, EVD {selected['evd']:.3f}")
        neural = [
            summary["cells"][f"{environment}:{count}"]["neural_evd_median"] for count in demo_counts
        ]
        linear = [
            summary["cells"][f"{environment}:{count}"]["linear_evd_median"] for count in demo_counts
        ]
        axes[row, 2].plot(demo_counts, neural, marker="o", label="Neural MCE-IRL")
        axes[row, 2].plot(demo_counts, linear, marker="o", label="Linear MCE-IRL")
        axes[row, 2].set_xscale("log", base=2)
        axes[row, 2].set_xlabel("Demonstrations")
        axes[row, 2].set_ylabel("Expected value difference")
        axes[row, 2].legend(frameon=False)
        for column in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    grid_size = 8 if args.smoke else FULL_GRID
    demo_counts = (8,) if args.smoke else FULL_DEMO_COUNTS
    n_panel_seeds = 1 if args.smoke else FULL_PANEL_SEEDS
    n_training_seeds = 1 if args.smoke else FULL_TRAINING_SEEDS
    max_epochs = 10 if args.smoke else 200
    output = args.output or (
        Path("/tmp/deep_mce_irl_wulfmeier_smoke.json") if args.smoke else DEFAULT_OUTPUT
    )
    checkpoint = args.checkpoint or output.with_suffix(".jsonl")
    records = run_study(
        checkpoint,
        grid_size=grid_size,
        demo_counts=demo_counts,
        n_panel_seeds=n_panel_seeds,
        n_training_seeds=n_training_seeds,
        max_epochs=max_epochs,
        quiet=args.quiet,
    )
    summary = summarize(records)
    final_run = (
        not args.smoke
        and grid_size == FULL_GRID
        and demo_counts == FULL_DEMO_COUNTS
        and n_panel_seeds == FULL_PANEL_SEEDS
        and n_training_seeds == FULL_TRAINING_SEEDS
    )
    checks = [
        {
            "name": "all_cells_finite",
            "passed": summary["n_successful"] == summary["n_requested"],
        },
        {"name": "full_configuration", "passed": final_run},
    ]
    if final_run:
        for n_demos in (64, 128):
            cell = summary["cells"][f"binaryworld:{n_demos}"]
            checks.append(
                {
                    "name": f"binaryworld_linear_solution_residuals_pass_{n_demos}",
                    "passed": cell["linear_solution_residuals_pass"] == cell["linear_fits"] == 5,
                }
            )
            checks.append(
                {
                    "name": f"binaryworld_neural_beats_linear_{n_demos}",
                    "passed": (
                        cell["neural_evd_median"] is not None
                        and cell["linear_evd_median"] is not None
                        and cell["neural_evd_median"] < cell["linear_evd_median"]
                    ),
                }
            )
        objectworld = summary["cells"]["objectworld:128"]
        checks.extend(
            [
                {
                    "name": "objectworld_linear_solution_residuals_pass_128",
                    "passed": objectworld["linear_solution_residuals_pass"]
                    == objectworld["linear_fits"]
                    == 5,
                },
                {
                    "name": "objectworld_128_commensurate",
                    "passed": (
                        objectworld["neural_evd_median"] is not None
                        and objectworld["linear_evd_median"] is not None
                        and objectworld["neural_evd_median"] <= objectworld["linear_evd_median"]
                    ),
                },
                {
                    "name": "objectworld_transfer_128_commensurate",
                    "passed": (
                        objectworld["neural_transfer_evd_median"] is not None
                        and objectworld["linear_transfer_evd_median"] is not None
                        and objectworld["neural_transfer_evd_median"]
                        <= objectworld["linear_transfer_evd_median"]
                    ),
                },
            ]
        )
    status = "ready" if all(check["passed"] for check in checks) else "incomplete"
    payload = {
        "study": "Wulfmeier-shaped generated Objectworld and Binaryworld",
        "paper_replication": False,
        "configuration": {
            "grid_size": grid_size,
            "demo_counts": demo_counts,
            "panel_seeds": n_panel_seeds,
            "training_seeds": n_training_seeds,
            "horizon": 50,
            "random_action_fraction": 0.3,
            "discount": 0.9,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha": git_sha(),
            "packages": package_versions(),
        },
        "summary": summary,
        "checks": checks,
        "status": status,
        "checkpoint": str(checkpoint),
        "figure": "docs/_static/estimators/deep_mce_irl_wulfmeier.png",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if summary["n_successful"] == summary["n_requested"]:
        render_figure(
            summary,
            grid_size,
            ROOT / "docs" / "_static" / "estimators" / "deep_mce_irl_wulfmeier.png",
        )
    print(f"wrote {output}")
    print(f"status: {status}")
    return 0 if (args.smoke or status == "ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
