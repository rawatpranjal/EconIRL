"""Shared Monte-Carlo parameter-recovery logic for structural estimators.

Each per-estimator ``recovery_mc.py`` is a thin wrapper that imports from here
and passes a small set of constants (cell id, estimator name, output paths,
tier). This module owns the MC loop, serialization, and figure rendering.

Tiers
-----
tier1 : low-dim caterpillar (ccp, mpec, ufxp)
    canonical_low_action, 4 params, standard caterpillar figure.

tier2 : high-dim compact (nnes, sees)
    canonical_high_action, 32 params, compact error-distribution figure
    plus aggregate fields (recovery_rmse, coverage_frac) in the JSON.

tier3 : tdccp bespoke -- NOT handled here; lives entirely in tdccp/recovery_mc.py
    Uses a custom DGP, not the shared known-truth path.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

# Make sure validation package is importable when called from sub-dirs.
for _candidate in (
    Path(__file__).resolve().parents[2],          # repo root
    Path(__file__).resolve().parents[2] / "src",  # src layout
    Path(__file__).resolve().parent,              # estimators/
):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from validation.known_truth import (  # noqa: E402
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    to_jsonable,
)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_tier1(
    *,
    estimator: str,
    cell_id: str,
    json_out: Path,
    fig_out: Path,
    n_reps: int,
    base_seed: int,
) -> dict[str, Any]:
    """Run a Tier-1 (low-dim caterpillar) MC recovery loop.

    Returns the full payload dict (also written to *json_out* and rendered to
    *fig_out*).
    """
    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    true_params = np.asarray(dgp.homogeneous_parameters, dtype=float)

    print(f"{estimator} Monte-Carlo parameter recovery")
    print(f"  cell: {cell_id}")
    print(f"  n_reps: {n_reps}  base_seed: {base_seed}")
    print(
        f"  panel: {cell.simulation_config.n_individuals} individuals "
        f"x {cell.simulation_config.n_periods} periods, "
        f"{dgp.problem.num_states} states, {true_params.size} params"
    )

    param_names: list[str] | None = None
    estimates: list[list[float]] = []
    t0 = time.time()

    for rep in range(n_reps):
        seed = base_seed + rep
        sim_config = replace(cell.simulation_config, seed=seed, show_progress=False)
        panel = simulate_known_truth_panel(dgp, sim_config)
        run = run_estimator(
            estimator,
            dgp,
            panel,
            smoke=False,
            enforce_gates=False,
        )
        summary = run.summary
        if param_names is None:
            param_names = list(summary.parameter_names)
        estimates.append(np.asarray(summary.parameters, dtype=float).tolist())

        if (rep + 1) % 10 == 0 or rep == 0 or rep + 1 == n_reps:
            elapsed = time.time() - t0
            print(f"  rep {rep + 1}/{n_reps}  (seed={seed}, {elapsed:.1f}s elapsed)")

    assert param_names is not None
    estimate_array = np.asarray(estimates, dtype=float)  # (n_reps, n_params)

    per_param: dict[str, dict[str, float]] = {}
    for j, name in enumerate(param_names):
        column = estimate_array[:, j]
        true_value = float(true_params[j])
        mean = float(np.mean(column))
        per_param[name] = {
            "true": true_value,
            "mean": mean,
            "std": float(np.std(column, ddof=1)) if column.size > 1 else 0.0,
            "ci_lo": float(np.percentile(column, 2.5)),
            "ci_hi": float(np.percentile(column, 97.5)),
            "bias": float(mean - true_value),
        }

    payload: dict[str, Any] = {
        "estimator": estimator,
        "n_reps": int(n_reps),
        "base_seed": int(base_seed),
        "cell_id": cell_id,
        "param_names": param_names,
        "params": per_param,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote: {json_out}")

    render_caterpillar(payload, param_names, fig_out, estimator=estimator)
    print(f"  wrote: {fig_out}")

    _print_summary_table(param_names, per_param)
    return payload


def run_tier2(
    *,
    estimator: str,
    cell_id: str,
    json_out: Path,
    fig_out: Path,
    n_reps: int,
    base_seed: int,
) -> dict[str, Any]:
    """Run a Tier-2 (high-dim compact) MC recovery loop.

    All 32 per-param stats are stored in the JSON (reproducibility), but the
    figure is a compact error-distribution histogram rather than 32 caterpillar
    rows.  Two aggregate fields are added: *recovery_rmse* and *coverage_frac*.
    """
    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    true_params = np.asarray(dgp.homogeneous_parameters, dtype=float)

    print(f"{estimator} Monte-Carlo parameter recovery (high-dim compact)")
    print(f"  cell: {cell_id}")
    print(f"  n_reps: {n_reps}  base_seed: {base_seed}")
    print(
        f"  panel: {cell.simulation_config.n_individuals} individuals "
        f"x {cell.simulation_config.n_periods} periods, "
        f"{dgp.problem.num_states} states, {true_params.size} params"
    )

    param_names: list[str] | None = None
    estimates: list[list[float]] = []
    t0 = time.time()

    for rep in range(n_reps):
        seed = base_seed + rep
        sim_config = replace(cell.simulation_config, seed=seed, show_progress=False)
        panel = simulate_known_truth_panel(dgp, sim_config)
        run = run_estimator(
            estimator,
            dgp,
            panel,
            smoke=False,
            enforce_gates=False,
        )
        summary = run.summary
        if param_names is None:
            param_names = list(summary.parameter_names)
        estimates.append(np.asarray(summary.parameters, dtype=float).tolist())

        if (rep + 1) % 10 == 0 or rep == 0 or rep + 1 == n_reps:
            elapsed = time.time() - t0
            print(f"  rep {rep + 1}/{n_reps}  (seed={seed}, {elapsed:.1f}s elapsed)")

    assert param_names is not None
    estimate_array = np.asarray(estimates, dtype=float)  # (n_reps, n_params)

    per_param: dict[str, dict[str, float]] = {}
    for j, name in enumerate(param_names):
        column = estimate_array[:, j]
        true_value = float(true_params[j])
        mean = float(np.mean(column))
        ci_lo = float(np.percentile(column, 2.5))
        ci_hi = float(np.percentile(column, 97.5))
        per_param[name] = {
            "true": true_value,
            "mean": mean,
            "std": float(np.std(column, ddof=1)) if column.size > 1 else 0.0,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "bias": float(mean - true_value),
        }

    # Aggregate fields.
    biases = np.array([per_param[n]["bias"] for n in param_names])
    recovery_rmse = float(np.sqrt(np.mean(biases**2)))
    coverage_frac = float(
        np.mean([
            1.0 if per_param[n]["ci_lo"] <= per_param[n]["true"] <= per_param[n]["ci_hi"]
            else 0.0
            for n in param_names
        ])
    )

    payload: dict[str, Any] = {
        "estimator": estimator,
        "n_reps": int(n_reps),
        "base_seed": int(base_seed),
        "cell_id": cell_id,
        "param_names": param_names,
        "params": per_param,
        "recovery_rmse": recovery_rmse,
        "coverage_frac": coverage_frac,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote: {json_out}")

    render_compact(payload, param_names, fig_out, estimator=estimator)
    print(f"  wrote: {fig_out}")

    print(f"\nAggregate recovery: rmse={recovery_rmse:.4f}  coverage_frac={coverage_frac:.3f}")
    return payload


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_caterpillar(
    payload: dict[str, Any],
    param_names: list[str],
    out_path: Path,
    *,
    estimator: str,
) -> None:
    """Render a caterpillar / forest plot (one row per parameter)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    per_param = payload["params"]
    n = len(param_names)
    rows = list(range(n))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 0.85 * n + 1.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    est_color = "#1f3a93"
    true_color = "#e08a1e"

    for row, name in zip(rows, param_names):
        p = per_param[name]
        ax.plot(
            [p["ci_lo"], p["ci_hi"]],
            [row, row],
            color=est_color,
            linewidth=1.4,
            solid_capstyle="round",
            zorder=2,
        )
        for x in (p["ci_lo"], p["ci_hi"]):
            ax.plot([x, x], [row - 0.12, row + 0.12], color=est_color, linewidth=1.0, zorder=2)
        ax.plot(
            p["mean"], row, marker="o", markersize=5.5,
            color=est_color, markeredgecolor="white", markeredgewidth=0.6, zorder=4,
        )
        ax.plot(
            p["true"], row, marker="D", markersize=7.0,
            color=true_color, markeredgecolor="white", markeredgewidth=0.6, zorder=5,
        )

    ax.set_yticks(rows)
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="both", length=3, width=0.6)

    lows = [per_param[name]["ci_lo"] for name in param_names]
    highs = [per_param[name]["ci_hi"] for name in param_names]
    trues = [per_param[name]["true"] for name in param_names]
    xmin = min(lows + trues)
    xmax = max(highs + trues)
    pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)

    ax.axvline(0.0, color="#cccccc", linewidth=0.6, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#444444")

    ax.set_xlabel("parameter value", fontsize=9)
    ax.set_title(
        f"{estimator} parameter recovery (Monte Carlo, {payload['n_reps']} reps)",
        fontsize=10.5,
        pad=10,
    )

    legend_handles = [
        Line2D(
            [0], [0], color=est_color, marker="o", markersize=5.5,
            markeredgecolor="white", linewidth=1.4, label="estimate + 95% CI",
        ),
        Line2D(
            [0], [0], color=true_color, marker="D", markersize=7.0,
            markeredgecolor="white", linestyle="none", label="true value",
        ),
    ]
    ax.legend(
        handles=legend_handles, fontsize=8, frameon=False,
        loc="lower right", handletextpad=0.5,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_compact(
    payload: dict[str, Any],
    param_names: list[str],
    out_path: Path,
    *,
    estimator: str,
) -> None:
    """Render a compact error-distribution figure for high-dim (32-param) cases.

    Shows a histogram of per-parameter recovery bias (mean - true) across the
    K parameters, annotated with the aggregate recovery RMSE and the fraction
    of parameters whose truth falls inside the 95% CI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_param = payload["params"]
    biases = np.array([per_param[n]["bias"] for n in param_names])
    recovery_rmse = float(payload["recovery_rmse"])
    coverage_frac = float(payload["coverage_frac"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_color = "#1f3a93"
    ax.hist(biases, bins=min(16, max(6, len(param_names) // 2)), color=bar_color,
            alpha=0.80, edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color="#e08a1e", linewidth=1.4, linestyle="--", label="zero bias")

    ax.text(
        0.97, 0.95,
        f"RMSE = {recovery_rmse:.4f}\ncoverage = {coverage_frac:.2%}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.5,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc",
                  linewidth=0.6, alpha=0.9),
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#444444")

    ax.tick_params(axis="both", labelsize=8, length=3, width=0.6)
    ax.set_xlabel("bias (mean estimate - true)", fontsize=9)
    ax.set_ylabel("number of parameters", fontsize=9)
    ax.set_title(
        f"{estimator} recovery bias across {len(param_names)} params"
        f" (Monte Carlo, {payload['n_reps']} reps)",
        fontsize=10.0,
        pad=10,
    )
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_summary_table(
    param_names: list[str],
    per_param: dict[str, dict[str, float]],
) -> None:
    print("\nParameter recovery summary")
    print(
        f"  {'parameter':<22}{'true':>10}{'mean':>10}"
        f"{'ci_lo':>10}{'ci_hi':>10}{'bias':>10}"
    )
    for name in param_names:
        p = per_param[name]
        print(
            f"  {name:<22}{p['true']:>10.4f}{p['mean']:>10.4f}"
            f"{p['ci_lo']:>10.4f}{p['ci_hi']:>10.4f}{p['bias']:>10.4f}"
        )
