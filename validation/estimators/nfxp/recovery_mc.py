#!/usr/bin/env python3
"""Monte-Carlo parameter-recovery driver for the NFXP estimator.

Re-simulates the canonical known-truth panel with a fresh seed each
replication, re-fits NFXP, and collects the sampling distribution of the
recovered structural parameters. Produces the figure and numbers behind the
NFXP public-docs Evidence section. Every number flows from this script; nothing
is hand-typed.

Per replication this mirrors what ``run_cell_estimator`` does internally for one
fit (build DGP, simulate panel, run NFXP), except:
  - the panel seed is ``base_seed + rep`` instead of the cell's fixed seed, and
  - recovery gates are disabled so a single off draw does not abort the loop;
    the point is the full sampling distribution, not a one-sample release check.

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python validation/estimators/nfxp/recovery_mc.py --n-reps 200

    # fast smoke test
    PYTHONPATH=src:. python validation/estimators/nfxp/recovery_mc.py --n-reps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "nfxp_recovery.json"
FIG_OUT = ROOT / "docs" / "_static" / "estimators" / "nfxp_recovery.png"
CELL_ID = "canonical_low_action"
ESTIMATOR = "NFXP"

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validation.known_truth import (  # noqa: E402
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    to_jsonable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--cell-id", default=CELL_ID)
    args = parser.parse_args()

    cell = get_cell(args.cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)  # deterministic; reused across reps
    true_params = np.asarray(dgp.homogeneous_parameters, dtype=float)

    print(f"NFXP Monte-Carlo parameter recovery")
    print(f"  cell: {args.cell_id}")
    print(f"  estimator: {ESTIMATOR}")
    print(f"  n_reps: {args.n_reps}  base_seed: {args.base_seed}")
    print(
        f"  panel: {cell.simulation_config.n_individuals} individuals "
        f"x {cell.simulation_config.n_periods} periods, "
        f"{dgp.problem.num_states} states, {true_params.size} params"
    )

    param_names: list[str] | None = None
    estimates: list[list[float]] = []
    t0 = time.time()

    for rep in range(args.n_reps):
        seed = args.base_seed + rep
        sim_config = replace(cell.simulation_config, seed=seed, show_progress=False)
        panel = simulate_known_truth_panel(dgp, sim_config)
        run = run_estimator(
            ESTIMATOR,
            dgp,
            panel,
            smoke=False,
            enforce_gates=False,  # collect every draw, not a release gate
        )
        summary = run.summary
        if param_names is None:
            param_names = list(summary.parameter_names)
        estimates.append(np.asarray(summary.parameters, dtype=float).tolist())

        if (rep + 1) % 10 == 0 or rep == 0 or rep + 1 == args.n_reps:
            elapsed = time.time() - t0
            print(
                f"  rep {rep + 1}/{args.n_reps}  "
                f"(seed={seed}, {elapsed:.1f}s elapsed)"
            )

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
        "estimator": ESTIMATOR,
        "n_reps": int(args.n_reps),
        "base_seed": int(args.base_seed),
        "cell_id": args.cell_id,
        "param_names": param_names,
        "params": per_param,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"  wrote: {JSON_OUT}")

    render_figure(payload, param_names, FIG_OUT)
    print(f"  wrote: {FIG_OUT}")

    print("\nParameter recovery summary")
    print(f"  {'parameter':<22}{'true':>10}{'mean':>10}{'ci_lo':>10}{'ci_hi':>10}{'bias':>10}")
    for name in param_names:
        p = per_param[name]
        print(
            f"  {name:<22}{p['true']:>10.4f}{p['mean']:>10.4f}"
            f"{p['ci_lo']:>10.4f}{p['ci_hi']:>10.4f}{p['bias']:>10.4f}"
        )


def render_figure(
    payload: dict[str, Any],
    param_names: list[str],
    out_path: Path,
) -> None:
    """Render a caterpillar/forest plot of the MC recovery distribution."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    per_param = payload["params"]
    n = len(param_names)
    # Top-to-bottom row order matching the param list.
    rows = list(range(n))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 0.85 * n + 1.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    est_color = "#1f3a93"
    true_color = "#e08a1e"

    for row, name in zip(rows, param_names):
        p = per_param[name]
        # 95% CI whisker.
        ax.plot(
            [p["ci_lo"], p["ci_hi"]],
            [row, row],
            color=est_color,
            linewidth=1.4,
            solid_capstyle="round",
            zorder=2,
        )
        # CI end caps.
        for x in (p["ci_lo"], p["ci_hi"]):
            ax.plot([x, x], [row - 0.12, row + 0.12], color=est_color, linewidth=1.0, zorder=2)
        # Mean estimate dot.
        ax.plot(
            p["mean"], row, marker="o", markersize=5.5,
            color=est_color, markeredgecolor="white", markeredgewidth=0.6, zorder=4,
        )
        # True value marker.
        ax.plot(
            p["true"], row, marker="D", markersize=7.0,
            color=true_color, markeredgecolor="white", markeredgewidth=0.6, zorder=5,
        )

    ax.set_yticks(rows)
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="both", length=3, width=0.6)

    # Tight x-limits around the param range so CIs stay visible.
    lows = [per_param[name]["ci_lo"] for name in param_names]
    highs = [per_param[name]["ci_hi"] for name in param_names]
    trues = [per_param[name]["true"] for name in param_names]
    xmin = min(lows + trues)
    xmax = max(highs + trues)
    pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)

    ax.axvline(0.0, color="#cccccc", linewidth=0.6, zorder=1)

    # Hairline axes.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#444444")

    ax.set_xlabel("parameter value", fontsize=9)
    ax.set_title(
        f"NFXP parameter recovery (Monte Carlo, {payload['n_reps']} reps)",
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


if __name__ == "__main__":
    main()
