#!/usr/bin/env python3
"""Monte-Carlo parameter-recovery driver for the TD-CCP estimator.

Tier-3 (bespoke) driver: uses the custom Shapeshifter DGP from
``run.py::build_paper_hard_case_dgp`` rather than the shared known-truth path.
Each replication calls ``build_paper_hard_case_dgp(base_seed + rep)`` so the
environment, feature matrix, and true parameters all vary with the seed, giving
a genuine sampling distribution. The estimator is configured with
``paper_hard_case_estimator_config`` exactly as ``run.py`` does.

Produces:
  * ``validation/results/tdccp_recovery.json``  -- per-param + aggregate stats
  * ``docs/_static/estimators/tdccp_recovery.png``  -- 6-row caterpillar

True params are the DGP's ``true_params`` vector (6 elements):
  action_1_intercept, action_1_x0, action_1_x1,
  action_2_intercept, action_2_x0, action_2_x1

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python validation/estimators/tdccp/recovery_mc.py --n-reps 200

    # fast smoke test
    PYTHONPATH=src:. python validation/estimators/tdccp/recovery_mc.py --n-reps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "tdccp_recovery.json"
FIG_OUT = ROOT / "docs" / "_static" / "estimators" / "tdccp_recovery.png"

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp  # noqa: E402

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.estimation.td_ccp import TDCCPEstimator  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402
from validation.estimators.tdccp.run import (  # noqa: E402
    HARD_CASE_N_INDIVIDUALS,
    HARD_CASE_N_PERIODS,
    build_paper_hard_case_dgp,
    paper_hard_case_estimator_config,
)
from validation.known_truth import to_jsonable  # noqa: E402


def run_one_rep(seed: int) -> tuple[list[str], list[float], list[float]]:
    """Build DGP, simulate panel, fit TD-CCP; return (param_names, true, estimated)."""
    dgp = build_paper_hard_case_dgp(seed)
    env = dgp["env"]
    utility = dgp["utility"]
    true_params = np.asarray(dgp["true_params"], dtype=np.float64)
    true_reward = jnp.asarray(dgp["true_reward"])

    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    truth = value_iteration(operator, true_reward, tol=1e-10, max_iter=10_000)

    initial_distribution = jnp.asarray(env._get_initial_state_distribution())
    panel = simulate_panel_from_policy(
        env.problem_spec,
        env.transition_matrices,
        truth.policy,
        initial_distribution,
        n_individuals=HARD_CASE_N_INDIVIDUALS,
        n_periods=HARD_CASE_N_PERIODS,
        seed=seed,
    )

    estimator_config = paper_hard_case_estimator_config(verbose=False)
    estimator = TDCCPEstimator(config=estimator_config, seed=seed)
    summary = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    param_names = list(utility.parameter_names)
    estimated = np.asarray(summary.parameters, dtype=np.float64).tolist()
    true_list = true_params.tolist()
    return param_names, true_list, estimated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    n_reps = args.n_reps
    base_seed = args.base_seed

    print("TD-CCP Monte-Carlo parameter recovery (Tier-3 bespoke)")
    print("  DGP: build_paper_hard_case_dgp (seed varies per rep)")
    print("  estimator: paper_hard_case_estimator_config (Algorithm 2, encoded, logit CCPs)")
    print(f"  n_reps: {n_reps}  base_seed: {base_seed}")
    print(f"  panel per rep: {HARD_CASE_N_INDIVIDUALS} individuals x {HARD_CASE_N_PERIODS} periods")

    param_names: list[str] | None = None
    true_params_list: list[list[float]] = []
    estimates: list[list[float]] = []
    t0 = time.time()

    for rep in range(n_reps):
        seed = base_seed + rep
        pnames, true_list, estimated = run_one_rep(seed)
        if param_names is None:
            param_names = pnames
        true_params_list.append(true_list)
        estimates.append(estimated)

        if (rep + 1) % 10 == 0 or rep == 0 or rep + 1 == n_reps:
            elapsed = time.time() - t0
            print(f"  rep {rep + 1}/{n_reps}  (seed={seed}, {elapsed:.1f}s elapsed)")

    assert param_names is not None
    estimate_array = np.asarray(estimates, dtype=float)     # (n_reps, n_params)
    true_array = np.asarray(true_params_list, dtype=float)  # (n_reps, n_params)

    # Per-param stats: use the per-rep true value mean as the "true" anchor
    # (true params vary across reps because the DGP seed varies). We report the
    # mean true, the mean estimate, and CIs on the estimation error.
    per_param: dict[str, dict[str, float]] = {}
    for j, name in enumerate(param_names):
        est_col = estimate_array[:, j]
        true_col = true_array[:, j]
        errors = est_col - true_col
        mean_true = float(np.mean(true_col))
        mean_est = float(np.mean(est_col))
        # CI on the error distribution (not the estimate itself)
        per_param[name] = {
            "true": mean_true,            # mean true value across reps
            "mean": mean_est,             # mean estimate across reps
            "std": float(np.std(est_col, ddof=1)) if est_col.size > 1 else 0.0,
            "ci_lo": float(np.percentile(est_col, 2.5)),
            "ci_hi": float(np.percentile(est_col, 97.5)),
            "bias": float(np.mean(errors)),
            "error_std": float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
            "error_ci_lo": float(np.percentile(errors, 2.5)),
            "error_ci_hi": float(np.percentile(errors, 97.5)),
        }

    payload: dict[str, Any] = {
        "estimator": "TD-CCP",
        "n_reps": int(n_reps),
        "base_seed": int(base_seed),
        "dgp": "build_paper_hard_case_dgp (seed varies per rep)",
        "param_names": param_names,
        "params": per_param,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote: {JSON_OUT}")

    render_caterpillar(payload, param_names, FIG_OUT)
    print(f"  wrote: {FIG_OUT}")

    print("\nParameter recovery summary")
    print(
        f"  {'parameter':<30}{'mean_true':>12}{'mean_est':>12}"
        f"{'ci_lo':>10}{'ci_hi':>10}{'bias':>10}"
    )
    for name in param_names:
        p = per_param[name]
        print(
            f"  {name:<30}{p['true']:>12.4f}{p['mean']:>12.4f}"
            f"{p['ci_lo']:>10.4f}{p['ci_hi']:>10.4f}{p['bias']:>10.4f}"
        )


def render_caterpillar(
    payload: dict[str, Any],
    param_names: list[str],
    out_path: Path,
) -> None:
    """Render a caterpillar plot for the 6 TD-CCP parameters."""
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
            color=est_color, linewidth=1.4, solid_capstyle="round", zorder=2,
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
        f"TD-CCP parameter recovery (Monte Carlo, {payload['n_reps']} reps)",
        fontsize=10.5, pad=10,
    )

    legend_handles = [
        Line2D(
            [0], [0], color=est_color, marker="o", markersize=5.5,
            markeredgecolor="white", linewidth=1.4, label="estimate + 95% CI",
        ),
        Line2D(
            [0], [0], color=true_color, marker="D", markersize=7.0,
            markeredgecolor="white", linestyle="none", label="mean true value",
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
