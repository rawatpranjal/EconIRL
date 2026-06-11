"""Figure generation for the simulation-studies page.

Four summary figures, built from the ``run_benchmark`` result dict:

1. recovery vs complexity (behavioral policy-TV across the difficulty ladder),
2. coverage of 95% CIs for the structural family (with a 0.95 reference line
   and Monte Carlo error bars),
3. bias vs variance for the structural family,
4. runtime across the ladder (log scale).

Uses matplotlib with the Agg backend so it runs headless in a docs build.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_FIGSIZE = (9, 5.5)
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]


def _estimator_colors(result: dict) -> dict[str, str]:
    names = [e["name"] for e in result["meta"]["estimators"]]
    return {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}


def _cells_sorted(result: dict) -> list[dict]:
    return sorted(result["cells"], key=lambda c: c["difficulty"])


def _structural_names(result: dict) -> list[str]:
    return [e["name"] for e in result["meta"]["estimators"] if e["family"] == "structural"]


def _all_names(result: dict) -> list[str]:
    return [e["name"] for e in result["meta"]["estimators"]]


def _entry(cell: dict, name: str) -> dict | None:
    for e in cell["estimators"]:
        if e["estimator"] == name:
            return e
    return None


def _mean_skip_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def plot_recovery_vs_complexity(result: dict, path: str) -> None:
    cells = _cells_sorted(result)
    colors = _estimator_colors(result)
    labels = [c["label"] for c in cells]
    x = np.arange(len(cells))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for name in _all_names(result):
        ys, es = [], []
        for cell in cells:
            e = _entry(cell, name)
            beh = e["behavioral"] if e else None
            ys.append(beh["policy_tv_mean"] if beh else None)
            es.append(beh["policy_tv_mc_se"] if beh else None)
        ys_plot = [np.nan if v is None else v for v in ys]
        es_plot = [0.0 if v is None else v for v in es]
        ax.errorbar(x, ys_plot, yerr=es_plot, marker="o", capsize=3,
                    color=colors[name], label=name, linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Policy total-variation distance (lower is better)")
    ax.set_title("Behavioral recovery across the difficulty ladder")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Estimator")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coverage(result: dict, path: str) -> None:
    cells = _cells_sorted(result)
    colors = _estimator_colors(result)
    struct = _structural_names(result)

    # Cells where coverage is defined (finite theta, structural SEs present).
    cov_cells = []
    for cell in cells:
        if not cell["has_finite_theta"]:
            continue
        if any(
            (_entry(cell, n) or {}).get("parameters")
            and _entry(cell, n)["parameters"]["se_available"]
            for n in struct
        ):
            cov_cells.append(cell)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if not cov_cells:
        ax.text(0.5, 0.5, "No cells with finite-theta coverage", ha="center", va="center")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return

    labels = [c["label"] for c in cov_cells]
    x = np.arange(len(cov_cells))
    width = 0.8 / max(len(struct), 1)
    for i, name in enumerate(struct):
        ys, es = [], []
        for cell in cov_cells:
            e = _entry(cell, name)
            p = e["parameters"] if e else None
            if p and p["se_available"]:
                ys.append(_mean_skip_none(p["coverage_95"]))
                es.append(_mean_skip_none(p["coverage_95_mc_se"]) or 0.0)
            else:
                ys.append(np.nan)
                es.append(0.0)
        ax.bar(x + i * width, ys, width, yerr=es, capsize=3,
               color=colors[name], label=name)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, label="nominal 0.95")
    ax.set_xticks(x + width * (len(struct) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("95% CI coverage (mean over parameters)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Confidence-interval coverage, structural family")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Estimator", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bias_variance(result: dict, path: str) -> None:
    cells = _cells_sorted(result)
    colors = _estimator_colors(result)
    struct = _structural_names(result)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    plotted = False
    for name in struct:
        xs, ys = [], []
        for cell in cells:
            e = _entry(cell, name)
            p = e["parameters"] if e else None
            if not p:
                continue
            mean_abs_bias = float(np.mean(np.abs(p["bias"])))
            mean_emp_se = _mean_skip_none(p["empirical_se"])
            if mean_emp_se is None:
                continue
            xs.append(mean_abs_bias)
            ys.append(mean_emp_se)
        if xs:
            ax.scatter(xs, ys, color=colors[name], label=name, s=60, alpha=0.8,
                       edgecolors="white")
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No structural parameter metrics", ha="center", va="center")
    ax.set_xlabel("Mean |bias| (over parameters and cells)")
    ax.set_ylabel("Empirical standard error")
    ax.set_title("Bias vs variance, structural family")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(title="Estimator")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_runtime(result: dict, path: str) -> None:
    cells = _cells_sorted(result)
    colors = _estimator_colors(result)
    labels = [c["label"] for c in cells]
    x = np.arange(len(cells))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for name in _all_names(result):
        ys = []
        for cell in cells:
            e = _entry(cell, name)
            ys.append(e["runtime_mean"] if e and e["runtime_mean"] else np.nan)
        ax.plot(x, ys, marker="s", color=colors[name], label=name, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean fit time per replication (s, log scale)")
    ax.set_title("Runtime across the difficulty ladder")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="Estimator")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


FIGURES = [
    ("recovery_vs_complexity.png", plot_recovery_vs_complexity,
     "Behavioral recovery (policy total-variation distance) for each estimator across the difficulty ladder."),
    ("coverage.png", plot_coverage,
     "Coverage of nominal 95% confidence intervals for the structural family, with the 0.95 reference line."),
    ("bias_variance.png", plot_bias_variance,
     "Bias against empirical standard error for the structural family."),
    ("runtime.png", plot_runtime,
     "Mean fit time per replication across the ladder, on a log scale."),
]


def generate_all(result: dict, out_dir: str) -> list[tuple[str, str]]:
    """Write all figures into ``out_dir``; return [(filename, caption)]."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for filename, fn, caption in FIGURES:
        fn(result, os.path.join(out_dir, filename))
        written.append((filename, caption))
    return written
