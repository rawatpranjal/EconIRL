"""Render the benchmark result dict into the generated Markdown results partial.

Produces the per-cell segmented tables and the failure-mode map that the public
``simulation_studies.md`` page includes. Kept separate from the prose so the
numbers regenerate cleanly while the narrative stays hand-authored.
"""

from __future__ import annotations

import numpy as np


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _pm(mean: float | None, se: float | None, nd: int = 3) -> str:
    if mean is None:
        return "n/a"
    if se is None or (isinstance(se, float) and np.isnan(se)):
        return _fmt(mean, nd)
    return f"{mean:.{nd}f} ± {se:.{nd}f}"


def _mean_skip_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _entry(cell: dict, name: str) -> dict | None:
    for e in cell["estimators"]:
        if e["estimator"] == name:
            return e
    return None


def render_cell(cell: dict) -> str:
    diag = cell["diagnostics"]
    lines = []
    lines.append(f"### {cell['label']}\n")
    lines.append(f"{cell['description']}\n")
    lines.append(
        f"States {cell['num_states']}, actions {cell['num_actions']}, "
        f"discount {cell['discount_factor']:.3f}. Stresses: {cell['stresses']}. "
        f"Feature rank {diag['feature_rank']} of {diag['num_features']}, "
        f"condition number {diag['condition_number']:.1f}.\n"
    )
    lines.append(
        "| Estimator | Family | Policy TV | Value RMSE | Mean \\|bias\\| | "
        "Coverage 95% | Runtime (s) | Converged | Crashed |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in cell["estimators"]:
        beh = e["behavioral"]
        tv = _pm(beh["policy_tv_mean"], beh["policy_tv_mc_se"], 3)
        vr = _pm(beh.get("value_rmse_mean"), beh.get("value_rmse_mc_se"), 3)
        p = e["parameters"]
        if p:
            mean_abs_bias = _fmt(float(np.mean(np.abs(p["bias"]))), 4)
            cov = _mean_skip_none(p["coverage_95"])
            cov_se = _mean_skip_none(p["coverage_95_mc_se"])
            cov_s = _pm(cov, cov_se, 2) if cov is not None else "n/a"
        else:
            mean_abs_bias = "n/a"
            cov_s = "n/a"
        rt = _fmt(e["runtime_mean"], 2)
        conv = _fmt(e["convergence_rate"], 2)
        crashed = str(e["n_crashed"])
        lines.append(
            f"| {e['estimator']} | {e['family']} | {tv} | {vr} | "
            f"{mean_abs_bias} | {cov_s} | {rt} | {conv} | {crashed} |"
        )
    lines.append("")
    return "\n".join(lines)


def _mean_abs_bias(p: dict) -> float:
    return float(np.mean(np.abs(np.asarray(p["bias"], dtype=np.float64))))


def render_failure_map(
    result: dict, tv_threshold: float = 0.1, cond_threshold: float = 1e6
) -> str:
    """One row per cell summarizing which estimators struggled and why.

    Flags an estimator when it crashed, when its policy total-variation distance
    exceeded ``tv_threshold``, or when its coverage fell below 0.80. On a cell
    whose design is rank-deficient or ill-conditioned (condition number above
    ``cond_threshold``), structural estimators are additionally flagged with
    their absolute parameter bias: there the parameters are not identified, so
    the standard errors widen honestly and confidence intervals still cover even
    though the point estimates are far off. That is the one failure coverage
    alone cannot show, so the cell's identification status is named explicitly.
    """
    lines = []
    lines.append("### Failure-mode map\n")
    lines.append(
        "Where each estimator struggles, and the data-generating condition "
        "responsible. An estimator is flagged when it crashed, when its policy "
        f"total-variation distance exceeded {tv_threshold:.2f}, or when coverage "
        "fell below 0.80. On a cell whose design is rank-deficient or "
        "ill-conditioned, structural estimators are also flagged with their "
        "parameter bias, because there the parameters are not identified even "
        "though behavior and coverage stay healthy.\n"
    )
    lines.append("| Cell | Stressed condition | Estimators that struggled |")
    lines.append("|---|---|---|")
    for cell in sorted(result["cells"], key=lambda c: c["difficulty"]):
        diag = cell["diagnostics"]
        ill_identified = (
            diag["feature_rank"] < diag["num_features"]
            or diag["condition_number"] > cond_threshold
        )
        condition = cell["stresses"]
        if ill_identified:
            condition += (
                f" (design rank {diag['feature_rank']}/{diag['num_features']}, "
                f"condition {diag['condition_number']:.0e}: parameters not identified)"
            )
        flagged = []
        for e in cell["estimators"]:
            reasons = []
            if e["n_crashed"] > 0:
                reasons.append("crashed")
            tv = e["behavioral"]["policy_tv_mean"]
            if tv is not None and tv > tv_threshold:
                reasons.append(f"TV {tv:.2f}")
            p = e["parameters"]
            if p:
                if p["se_available"]:
                    cov = _mean_skip_none(p["coverage_95"])
                    if cov is not None and cov < 0.80:
                        reasons.append(f"cov {cov:.2f}")
                if ill_identified:
                    reasons.append(f"params unidentified (|bias| {_mean_abs_bias(p):.2f})")
            if reasons:
                flagged.append(f"{e['estimator']} ({', '.join(reasons)})")
        cell_label = flagged and ", ".join(flagged) or "none"
        lines.append(f"| {cell['label']} | {condition} | {cell_label} |")
    lines.append("")
    return "\n".join(lines)


def render_results_markdown(result: dict) -> str:
    """Full generated results partial: header note, per-cell tables, failure map."""
    meta = result["meta"]
    est_list = ", ".join(
        f"{e['name']} ({e['family']})" for e in meta["estimators"]
    )
    out = []
    out.append(
        "<!-- GENERATED by scripts/run_simulation_studies.py. Do not edit by hand. -->\n"
    )
    out.append(
        f"Generated {meta['date']} with econirl {meta['package_version']}, "
        f"{meta['n_replications']} Monte Carlo replications per cell. "
        f"Estimators: {est_list}. Coverage uses a {meta['coverage_convention']}. "
        "Reported uncertainties are Monte Carlo standard errors.\n"
    )
    for cell in sorted(result["cells"], key=lambda c: c["difficulty"]):
        out.append(render_cell(cell))
    out.append(render_failure_map(result))
    return "\n".join(out)
