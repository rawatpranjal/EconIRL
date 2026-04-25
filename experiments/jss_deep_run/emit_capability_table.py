"""Generate the empirical capability table for the JSS paper.

Reads aggregated Tier 4 results from
``experiments/jss_deep_run/results/shapeshifter/headline.csv`` and
emits a LaTeX ``tabular*`` snippet to
``papers/econirl_package/tables/capability_empirical.tex`` that the
paper ``\\input``s into Table 3.

Cell semantics:
- ``\\checkmark`` if the estimator's row in the relevant ss-* cell
  meets the acceptance criterion (per ``plan_shapeshifter.md``).
- ``$\\times$`` if the cell ran but failed (failure_mode != none).
- ``na`` if the cell was not run because the source paper does not
  claim support for that regime (decided by the matrix.py per-cell
  estimator scope).
- ``pending`` if Tier 4 has not yet been dispatched for that cell.

The script is intended to run on a RunPod CPU pod after the Tier 4
dispatch completes; it does not require GPU. Schema of the input CSV:

    cell_id, estimator, headline_tag, mean_log_likelihood,
    mean_policy_kl, mean_relative_param_error, mean_reward_correlation,
    convergence_rate, failure_mode_majority

The script does not run any estimators; it only reshapes the
aggregated CSV. Safe to invoke from cloud_test.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout: rows and columns must match Table 2 in estimators.tex
# ---------------------------------------------------------------------------

# The per-axis acceptance criterion is met when the cell's failure_mode
# majority is "none" and the convergence rate is at least this threshold.
_CONVERGENCE_THRESHOLD = 0.85

# Column id -> ss-cell suffix that exercises that capability.
_COLUMN_TO_CELL = {
    "linear_r":     "spine",
    "neural_r":     "neural_r",
    "linear_phi":   "spine",
    "neural_phi":   "neural_phi",
    "action_dep_r": "spine",
    "state_only_r": "state_only",
    "finite":       "finite",
}

# Row id (display name) -> matrix.py estimator name.
# Order matches the theoretical capability table for visual diff.
_ROWS_BY_FAMILY = [
    [("NFXP-NK", "NFXP"), ("CCP-NPL", "CCP"), ("MPEC", "MPEC")],
    [("NNES", "NNES"), ("TD-CCP", "TD-CCP"), ("SEES", "SEES")],
    [
        ("MCE-IRL", "MCE-IRL"),
        ("AIRL", "AIRL"),
        ("GLADIUS", "GLADIUS"),
        ("f-IRL", "f-IRL"),
        ("IQ-Learn", "IQ-Learn"),
        ("BC", "BC"),
    ],
]

# Theoretical-support map (from Table 2). Cells the paper claims as
# unsupported get an "na" mark in the empirical table regardless of
# whether the cell ran.
_THEORETICAL_SUPPORT: dict[str, dict[str, bool]] = {
    "NFXP":     {"linear_r": True,  "neural_r": False, "linear_phi": True,  "neural_phi": False, "action_dep_r": True, "state_only_r": True,  "finite": True},
    "CCP":      {"linear_r": True,  "neural_r": False, "linear_phi": True,  "neural_phi": False, "action_dep_r": True, "state_only_r": True,  "finite": True},
    "MPEC":     {"linear_r": True,  "neural_r": False, "linear_phi": True,  "neural_phi": False, "action_dep_r": True, "state_only_r": True,  "finite": True},
    "NNES":     {"linear_r": True,  "neural_r": True,  "linear_phi": True,  "neural_phi": True,  "action_dep_r": True, "state_only_r": True,  "finite": True},
    "TD-CCP":   {"linear_r": True,  "neural_r": True,  "linear_phi": True,  "neural_phi": True,  "action_dep_r": True, "state_only_r": False, "finite": True},
    "SEES":     {"linear_r": True,  "neural_r": False, "linear_phi": True,  "neural_phi": True,  "action_dep_r": True, "state_only_r": True,  "finite": True},
    "MCE-IRL":  {"linear_r": True,  "neural_r": True,  "linear_phi": True,  "neural_phi": True,  "action_dep_r": True, "state_only_r": False, "finite": True},
    "AIRL":     {"linear_r": False, "neural_r": True,  "linear_phi": False, "neural_phi": True,  "action_dep_r": False, "state_only_r": True, "finite": True},
    "GLADIUS":  {"linear_r": False, "neural_r": True,  "linear_phi": False, "neural_phi": True,  "action_dep_r": True, "state_only_r": True,  "finite": True},
    "f-IRL":    {"linear_r": True,  "neural_r": False, "linear_phi": True,  "neural_phi": False, "action_dep_r": True, "state_only_r": True,  "finite": True},
    "IQ-Learn": {"linear_r": True,  "neural_r": True,  "linear_phi": True,  "neural_phi": True,  "action_dep_r": True, "state_only_r": True,  "finite": True},
    "BC":       {"linear_r": False, "neural_r": False, "linear_phi": True,  "neural_phi": True,  "action_dep_r": True, "state_only_r": True,  "finite": True},
}


def _load_headline(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Map (estimator, ss-cell-suffix) -> aggregated row dict."""
    table: dict[tuple[str, str], dict[str, str]] = {}
    if not path.exists():
        return table
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = row.get("cell_id", "")
            est = row.get("estimator", "")
            if not cell_id.startswith("tier4_"):
                continue
            # cell_id format: tier4_<suffix>_<estimator_slug>
            # The suffix may itself contain underscores (e.g. "neural_r_phi").
            stripped = cell_id[len("tier4_") :]
            est_slug = est.lower().replace("-", "").replace("_", "")
            if stripped.endswith(f"_{est_slug}"):
                suffix = stripped[: -(len(est_slug) + 1)]
            else:
                suffix = stripped
            table[(est, suffix)] = row
    return table


def _cell_mark(
    estimator: str,
    column: str,
    headline: dict[tuple[str, str], dict[str, str]],
) -> str:
    """Return the LaTeX cell content for one (estimator, column) cell."""
    if not _THEORETICAL_SUPPORT.get(estimator, {}).get(column, False):
        return "na"
    suffix = _COLUMN_TO_CELL[column]
    row = headline.get((estimator, suffix))
    if row is None:
        return "pending"
    failure = row.get("failure_mode_majority", "none")
    try:
        conv_rate = float(row.get("convergence_rate", "0"))
    except ValueError:
        conv_rate = 0.0
    if failure == "none" and conv_rate >= _CONVERGENCE_THRESHOLD:
        return "\\checkmark"
    return "$\\times$"


def emit_table(headline_csv: Path, output_tex: Path) -> None:
    headline = _load_headline(headline_csv)
    columns = list(_COLUMN_TO_CELL.keys())
    column_labels = [
        "Linear $r$", "Neural $r$", "Linear $\\phi$", "Neural $\\phi$",
        "Action-dep $r$", "State-only $r$", "Finite horizon",
    ]

    lines = []
    lines.append("% Auto-generated by experiments/jss_deep_run/emit_capability_table.py")
    lines.append("% Do not edit by hand. Refresh after each Tier 4 dispatch.")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l c c c c c c c}")
    lines.append("\\toprule")
    lines.append(
        "Estimator & " + " & ".join(column_labels) + " \\\\"
    )
    lines.append("\\midrule")

    for fi, family in enumerate(_ROWS_BY_FAMILY):
        if fi > 0:
            lines.append("\\midrule")
        for display_name, est in family:
            cells = [_cell_mark(est, col, headline) for col in columns]
            lines.append(f"{display_name:<10} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headline-csv",
        type=Path,
        default=Path("experiments/jss_deep_run/results/shapeshifter/headline.csv"),
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("papers/econirl_package/tables/capability_empirical.tex"),
    )
    args = parser.parse_args()

    emit_table(args.headline_csv, args.output_tex)
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
