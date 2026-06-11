"""Regenerate the simulation-studies evidence: results JSON, figures, tables.

Single source of truth for the public ``docs/simulation_studies.md`` page.
Runs the cross-estimator benchmark over the cell ladder, then writes:

- ``validation/results/simulation_studies.json`` (machine-readable, stamped),
- ``docs/_static/simulation_studies/*.png`` (four summary figures),
- ``docs/_generated/simulation_studies_results.md`` (segmented tables + failure map).

This is a heavy compute job (many estimator fits). Use ``--replications`` to
trade Monte Carlo precision for wall-clock, and ``--quick`` for a fast check.

Usage:
    python scripts/run_simulation_studies.py --replications 20
    python scripts/run_simulation_studies.py --quick
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os

# Repo root on sys.path so ``validation`` imports work when run as a script.
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.benchmark import plots, report  # noqa: E402
from validation.benchmark.cells import CELLS  # noqa: E402
from validation.benchmark.runner import (  # noqa: E402
    DEFAULT_ESTIMATORS,
    EstimatorSpec,
    _run_ccp,
    _run_nfxp,
    run_benchmark,
)

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "simulation_studies.json")
FIG_DIR = os.path.join(_ROOT, "docs", "_static", "simulation_studies")
TABLES_MD = os.path.join(_ROOT, "docs", "_generated", "simulation_studies_results.md")


def _package_version() -> str:
    try:
        import econirl

        return getattr(econirl, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=20,
                        help="Monte Carlo replications per cell (default 20).")
    parser.add_argument("--quick", action="store_true",
                        help="Fast check: 2 cells, 2 estimators, 3 replications.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated cell ids to (re)run and merge into "
                             "the existing JSON, leaving other cells untouched.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    args = parser.parse_args()

    date = _dt.date.today().isoformat()
    version = _package_version()

    if args.quick:
        cells = tuple(c for c in CELLS if c.cell_id in ("simple_binary", "high_beta"))
        estimators = (
            EstimatorSpec("NFXP", "structural", _run_nfxp),
            EstimatorSpec("CCP", "structural", _run_ccp),
        )
        n_reps = 3
    elif args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cells = tuple(c for c in CELLS if c.cell_id in wanted)
        estimators = DEFAULT_ESTIMATORS
        n_reps = args.replications
    else:
        cells = CELLS
        estimators = DEFAULT_ESTIMATORS
        n_reps = args.replications

    print(f"Running benchmark: {len(cells)} cells x {len(estimators)} estimators "
          f"x {n_reps} replications. date={date} version={version}")
    result = run_benchmark(
        cells=cells, estimators=estimators, n_replications=n_reps,
        date=date, package_version=version, verbose=args.verbose,
    )

    # Merge mode: replace only the requested cells in the existing JSON.
    if args.only and os.path.exists(RESULTS_JSON):
        existing = json.load(open(RESULTS_JSON))
        new_by_id = {c["cell_id"]: c for c in result["cells"]}
        merged = [new_by_id.get(c["cell_id"], c) for c in existing["cells"]]
        for c in result["cells"]:
            if c["cell_id"] not in {e["cell_id"] for e in existing["cells"]}:
                merged.append(c)
        existing["cells"] = merged
        existing["meta"]["date"] = date
        result = existing

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {RESULTS_JSON}")

    written = plots.generate_all(result, FIG_DIR)
    print(f"Wrote {len(written)} figures to {FIG_DIR}")

    os.makedirs(os.path.dirname(TABLES_MD), exist_ok=True)
    with open(TABLES_MD, "w") as f:
        f.write(report.render_results_markdown(result))
    print(f"Wrote {TABLES_MD}")


if __name__ == "__main__":
    main()
