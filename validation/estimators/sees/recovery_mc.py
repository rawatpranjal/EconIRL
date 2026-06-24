#!/usr/bin/env python3
"""Monte-Carlo parameter-recovery driver for the SEES estimator.

Thin Tier-2 (high-dim compact) wrapper around
``validation/estimators/_recovery_mc.py``.
Re-simulates the canonical_high_action known-truth panel with a fresh seed
each replication, re-fits SEES, and collects the sampling distribution across
all 32 value-function-weight params. Stores full per-param stats in the JSON
(reproducibility) and renders a compact error-distribution histogram.

The JSON additionally stores:
  * ``recovery_rmse``   -- sqrt mean of (mean_estimate - true)^2 across params
  * ``coverage_frac``   -- fraction of params whose truth falls in their 95% CI

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python validation/estimators/sees/recovery_mc.py --n-reps 200

    # fast smoke test
    PYTHONPATH=src:. python validation/estimators/sees/recovery_mc.py --n-reps 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validation.estimators._recovery_mc import run_tier2  # noqa: E402

CELL_ID = "canonical_high_action"
ESTIMATOR = "SEES"
JSON_OUT = ROOT / "validation" / "results" / "sees_recovery.json"
FIG_OUT = ROOT / "docs" / "_static" / "estimators" / "sees_recovery.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    run_tier2(
        estimator=ESTIMATOR,
        cell_id=CELL_ID,
        json_out=JSON_OUT,
        fig_out=FIG_OUT,
        n_reps=args.n_reps,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
