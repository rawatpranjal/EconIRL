#!/usr/bin/env python3
"""Monte-Carlo parameter-recovery driver for the MPEC estimator.

Thin Tier-1 wrapper around ``validation/estimators/_recovery_mc.py``.
Re-simulates the canonical_low_action known-truth panel with a fresh seed
each replication, re-fits MPEC, and collects the sampling distribution of the
4 recovered structural parameters. Produces a caterpillar figure and a JSON.

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python validation/estimators/mpec/recovery_mc.py --n-reps 200

    # fast smoke test
    PYTHONPATH=src:. python validation/estimators/mpec/recovery_mc.py --n-reps 3
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

from validation.estimators._recovery_mc import run_tier1  # noqa: E402

CELL_ID = "canonical_low_action"
ESTIMATOR = "MPEC"
JSON_OUT = ROOT / "validation" / "results" / "mpec_recovery.json"
FIG_OUT = ROOT / "docs" / "_static" / "estimators" / "mpec_recovery.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    run_tier1(
        estimator=ESTIMATOR,
        cell_id=CELL_ID,
        json_out=JSON_OUT,
        fig_out=FIG_OUT,
        n_reps=args.n_reps,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
