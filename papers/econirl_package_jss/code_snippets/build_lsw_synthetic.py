"""Driver script for the lsw-synthetic dataset.

Calls `econirl.datasets.load_lsw_synthetic` with the JSS paper defaults
(`n_users=5000`, `n_books=50`, `n_chapters_per_book=30`,
`pi_pay_segment=0.4`, `seed=42`) and writes the resulting CSV plus
metadata JSON to `src/econirl/datasets/lsw_synthetic_data.csv` and
`src/econirl/datasets/lsw_synthetic_metadata.json`.

The script prints the simulated marginal purchase rate so the build
log surfaces whether the calibration check passed. The calibration
target is six percent with a tolerance window of plus or minus three
percentage points, matching the simulated mirror of the marginal
reported in Lee, Sudhir, and Wang 2026.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from econirl.datasets.lsw_synthetic import (
    _DEFAULT_THETA,
    _generate_lsw_synthetic,
    load_lsw_synthetic,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = repo_root / "src" / "econirl" / "datasets" / "lsw_synthetic_data.csv"
    meta_path = (
        repo_root / "src" / "econirl" / "datasets" / "lsw_synthetic_metadata.json"
    )

    start = time.time()
    df, metadata = _generate_lsw_synthetic(
        n_users=5000,
        n_books=50,
        n_chapters_per_book=30,
        pi_pay_segment=0.4,
        seed=42,
        theta=_DEFAULT_THETA,
    )
    elapsed = time.time() - start

    df.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"lsw-synthetic users: {metadata['n_users']}")
    print(f"lsw-synthetic rows: {len(df)}")
    print(f"lsw-synthetic states: {metadata['n_states']}")
    print(f"lsw-synthetic actions: {metadata['n_actions']}")
    print(f"lsw-synthetic mixture weights (pay, wait): "
          f"{metadata['pi_pay_segment']:.2f}, {metadata['pi_wait_segment']:.2f}")
    print(f"lsw-synthetic simulated purchase rate: "
          f"{metadata['simulated_purchase_rate']:.4f} "
          f"(target {metadata['purchase_rate_target']:.2f} "
          f"+/- {metadata['purchase_rate_tol']:.2f})")
    print(f"lsw-synthetic wall-clock time: {elapsed:.1f}s")
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")

    # Sanity check the loader can read the cached files back.
    panel = load_lsw_synthetic(as_panel=True)
    print(f"lsw-synthetic loader returned panel with {panel.num_observations} obs "
          f"and {panel.num_individuals} trajectories")


if __name__ == "__main__":
    main()
