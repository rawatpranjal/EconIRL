"""Driver script for the ziebart-big dataset.

Calls `econirl.datasets.load_ziebart_big` with the JSS paper defaults
(`grid_size=50`, `n_actions=8`, `slip_prob=0.1`, `n_basis=16`,
`n_trajectories=50000`, `trajectory_length=30`, `seed=42`) and writes
the resulting CSV plus metadata JSON to
`src/econirl/datasets/ziebart_big_data.csv` and
`src/econirl/datasets/ziebart_big_metadata.json`.

The script prints the wall-clock time and the number of rows so the
build log surfaces the cost of regeneration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from econirl.datasets import load_ziebart_big, get_ziebart_big_info
from econirl.datasets.ziebart_big import _generate_ziebart_big_data


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = repo_root / "src" / "econirl" / "datasets" / "ziebart_big_data.csv"
    meta_path = (
        repo_root / "src" / "econirl" / "datasets" / "ziebart_big_metadata.json"
    )

    start = time.time()
    df, metadata = _generate_ziebart_big_data(
        grid_size=50,
        n_actions=8,
        slip_prob=0.1,
        n_basis=16,
        n_trajectories=50000,
        trajectory_length=30,
        discount_factor=0.99,
        scale_parameter=1.0,
        seed=42,
    )
    elapsed = time.time() - start

    df.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"ziebart-big trajectories: {metadata['n_trajectories']}")
    print(f"ziebart-big rows: {len(df)}")
    print(f"ziebart-big states: {metadata['n_states']}")
    print(f"ziebart-big actions: {metadata['n_actions']}")
    print(f"ziebart-big basis dimension: {metadata['n_basis']}")
    print(f"ziebart-big wall-clock time: {elapsed:.1f}s")
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")

    # Sanity check the loader can read the cached files back.
    panel = load_ziebart_big(as_panel=True)
    print(f"ziebart-big loader returned panel with {panel.num_observations} obs")


if __name__ == "__main__":
    main()
