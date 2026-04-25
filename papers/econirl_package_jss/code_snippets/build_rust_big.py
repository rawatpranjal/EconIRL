"""Driver script for the rust-big dataset.

Calls `econirl.datasets.load_rust_big` with the JSS paper defaults
(`n_dummies=30`, `dummy_cardinality=20`, `seed=42`) and prints the
augmented state-space dimensionality. The loader holds the data in
memory and does not write a CSV because the dummies are generated
on demand from the bundled `rust-small` panel.

The script writes a one-line confirmation to
`papers/econirl_package_jss/figures/rust_big_built.txt` so the LaTeX
build can verify it ran.
"""

from __future__ import annotations

from pathlib import Path

from econirl.datasets import load_rust_big, get_rust_big_info


def main() -> None:
    panel = load_rust_big(
        n_dummies=30,
        dummy_cardinality=20,
        seed=42,
        as_panel=True,
    )
    info = get_rust_big_info(n_dummies=30, dummy_cardinality=20)

    print(f"rust-big augmented state dimensionality: {info['augmented_state_dim']}")
    print(f"rust-big primary cardinality: {info['primary_state_cardinality']}")
    print(f"rust-big number of trajectories: {len(panel.trajectories)}")
    print(f"rust-big number of observations: {panel.num_observations}")
    print(f"rust-big number of dummy columns: {info['n_dummies']}")

    out_path = (
        Path(__file__).resolve().parent.parent / "figures" / "rust_big_built.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"rust-big built with n_dummies=30 dummy_cardinality=20 seed=42 "
        f"observations={panel.num_observations} trajectories={len(panel.trajectories)}\n"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
