"""High-dimensional Rust bus engine dataset.

This module wraps the canonical Rust (1987) bus engine panel and appends
a configurable number of dummy state variables drawn IID independent of
the action. The dummies are noise by construction so the true reward
depends only on the original mileage state. The augmented dataset is
intended to break tabular value iteration while staying tractable for
neural function approximation.

The construction follows Kang, Yoganarasimhan, and Jain (2025), who use
this scaling test to demonstrate that GLADIUS scales where tabular
methods do not.

Reference:
    Kang, E. H., Yoganarasimhan, H., and Jain, L. (2025).
    "An Empirical Risk Minimization Approach for Offline Inverse RL
    and Dynamic Discrete Choice Model." arXiv:2502.14131.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from econirl.datasets.rust_bus import load_rust_bus


def load_rust_big(
    n_dummies: int = 30,
    dummy_cardinality: int = 20,
    seed: int = 42,
    as_panel: bool = False,
) -> pd.DataFrame:
    """Load the Rust bus panel augmented with dummy state variables.

    The mileage column from the original panel is preserved exactly so
    that any estimator that projects onto mileage recovers the same
    parameters as on `rust-small`. The dummies are appended as columns
    `dummy_0`, `dummy_1`, ..., each drawn uniformly over `0` to
    `dummy_cardinality - 1` independently of the action.

    Args:
        n_dummies: Number of dummy state variables to append.
            Default 30 matches the middle test value in Kang et al.\
            (2025). Set to zero to recover `rust-small`.
        dummy_cardinality: Number of values each dummy can take.
            Default 20 matches Kang et al.\\ (2025).
        seed: Random seed for the dummy draws.
        as_panel: If True, return a Panel object whose state column
            is the original mileage bin and whose metadata declares
            the dummy column names. If False, return the augmented
            DataFrame.

    Returns:
        DataFrame with the original Rust columns plus `n_dummies`
        dummy columns. If `as_panel=True`, a Panel object whose
        `metadata['dummy_columns']` lists the dummy names.

    Notes:
        The Panel object encodes only the mileage state as the integer
        state column to keep it compatible with tabular estimators that
        expect a small state space. The dummy columns travel in the
        Panel `metadata` field. Estimators that consume the augmented
        state space (NNES, GLADIUS, TD-CCP) read the dummies through
        a multi-dimensional state encoder rather than through the
        primary state column.
    """
    if n_dummies < 0:
        raise ValueError(f"n_dummies must be non-negative, got {n_dummies}")
    if dummy_cardinality < 1:
        raise ValueError(
            f"dummy_cardinality must be positive, got {dummy_cardinality}"
        )

    df = load_rust_bus().copy()
    rng = np.random.default_rng(seed)
    dummy_columns = [f"dummy_{k}" for k in range(n_dummies)]
    if n_dummies > 0:
        dummies = rng.integers(
            low=0, high=dummy_cardinality, size=(len(df), n_dummies)
        )
        for k, col in enumerate(dummy_columns):
            df[col] = dummies[:, k]

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import jax.numpy as jnp

        bus_ids = df["bus_id"].unique()
        trajectories = []
        for bus_id in bus_ids:
            bus_data = df[df["bus_id"] == bus_id].sort_values("period")
            states = jnp.array(bus_data["mileage_bin"].values, dtype=jnp.int32)
            actions = jnp.array(bus_data["replaced"].values, dtype=jnp.int32)
            next_states = jnp.concatenate([states[1:], jnp.array([0])])
            traj = Trajectory(
                states=states,
                actions=actions,
                next_states=next_states,
                individual_id=int(bus_id),
            )
            trajectories.append(traj)

        return Panel(
            trajectories=trajectories,
            metadata={
                "dummy_columns": dummy_columns,
                "dummy_cardinality": dummy_cardinality,
                "n_dummies": n_dummies,
                "primary_state_column": "mileage_bin",
                "n_mileage_bins": 90,
                "augmented_state_dim": 1 + n_dummies,
            },
        )

    return df


def get_rust_big_info(n_dummies: int = 30, dummy_cardinality: int = 20) -> dict:
    """Return metadata about the augmented Rust bus dataset.

    Args:
        n_dummies: Number of dummy state variables in the configuration.
        dummy_cardinality: Cardinality of each dummy state variable.

    Returns:
        Dictionary with dataset information including the augmented
        state-space dimensionality, the source paper, and the true
        reward parameters on the genuine mileage dimension.
    """
    return {
        "name": "Rust (1987) bus engine, high-dimensional augmentation",
        "n_dummies": n_dummies,
        "dummy_cardinality": dummy_cardinality,
        "augmented_state_dim": 1 + n_dummies,
        "augmented_state_cardinality": 90 * (dummy_cardinality**n_dummies),
        "primary_state_dim": 1,
        "primary_state_cardinality": 90,
        "n_actions": 2,
        "true_parameters": {"theta_1": 0.001, "RC": 3.0},
        "discount_factor": 0.9999,
        "ground_truth": True,
        "use_case": "Neural function approximation scaling benchmark",
        "reference": (
            "Kang, E. H., Yoganarasimhan, H., and Jain, L. (2025). "
            "An Empirical Risk Minimization Approach for Offline Inverse RL "
            "and Dynamic Discrete Choice Model. arXiv:2502.14131."
        ),
    }
