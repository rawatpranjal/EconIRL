"""First-stage transition probability estimation.

Estimates the mileage transition probabilities theta = (theta_0, theta_1, theta_2) from data.
These represent P(mileage increases by k bins | keep engine).

Reference:
    Rust (1987), Section 4.1, Table IV
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def estimate_transition_probs(
    df: pd.DataFrame,
    max_increment: int = 2,
    mileage_col: str = "mileage_bin",
    action_col: str = "replaced",
    bus_col: str = "bus_id",
    period_col: str = "period",
) -> np.ndarray:
    """Estimate mileage transition probabilities from panel data.

    Uses observations where the engine was NOT replaced to estimate
    the distribution of mileage increments.

    P(delta_x = k) = #{transitions with increment k} / #{total transitions}

    Args:
        df: Panel data with mileage and replacement decisions
        max_increment: Maximum mileage increment to consider (default 2)
        mileage_col: Column name for mileage bin
        action_col: Column name for replacement decision
        bus_col: Column name for bus identifier
        period_col: Column name for time period

    Returns:
        Array of probabilities [P(delta_x=0), P(delta_x=1), P(delta_x=2)]

    Example:
        >>> df = load_rust_bus(group=4)
        >>> probs = estimate_transition_probs(df)
        >>> print(f"theta = ({probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f})")
    """
    # Sort by bus and period
    df = df.sort_values([bus_col, period_col]).copy()

    # Compute mileage increments for non-replacement periods
    increments = []

    for bus_id in df[bus_col].unique():
        bus_data = df[df[bus_col] == bus_id].sort_values(period_col)

        for i in range(len(bus_data) - 1):
            current = bus_data.iloc[i]
            next_obs = bus_data.iloc[i + 1]

            # Only use transitions where engine was NOT replaced
            if current[action_col] == 0:
                # And next period is consecutive
                if next_obs[period_col] == current[period_col] + 1:
                    increment = next_obs[mileage_col] - current[mileage_col]
                    # Clamp to valid range (could be negative due to measurement error)
                    increment = max(0, min(increment, max_increment))
                    increments.append(increment)

    if len(increments) == 0:
        # Return uniform if no valid transitions
        return np.ones(max_increment + 1) / (max_increment + 1)

    # Count increments
    counts = np.zeros(max_increment + 1)
    for inc in increments:
        counts[int(inc)] += 1

    # Normalize to probabilities
    probs = counts / counts.sum()

    return probs


def estimate_transition_probs_by_group(
    df: pd.DataFrame,
    group_col: str = "group",
    **kwargs,
) -> dict[int, np.ndarray]:
    """Estimate transition probabilities separately for each bus group.

    Args:
        df: Panel data with group identifier
        group_col: Column name for group
        **kwargs: Additional arguments passed to estimate_transition_probs

    Returns:
        Dictionary mapping group ID to probability array

    Example:
        >>> df = load_rust_bus()
        >>> probs_by_group = estimate_transition_probs_by_group(df)
        >>> for g, p in probs_by_group.items():
        ...     print(f"Group {g}: theta = {p}")
    """
    result = {}
    for group in sorted(df[group_col].unique()):
        group_df = df[df[group_col] == group]
        result[group] = estimate_transition_probs(group_df, **kwargs)
    return result
