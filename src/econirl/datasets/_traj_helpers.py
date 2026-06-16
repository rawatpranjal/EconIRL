"""Shared coordinate-trajectory helpers for the mobility datasets.

tdrive, geolife, eth_ucy, and stanford_drone all discretize raw coordinates
onto a grid and group points into per-agent trajectories. Only the column
names differ, so the logic lives here once and each loader passes its own.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def discretize_coords(
    df: pd.DataFrame, grid_size: int, coord_cols: tuple[str, str]
) -> pd.DataFrame:
    """Map two continuous coordinate columns onto a grid_size x grid_size grid.

    Writes a single flat ``state`` index = ``second_idx * grid_size + first_idx``.
    """
    x_col, y_col = coord_cols
    df = df.copy()
    x_bins = np.linspace(df[x_col].min(), df[x_col].max(), grid_size + 1)
    y_bins = np.linspace(df[y_col].min(), df[y_col].max(), grid_size + 1)
    x_idx = np.clip(np.digitize(df[x_col], x_bins) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(df[y_col], y_bins) - 1, 0, grid_size - 1)
    df["state"] = y_idx * grid_size + x_idx
    return df


def to_trajectories(
    df: pd.DataFrame,
    has_states: bool,
    id_col: str,
    sort_col: str,
    coord_cols: tuple[str, str],
) -> List[np.ndarray]:
    """Group rows by ``id_col`` (time-ordered by ``sort_col``) into per-agent
    arrays: discrete ``state`` values if ``has_states``, else the raw coords."""
    trajectories = []
    for agent_id in df[id_col].unique():
        data = df[df[id_col] == agent_id].sort_values(sort_col)
        if has_states:
            traj = data["state"].values
        else:
            traj = data[list(coord_cols)].values
        trajectories.append(traj)
    return trajectories
