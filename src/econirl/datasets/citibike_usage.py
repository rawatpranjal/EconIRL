"""Citibike daily usage frequency dataset for transportation DDC.

This module loads preprocessed Citibike member-day panel data for
usage frequency estimation. Members decide each day whether to take
a bikeshare trip. The data must first be downloaded and preprocessed
using scripts/download_citibike.py.

If the processed data is not available, the loader falls back to
generating synthetic usage panels from the CitibikeUsageEnvironment.

State space:
    n_day_types x n_usage_buckets discrete states.
    Default: 2 day types x 4 usage buckets = 8 states.

Action space:
    2 actions: no ride (0) and ride (1).

Reference:
    Citibike System Data (NYC): https://citibikenyc.com/system-data
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from econirl.core.types import Panel, Trajectory
from econirl.environments.citibike_usage import (
    N_ACTIONS,
    N_FEATURES,
    N_STATES,
)


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed" / "citibike_usage.csv"


def load_citibike_usage(
    as_panel: bool = False,
    data_path: str | Path | None = None,
    n_individuals: int = 500,
    n_periods: int = 90,
    seed: int = 42,
) -> Union[pd.DataFrame, Panel]:
    """Load Citibike daily usage frequency data.

    If preprocessed data exists (from scripts/download_citibike.py),
    loads it directly. Otherwise generates synthetic usage panels
    from the CitibikeUsageEnvironment.

    Args:
        as_panel: If True, return Panel object for econirl estimators.
        data_path: Path to citibike_usage.csv. If None, checks default
            location then falls back to synthetic generation.
        n_individuals: Number of members for synthetic fallback.
        n_periods: Number of days per member for synthetic fallback.
        seed: Random seed for synthetic fallback.

    Returns:
        DataFrame with daily usage data. If as_panel=True, returns
        Panel object.
    """
    if data_path is not None:
        data_path = Path(data_path)
    else:
        data_path = DEFAULT_DATA_PATH

    if data_path.exists():
        df = pd.read_csv(data_path)
        if as_panel:
            return _dataframe_to_panel(df)
        return df
    else:
        print(
            f"Citibike usage data not found at {data_path}. "
            "Generating synthetic data from CitibikeUsageEnvironment. "
            "Run 'python scripts/download_citibike.py' to download real data."
        )
        return _generate_synthetic(as_panel, n_individuals, n_periods, seed)


def _generate_synthetic(
    as_panel: bool, n_individuals: int, n_periods: int, seed: int
) -> Union[pd.DataFrame, Panel]:
    """Generate synthetic usage frequency data as fallback."""
    from econirl.environments.citibike_usage import (
        CitibikeUsageEnvironment,
        DAY_LABELS,
        USAGE_LABELS,
        state_to_components,
    )
    from econirl.simulation.synthetic import simulate_panel

    env = CitibikeUsageEnvironment(seed=seed)
    panel = simulate_panel(env, n_individuals=n_individuals, n_periods=n_periods, seed=seed)

    if as_panel:
        return panel

    records = []
    for traj in panel.trajectories:
        tid = traj.individual_id
        for t in range(len(traj.states)):
            s = int(traj.states[t])
            a = int(traj.actions[t])
            ns = int(traj.next_states[t])
            dt, ub = state_to_components(s)
            records.append({
                "rider_id": tid,
                "day": t,
                "state": s,
                "action": a,
                "next_state": ns,
                "day_type": dt,
                "usage_bucket": ub,
                "day_label": DAY_LABELS[dt],
                "usage_label": USAGE_LABELS[ub],
                "rode": a == 1,
            })

    return pd.DataFrame(records)


def _dataframe_to_panel(df: pd.DataFrame) -> Panel:
    """Convert usage DataFrame to Panel."""
    trajectories = []
    rider_col = "rider_id"
    for rid, group in df.groupby(rider_col):
        group = group.sort_values("date" if "date" in group.columns else "day")
        trajectories.append(
            Trajectory(
                individual_id=int(rid) if isinstance(rid, (int, float)) else hash(rid) % 100000,
                states=np.array(group["state"].values, dtype=np.int32),
                actions=np.array(group["action"].values, dtype=np.int32),
                next_states=np.array(group["next_state"].values, dtype=np.int32),
            )
        )
    return Panel(trajectories=trajectories)


def get_citibike_usage_info() -> dict:
    """Return metadata about the Citibike usage frequency dataset."""
    return {
        "name": "Citibike Daily Usage Frequency",
        "description": (
            "NYC Citibike member daily ride/no-ride decisions. "
            "8 states (day type x recent usage bucket), "
            "2 actions (ride/no ride). Real data requires "
            "running scripts/download_citibike.py; falls back to "
            "synthetic generation."
        ),
        "source": "https://citibikenyc.com/system-data",
        "license": "Non-commercial research",
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
        "n_features": N_FEATURES,
        "state_description": "Day type (weekday/weekend) x recent usage bucket",
        "action_description": "No ride (0) / Ride (1)",
        "ground_truth": False,
        "use_case": "Transportation DDC, labor supply, habitual behavior, usage frequency",
    }
