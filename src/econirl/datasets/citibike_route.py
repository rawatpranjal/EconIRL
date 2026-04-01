"""Citibike route choice dataset for destination IRL.

This module loads preprocessed Citibike trip data for route choice
estimation. Riders choose a destination station cluster given their
origin cluster and time of day. The data must first be downloaded
and preprocessed using scripts/download_citibike.py.

If the processed data is not available, the loader falls back to
generating synthetic route choice trajectories from the
CitibikeRouteEnvironment with default parameters.

State space:
    n_clusters x n_time_buckets discrete states.
    Default: 20 station clusters x 4 time buckets = 80 states.

Action space:
    n_clusters destination choices.
    Default: 20 destination clusters.

Reference:
    Citibike System Data (NYC): https://citibikenyc.com/system-data
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from econirl.core.types import Panel, Trajectory
from econirl.environments.citibike_route import (
    N_ACTIONS,
    N_FEATURES,
    N_STATES,
)


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed" / "citibike_route.csv"


def load_citibike_route(
    as_panel: bool = False,
    data_path: str | Path | None = None,
    n_individuals: int = 1000,
    n_periods: int = 50,
    seed: int = 42,
) -> Union[pd.DataFrame, Panel]:
    """Load Citibike route choice data.

    If preprocessed data exists (from scripts/download_citibike.py),
    loads it directly. Otherwise generates synthetic route choice
    trajectories from the CitibikeRouteEnvironment.

    Args:
        as_panel: If True, return Panel object for econirl estimators.
        data_path: Path to citibike_route.csv. If None, checks default
            location then falls back to synthetic generation.
        n_individuals: Number of riders for synthetic fallback.
        n_periods: Number of trips per rider for synthetic fallback.
        seed: Random seed for synthetic fallback.

    Returns:
        DataFrame with route choice data. If as_panel=True, returns
        Panel object.
    """
    if data_path is not None:
        data_path = Path(data_path)
    else:
        data_path = DEFAULT_DATA_PATH

    if data_path.exists():
        df = pd.read_csv(data_path)
        if "next_state" not in df.columns:
            # Build next_state from sequential trips
            df = df.sort_values(["trip_idx"])
            df["next_state"] = df["state"].shift(-1)
            df = df.dropna(subset=["next_state"])
            df["next_state"] = df["next_state"].astype(int)

        if as_panel:
            return _dataframe_to_panel(df)
        return df
    else:
        print(
            f"Citibike data not found at {data_path}. "
            "Generating synthetic data from CitibikeRouteEnvironment. "
            "Run 'python scripts/download_citibike.py' to download real data."
        )
        return _generate_synthetic(as_panel, n_individuals, n_periods, seed)


def _generate_synthetic(
    as_panel: bool, n_individuals: int, n_periods: int, seed: int
) -> Union[pd.DataFrame, Panel]:
    """Generate synthetic route choice data as fallback."""
    from econirl.environments.citibike_route import (
        CitibikeRouteEnvironment,
        state_to_components,
    )
    from econirl.simulation.synthetic import simulate_panel

    env = CitibikeRouteEnvironment(seed=seed)
    panel = simulate_panel(env, n_individuals=n_individuals, n_periods=n_periods, seed=seed)

    if as_panel:
        return panel

    records = []
    for traj in panel.trajectories:
        for t in range(len(traj.states)):
            s = int(traj.states[t])
            a = int(traj.actions[t])
            ns = int(traj.next_states[t])
            oc, tb = state_to_components(s)
            records.append({
                "trip_idx": len(records),
                "state": s,
                "action": a,
                "next_state": ns,
                "origin_cluster": oc,
                "dest_cluster": a,
                "time_bucket": tb,
            })

    return pd.DataFrame(records)


def _dataframe_to_panel(df: pd.DataFrame) -> Panel:
    """Convert route choice DataFrame to Panel by chunking trips."""
    # Group sequential trips into pseudo-individuals
    chunk_size = 50
    trajectories = []
    n_chunks = len(df) // chunk_size

    for i in range(n_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        trajectories.append(
            Trajectory(
                individual_id=i,
                states=np.array(chunk["state"].values, dtype=np.int32),
                actions=np.array(chunk["action"].values, dtype=np.int32),
                next_states=np.array(chunk["next_state"].values, dtype=np.int32),
            )
        )

    return Panel(trajectories=trajectories)


def get_citibike_route_info() -> dict:
    """Return metadata about the Citibike route choice dataset."""
    return {
        "name": "Citibike Route Choice",
        "description": (
            "NYC Citibike station-to-station destination choice. "
            "80 states (20 station clusters x 4 time buckets), "
            "20 actions (destination clusters). Real data requires "
            "running scripts/download_citibike.py; falls back to "
            "synthetic generation."
        ),
        "source": "https://citibikenyc.com/system-data",
        "license": "Non-commercial research",
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
        "n_features": N_FEATURES,
        "state_description": "Origin station cluster x time-of-day bucket",
        "action_description": "Destination station cluster (0-19)",
        "ground_truth": False,
        "use_case": "Route choice IRL, urban mobility, transportation planning",
    }
