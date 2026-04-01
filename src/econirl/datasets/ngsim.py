"""
NGSIM US-101 Vehicle Trajectory Dataset for Lane-Change IRL.

This module provides the NGSIM US-101 highway vehicle trajectory dataset,
modeling lane-change decisions as a dynamic discrete choice problem.

Each vehicle's trajectory is treated as a panel observation where:
- State: (lane, speed_bin) — which lane and how fast
- Action: lane change decision (left, stay, right)
- Transitions: empirical lane/speed dynamics

Reference:
    FHWA (2006). "Next Generation Simulation (NGSIM) Vehicle Trajectories
    and Supporting Data." U.S. Department of Transportation.
    https://datahub.transportation.gov/
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


N_LANES = 5  # Mainline lanes 1-5 (drop ramps 6-8)
N_SPEED_BINS = 10
SPEED_BIN_WIDTH = 5.0  # ft/s per bin (~3.4 mph)
ACTION_LEFT = 0
ACTION_STAY = 1
ACTION_RIGHT = 2
N_ACTIONS = 3
LANE_NAMES = ["Lane 1 (leftmost)", "Lane 2", "Lane 3 (center)", "Lane 4", "Lane 5 (rightmost)"]
ACTION_NAMES = ["Lane Left", "Stay", "Lane Right"]


def load_ngsim(
    as_panel: bool = False,
    n_speed_bins: int = N_SPEED_BINS,
    subsample_frames: int = 10,
    min_frames: int = 50,
    max_vehicles: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the NGSIM US-101 dataset as a lane-change discrete choice problem.

    Args:
        as_panel: If True, return as Panel object for econirl estimators.
        n_speed_bins: Number of speed bins (default 10, covering 0-50 ft/s).
        subsample_frames: Take every Nth frame to reduce autocorrelation
            (default 10, i.e., 1Hz from 10Hz raw data).
        min_frames: Minimum frames per vehicle after subsampling (default 50).
        max_vehicles: If set, limit to this many vehicles (for faster testing).

    Returns:
        DataFrame with columns: vehicle_id, frame, state, action, next_state,
        lane, speed_bin, v_vel, space_headway, lane_change
    """
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "ngsim" / "us101_trajectories.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"NGSIM data not found at {data_path}. "
            "Download from: https://datahub.transportation.gov/resource/8ect-6jqj.csv"
        )

    # Load with only needed columns for memory efficiency
    usecols = ["vehicle_id", "frame_id", "v_vel", "v_acc", "lane_id",
               "space_headway", "time_headway"]
    df = pd.read_csv(data_path, usecols=usecols, dtype={
        "vehicle_id": "int32", "frame_id": "int32",
        "lane_id": "int8", "v_vel": "float32", "v_acc": "float32",
        "space_headway": "float32", "time_headway": "float32",
    })

    # Strip quotes from column values if present
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.strip('"'), errors="coerce")

    # Filter to mainline lanes only (1-5)
    df = df[df["lane_id"].between(1, N_LANES)].copy()

    # Sort by vehicle and frame
    df = df.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)

    # Subsample frames (10Hz → 1Hz by default)
    if subsample_frames > 1:
        df = df.groupby("vehicle_id", group_keys=False).apply(
            lambda x: x.iloc[::subsample_frames]
        ).reset_index(drop=True)

    # Discretize speed
    df["speed_bin"] = np.clip(
        (df["v_vel"] / SPEED_BIN_WIDTH).astype(int),
        0, n_speed_bins - 1
    )

    # Compute lane (0-indexed)
    df["lane"] = (df["lane_id"] - 1).astype(int)

    # Compute state: lane * n_speed_bins + speed_bin
    df["state"] = df["lane"] * n_speed_bins + df["speed_bin"]

    # Detect lane changes (action) from consecutive frames
    df["next_lane"] = df.groupby("vehicle_id")["lane"].shift(-1)
    df["next_speed_bin"] = df.groupby("vehicle_id")["speed_bin"].shift(-1)
    df["lane_change"] = df["next_lane"] - df["lane"]

    # Map lane change to action
    df["action"] = ACTION_STAY  # default
    df.loc[df["lane_change"] == -1, "action"] = ACTION_LEFT
    df.loc[df["lane_change"] == 1, "action"] = ACTION_RIGHT
    # Drop multi-lane changes (rare, noisy)
    df = df[df["lane_change"].abs() <= 1].copy()

    # Compute next_state
    df["next_state"] = df["next_lane"] * n_speed_bins + df["next_speed_bin"]

    # Drop last frame per vehicle (no next_state) and NaN rows
    df = df.dropna(subset=["next_lane", "next_speed_bin"]).copy()
    df["next_state"] = df["next_state"].astype(int)
    df["action"] = df["action"].astype(int)

    # Filter vehicles with enough frames
    vehicle_counts = df["vehicle_id"].value_counts()
    valid_vehicles = vehicle_counts[vehicle_counts >= min_frames].index
    df = df[df["vehicle_id"].isin(valid_vehicles)].copy()

    if max_vehicles is not None:
        selected = df["vehicle_id"].unique()[:max_vehicles]
        df = df[df["vehicle_id"].isin(selected)].copy()

    # Add period (time index within vehicle)
    df["period"] = df.groupby("vehicle_id").cumcount()

    # Select output columns
    result = df[["vehicle_id", "period", "state", "action", "next_state",
                 "lane", "speed_bin", "v_vel", "space_headway", "lane_change"]].copy()
    result = result.reset_index(drop=True)

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import jax.numpy as jnp

        trajectories = []
        for vid in result["vehicle_id"].unique():
            vdata = result[result["vehicle_id"] == vid].sort_values("period")
            traj = Trajectory(
                states=jnp.array(vdata["state"].values, dtype=jnp.int32),
                actions=jnp.array(vdata["action"].values, dtype=jnp.int32),
                next_states=jnp.array(vdata["next_state"].values, dtype=jnp.int32),
                individual_id=int(vid),
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    return result


def get_ngsim_info() -> dict:
    """Return metadata about the NGSIM US-101 dataset."""
    return {
        "name": "NGSIM US-101 Vehicle Trajectories",
        "description": "Lane-change decisions on US-101 freeway, Los Angeles",
        "source": "FHWA Next Generation Simulation",
        "url": "https://datahub.transportation.gov/resource/8ect-6jqj",
        "n_states": N_LANES * N_SPEED_BINS,  # 50
        "n_actions": N_ACTIONS,  # 3
        "n_vehicles": 2848,
        "n_observations": "~4.8M raw frames (480K at 1Hz)",
        "state_description": "(lane, speed_bin)",
        "action_description": "lane change: left / stay / right",
        "lane_names": LANE_NAMES,
        "action_names": ACTION_NAMES,
    }
