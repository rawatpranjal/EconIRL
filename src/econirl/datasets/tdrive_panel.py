"""T-Drive dataloader for econirl estimation.

Processes raw T-Drive GPS data into Panel, transitions, and feature
matrices suitable for MCE IRL and NFXP estimation.

Pipeline: GPS files → filter bbox → discretize grid → infer actions →
          split trajectories → build Panel + transitions + features

Usage:
    from econirl.datasets.tdrive_panel import load_tdrive_panel
    data = load_tdrive_panel(n_taxis=50, grid_size=15)
    panel, transitions, features = data["panel"], data["transitions"], data["feature_matrix"]
"""

from __future__ import annotations

import csv
import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from econirl.core.types import DDCProblem, Panel, Trajectory


# Actions: N=0, S=1, E=2, W=3, Stay=4
ACTION_NAMES = ["North", "South", "East", "West", "Stay"]
N_ACTIONS = 5


def load_tdrive_panel(
    data_dir: str | Path | None = None,
    n_taxis: int = 100,
    grid_size: int = 20,
    bbox: tuple[float, float, float, float] = (116.2, 116.6, 39.75, 40.05),
    min_traj_length: int = 5,
    max_gap_minutes: float = 30.0,
    discount_factor: float = 0.95,
    seed: int = 42,
) -> dict:
    """Load T-Drive data and preprocess for econirl estimation.

    Args:
        data_dir: Path to raw T-Drive data (folder with .txt files).
            If None, searches standard locations.
        n_taxis: Number of taxi files to load (for speed).
        grid_size: N for the NxN discretization grid.
        bbox: (lon_min, lon_max, lat_min, lat_max) bounding box.
        min_traj_length: Minimum trajectory length to keep.
        max_gap_minutes: Max time gap before splitting trajectory.
        discount_factor: Discount factor for DDCProblem.
        seed: Random seed for reproducible taxi sampling.

    Returns:
        Dictionary with:
            panel: Panel with Trajectory objects
            transitions: jnp.ndarray (n_actions, n_states, n_states)
            feature_matrix: jnp.ndarray (n_states, n_actions, n_features)
            feature_names: list[str]
            problem: DDCProblem
            metadata: dict with grid_size, bbox, n_taxis, etc.
    """
    if data_dir is None:
        data_dir = _find_data_dir()
    data_dir = Path(data_dir)

    # 1. Load GPS data from files
    print(f"  Loading GPS data from {n_taxis} taxis...")
    gps_data = _load_gps_files(data_dir, n_taxis, seed)
    print(f"  Raw GPS points: {sum(len(v) for v in gps_data.values()):,}")

    # 2. Filter to bounding box
    lon_min, lon_max, lat_min, lat_max = bbox
    filtered = {}
    for taxi_id, points in gps_data.items():
        pts = [(ts, lon, lat) for ts, lon, lat in points
               if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max]
        if pts:
            filtered[taxi_id] = pts
    print(f"  After bbox filter: {sum(len(v) for v in filtered.values()):,} points from {len(filtered)} taxis")

    # 3. Discretize to grid
    n_states = grid_size * grid_size
    all_trajectories = []
    transition_counts = np.zeros((N_ACTIONS, n_states, n_states), dtype=np.float64)

    for taxi_id, points in filtered.items():
        # Sort by timestamp
        points.sort(key=lambda x: x[0])

        # Convert to grid cells
        cells = []
        for ts, lon, lat in points:
            col = int((lon - lon_min) / (lon_max - lon_min) * grid_size)
            row = int((lat - lat_min) / (lat_max - lat_min) * grid_size)
            col = min(max(col, 0), grid_size - 1)
            row = min(max(row, 0), grid_size - 1)
            state = row * grid_size + col
            cells.append((ts, state))

        # Split into trajectories on time gaps
        trajs = _split_trajectories(cells, max_gap_minutes)

        for traj_cells in trajs:
            if len(traj_cells) < min_traj_length:
                continue

            states = []
            actions = []
            next_states = []

            for i in range(len(traj_cells) - 1):
                s = traj_cells[i][1]
                s_next = traj_cells[i + 1][1]
                a = _infer_action(s, s_next, grid_size)

                states.append(s)
                actions.append(a)
                next_states.append(s_next)

                # Count transitions
                transition_counts[a, s, s_next] += 1

            if len(states) >= min_traj_length - 1:
                all_trajectories.append(Trajectory(
                    states=jnp.array(states, dtype=jnp.int32),
                    actions=jnp.array(actions, dtype=jnp.int32),
                    next_states=jnp.array(next_states, dtype=jnp.int32),
                    individual_id=str(taxi_id),
                ))

    print(f"  Trajectories: {len(all_trajectories)}, Observations: {sum(len(t) for t in all_trajectories):,}")

    if len(all_trajectories) == 0:
        raise ValueError("No valid trajectories found. Try increasing n_taxis or relaxing filters.")

    # 4. Build transition matrices (normalize rows)
    transitions = jnp.zeros((N_ACTIONS, n_states, n_states), dtype=jnp.float32)
    for a in range(N_ACTIONS):
        for s in range(n_states):
            row_sum = transition_counts[a, s, :].sum()
            if row_sum > 0:
                transitions[a, s, :] = jnp.array(transition_counts[a, s, :] / row_sum, dtype=jnp.float32)
            else:
                # Unobserved (s, a): default to self-loop
                transitions[a, s, s] = 1.0

    # 5. Build feature matrix (n_states, n_actions, n_features)
    # Design: orthogonal features that avoid collinearity.
    # - move_cost: per-step cost of moving (not staying)
    # - dist_to_center: preference for central locations
    # - northward_pref: directional preference N vs S
    # - eastward_pref: directional preference E vs W
    feature_names = ["move_cost", "dist_to_center", "northward_pref", "eastward_pref"]
    n_features = len(feature_names)
    feature_matrix = jnp.zeros((n_states, N_ACTIONS, n_features), dtype=jnp.float32)

    center_row = grid_size / 2.0
    center_col = grid_size / 2.0
    max_dist = np.sqrt(center_row**2 + center_col**2)

    for s in range(n_states):
        row = s // grid_size
        col = s % grid_size
        dist = np.sqrt((row - center_row)**2 + (col - center_col)**2) / max_dist

        for a in range(N_ACTIONS):
            # move_cost: -1 for any movement, 0 for stay
            feature_matrix[s, a, 0] = -1.0 if a != 4 else 0.0
            # dist_to_center: normalized negative distance (closer = higher)
            feature_matrix[s, a, 1] = -dist
            # northward_pref: +1 for North, -1 for South, 0 otherwise
            feature_matrix[s, a, 2] = 1.0 if a == 0 else (-1.0 if a == 1 else 0.0)
            # eastward_pref: +1 for East, -1 for West, 0 otherwise
            feature_matrix[s, a, 3] = 1.0 if a == 2 else (-1.0 if a == 3 else 0.0)

    # 6. Build Panel and Problem
    panel = Panel(trajectories=all_trajectories)
    problem = DDCProblem(
        num_states=n_states,
        num_actions=N_ACTIONS,
        discount_factor=discount_factor,
    )

    metadata = {
        "grid_size": grid_size,
        "n_states": n_states,
        "n_actions": N_ACTIONS,
        "bbox": bbox,
        "n_taxis_loaded": len(filtered),
        "n_trajectories": len(all_trajectories),
        "n_observations": panel.num_observations,
        "action_names": ACTION_NAMES,
        "data_dir": str(data_dir),
    }

    return {
        "panel": panel,
        "transitions": transitions,
        "feature_matrix": feature_matrix,
        "feature_names": feature_names,
        "problem": problem,
        "metadata": metadata,
    }


def _find_data_dir() -> Path:
    """Search standard locations for T-Drive raw data."""
    candidates = [
        Path(__file__).parent.parent.parent.parent / "data" / "raw" / "tdrive",
        Path("data/raw/tdrive"),
        Path("../data/raw/tdrive"),
    ]
    for p in candidates:
        if p.exists() and any(p.glob("*.txt")):
            return p
    raise FileNotFoundError(
        "T-Drive data not found. Expected .txt files in data/raw/tdrive/. "
        "Download from: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/"
    )


def _load_gps_files(data_dir: Path, n_taxis: int, seed: int) -> dict:
    """Load GPS data from T-Drive .txt files."""
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    # Sample taxis reproducibly
    rng = np.random.RandomState(seed)
    if n_taxis < len(txt_files):
        indices = rng.choice(len(txt_files), size=n_taxis, replace=False)
        txt_files = [txt_files[i] for i in sorted(indices)]
    else:
        txt_files = txt_files[:n_taxis]

    gps_data = {}
    for fpath in txt_files:
        taxi_id = fpath.stem
        points = []
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    ts = datetime.strptime(row[1].strip(), "%Y-%m-%d %H:%M:%S")
                    lon = float(row[2])
                    lat = float(row[3])
                    points.append((ts, lon, lat))
                except (ValueError, IndexError):
                    continue
        if points:
            gps_data[taxi_id] = points

    return gps_data


def _split_trajectories(cells: list, max_gap_minutes: float) -> list:
    """Split a sequence of (timestamp, state) into trajectories on time gaps."""
    if len(cells) < 2:
        return [cells] if cells else []

    trajs = []
    current = [cells[0]]

    for i in range(1, len(cells)):
        gap = (cells[i][0] - cells[i - 1][0]).total_seconds() / 60.0
        if gap > max_gap_minutes:
            if current:
                trajs.append(current)
            current = [cells[i]]
        else:
            current.append(cells[i])

    if current:
        trajs.append(current)

    return trajs


def _infer_action(state: int, next_state: int, grid_size: int) -> int:
    """Infer action from state transition on grid."""
    if state == next_state:
        return 4  # Stay

    row, col = state // grid_size, state % grid_size
    next_row, next_col = next_state // grid_size, next_state % grid_size

    d_row = next_row - row
    d_col = next_col - col

    # Pick dominant direction
    if abs(d_row) >= abs(d_col):
        return 0 if d_row > 0 else 1  # North or South
    else:
        return 2 if d_col > 0 else 3  # East or West
