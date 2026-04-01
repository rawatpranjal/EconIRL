"""Shanghai taxi route-choice dataset (Zhao & Liang 2022).

Road network MDP for inverse reinforcement learning on taxi route choice.
320 intersections, 714 road segments (states), 8 directional actions.

States are road segments (edges) indexed by n_id (0-713). Actions are 8
compass directions (0-7). The transition matrix encodes which segments
are reachable from each segment in each direction.

Reference:
    Zhao, Z., & Liang, Y. (2022). Deep Inverse Reinforcement Learning
    for Route Choice Modeling. arXiv:2206.10598.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

DEFAULT_DATA_DIR = "/Volumes/Expansion/datasets/shanghai_taxi_rcm_airl/data"

N_STATES = 714
N_ACTIONS = 8  # 8 compass directions (0-7)

# Highway type mapping for one-hot encoding
HIGHWAY_TYPES = {
    "residential": 0,
    "primary": 1,
    "secondary": 2,
    "tertiary": 3,
    "living_street": 4,
    "unclassified": 5,
}


def load_shanghai_network(data_dir: Optional[str] = None) -> dict:
    """Load Shanghai road network data.

    Parameters
    ----------
    data_dir : str or None
        Path to the dataset directory. If None, uses the default location
        at ``/Volumes/Expansion/datasets/shanghai_taxi_rcm_airl/data``.

    Returns
    -------
    dict
        Keys: ``"nodes"`` (DataFrame), ``"edges"`` (DataFrame),
        ``"transit"`` (ndarray of shape (1737, 3)), ``"n_states"`` (714),
        ``"n_actions"`` (8).
    """
    data_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)

    nodes = pd.read_csv(data_dir / "node.txt")
    nodes = nodes.rename(columns={"y": "lat", "x": "lon"})

    edges = pd.read_csv(data_dir / "edge.txt")

    transit = np.load(data_dir / "transit.npy")

    return {
        "nodes": nodes,
        "edges": edges,
        "transit": transit,
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
    }


def load_shanghai_trajectories(
    split: str = "train",
    cv: int = 0,
    size: int = 1000,
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load Shanghai route-choice trajectory data.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    cv : int
        Cross-validation fold index (0-4).
    size : int
        Training set size (100, 1000, or 10000). Ignored for test split.
    data_dir : str or None
        Path to the dataset directory.

    Returns
    -------
    pd.DataFrame
        Columns: ``ori``, ``des``, ``path`` (underscore-separated n_id
        sequence), ``len`` (number of edges in path).
    """
    data_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
    cv_dir = data_dir / "cross_validation"

    if split == "train":
        filename = f"train_CV{cv}_size{size}.csv"
    else:
        filename = f"test_CV{cv}.csv"

    return pd.read_csv(cv_dir / filename)


def build_transition_matrix(
    transit: np.ndarray,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> jnp.ndarray:
    """Build deterministic transition matrix from transit triples.

    For each ``(from_state, direction, to_state)`` row in ``transit``, sets
    ``T[direction, from_state, to_state] = 1.0``. State-action pairs not
    present in transit get a self-loop (absorbing). Each row sums to 1.

    Parameters
    ----------
    transit : ndarray of shape (K, 3)
        Each row is ``[from_n_id, direction_id, to_n_id]``.
    n_states : int
        Number of states (714).
    n_actions : int
        Number of actions (8).

    Returns
    -------
    jnp.ndarray of shape (n_actions, n_states, n_states)
        Deterministic transition matrix with rows summing to 1.
    """
    T = jnp.zeros(n_actions, n_states, n_states)

    for row in transit:
        from_s, action, to_s = int(row[0]), int(row[1]), int(row[2])
        T[action, from_s, to_s] = 1.0

    # For (state, action) pairs not in transit, add self-loop
    row_sums = T.sum(axis=2)  # (n_actions, n_states)
    missing = row_sums == 0.0
    # Set self-loop for missing (s, a) pairs
    for a in range(n_actions):
        for s in range(n_states):
            if missing[a, s]:
                T[a, s, s] = 1.0

    # Normalize rows to sum to 1 (already 1 for deterministic, but safe)
    row_sums = T.sum(axis=2, keepdims=True)
    T = T / row_sums.clamp(min=1e-12)

    return T


def _classify_highway(highway_str: str) -> int:
    """Map highway type string to category index.

    Handles list-like strings (e.g., "['residential', 'unclassified']") by
    taking the first recognized type.
    """
    hw = str(highway_str).strip()
    if hw in HIGHWAY_TYPES:
        return HIGHWAY_TYPES[hw]
    # Handle list-like strings from OSM data
    for key in HIGHWAY_TYPES:
        if key in hw:
            return HIGHWAY_TYPES[key]
    return HIGHWAY_TYPES["unclassified"]


def build_edge_features(
    edges_df: pd.DataFrame,
    n_states: int = N_STATES,
) -> jnp.ndarray:
    """Build per-edge feature matrix.

    Features:
        0: length (normalized to [0, 1] by max length)
        1-6: one-hot encoding of highway type

    Parameters
    ----------
    edges_df : pd.DataFrame
        Edge data with columns ``n_id``, ``length``, ``highway``.
    n_states : int
        Number of edge-states (714).

    Returns
    -------
    jnp.ndarray of shape (n_states, 7)
    """
    features = jnp.zeros(n_states, 7)

    max_length = edges_df["length"].max()

    for _, row in edges_df.iterrows():
        nid = int(row["n_id"])
        # Feature 0: normalized length
        features[nid, 0] = row["length"] / max_length
        # Features 1-6: one-hot highway type
        hw_idx = _classify_highway(row["highway"])
        features[nid, 1 + hw_idx] = 1.0

    return features


def build_state_action_features(
    edge_features: jnp.ndarray,
    transit: np.ndarray,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> jnp.ndarray:
    """Build state-action feature matrix.

    For each ``(from_state, action, to_state)`` triple, the feature vector
    at ``[from_state, action, :]`` is ``edge_features[to_state, :]`` -- i.e.,
    the features of the road segment reached by taking that action.

    Invalid (state, action) pairs get zero features.

    Parameters
    ----------
    edge_features : jnp.ndarray of shape (n_states, n_features)
        Per-edge features.
    transit : ndarray of shape (K, 3)
        Each row is ``[from_n_id, direction_id, to_n_id]``.
    n_states : int
        Number of states (714).
    n_actions : int
        Number of actions (8).

    Returns
    -------
    jnp.ndarray of shape (n_states, n_actions, n_features)
    """
    n_features = edge_features.shape[1]
    features = jnp.zeros(n_states, n_actions, n_features)

    for row in transit:
        from_s, action, to_s = int(row[0]), int(row[1]), int(row[2])
        features[from_s, action, :] = edge_features[to_s]

    return features


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters between two (lat, lon) points."""
    R = 6_371_000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def add_destination_feature(
    features: jnp.ndarray,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    destination_nid: int,
) -> jnp.ndarray:
    """Append a normalized distance-to-destination feature.

    Computes the straight-line (haversine) distance from each edge's
    endpoint to the destination edge's endpoint, normalized by the
    maximum distance in the network.

    Parameters
    ----------
    features : jnp.ndarray
        Either edge features of shape ``(S, F)`` or state-action features
        of shape ``(S, A, F)``.
    nodes_df : pd.DataFrame
        Node data with columns ``osmid``, ``lat``, ``lon``.
    edges_df : pd.DataFrame
        Edge data with columns ``n_id``, ``v`` (endpoint node osmid).
    destination_nid : int
        The n_id of the destination edge.

    Returns
    -------
    jnp.ndarray
        Features with an appended distance column: shape ``(S, F+1)``
        or ``(S, A, F+1)``.
    """
    # Build osmid -> (lat, lon) lookup
    node_coords = {}
    for _, row in nodes_df.iterrows():
        node_coords[int(row["osmid"])] = (row["lat"], row["lon"])

    # Map each edge n_id to its endpoint (v node) coordinates
    edge_endpoints = {}
    for _, row in edges_df.iterrows():
        nid = int(row["n_id"])
        v_node = int(row["v"])
        if v_node in node_coords:
            edge_endpoints[nid] = node_coords[v_node]

    # Destination endpoint
    dest_lat, dest_lon = edge_endpoints[destination_nid]

    # Compute distances
    n_states = features.shape[0]
    distances = jnp.zeros(n_states)
    for s in range(n_states):
        if s in edge_endpoints:
            lat, lon = edge_endpoints[s]
            distances[s] = _haversine(lat, lon, dest_lat, dest_lon)

    # Normalize by max distance
    max_dist = distances.max()
    if max_dist > 0:
        distances = distances / max_dist

    if features.dim() == 2:
        # (S, F) -> (S, F+1)
        return jnp.concatenate([features, distances.unsqueeze(1)], axis=1)
    elif features.dim() == 3:
        # (S, A, F) -> (S, A, F+1)
        n_actions = features.shape[1]
        dist_expanded = distances.unsqueeze(1).unsqueeze(2).expand(-1, n_actions, 1)
        return jnp.concatenate([features, dist_expanded], axis=2)
    else:
        raise ValueError(f"Expected 2D or 3D features, got {features.dim()}D")


def _build_transit_lookup(
    transit: np.ndarray,
) -> dict[tuple[int, int], int]:
    """Build (from_state, to_state) -> direction_id lookup from transit."""
    lookup = {}
    for row in transit:
        from_s, direction, to_s = int(row[0]), int(row[1]), int(row[2])
        lookup[(from_s, to_s)] = direction
    return lookup


def parse_trajectories_to_panel(
    traj_df: pd.DataFrame,
    transit: np.ndarray,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> "TrajectoryPanel":
    """Parse route trajectory DataFrame into a TrajectoryPanel.

    For each row in traj_df, parses the underscore-separated path string
    into a sequence of states, infers actions from the transit lookup, and
    constructs Trajectory objects.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Trajectory data with columns ``ori``, ``des``, ``path``, ``len``.
    transit : ndarray of shape (K, 3)
        Each row is ``[from_n_id, direction_id, to_n_id]``.
    n_states : int
        Number of states (714).
    n_actions : int
        Number of actions (8).

    Returns
    -------
    TrajectoryPanel
    """
    from econirl.core.types import Trajectory, TrajectoryPanel

    lookup = _build_transit_lookup(transit)
    trajectories = []

    for idx, row in traj_df.iterrows():
        path_str = str(row["path"])
        path = [int(x) for x in path_str.split("_")]

        if len(path) < 2:
            continue

        states = path[:-1]
        next_states = path[1:]
        actions = []
        for s, ns in zip(states, next_states):
            action = lookup.get((s, ns), 0)
            actions.append(action)

        traj = Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states, dtype=jnp.int32),
            individual_id=idx,
        )
        trajectories.append(traj)

    return TrajectoryPanel(trajectories=trajectories)


def load_shanghai_route(
    split: str = "train",
    cv: int = 0,
    size: int = 1000,
    data_dir: Optional[str] = None,
) -> "TrajectoryPanel":
    """Load Shanghai route-choice data as a TrajectoryPanel.

    Convenience function that loads trajectories and transit data, then
    parses them into a TrajectoryPanel ready for IRL estimators.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    cv : int
        Cross-validation fold index (0-4).
    size : int
        Training set size (100, 1000, or 10000). Ignored for test split.
    data_dir : str or None
        Path to the dataset directory.

    Returns
    -------
    TrajectoryPanel
    """
    data_dir_str = data_dir
    network = load_shanghai_network(data_dir_str)
    traj_df = load_shanghai_trajectories(split=split, cv=cv, size=size, data_dir=data_dir_str)
    return parse_trajectories_to_panel(
        traj_df, network["transit"], network["n_states"], network["n_actions"]
    )
