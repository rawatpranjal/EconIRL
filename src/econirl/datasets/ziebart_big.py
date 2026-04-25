"""Large gridworld dataset for IRL scalability benchmarks.

A 50-by-50 stochastic gridworld with 8 actions including diagonals and
a slip probability of 0.1. The reward is parameterized as a linear
combination of 16 radial basis functions on the grid coordinates,
generalizing the canonical Ziebart (2008) gridworld validation panel
to a state space large enough that tabular soft value iteration becomes
the bottleneck.

This dataset is the inverse-reinforcement-learning scalability counterpart
to `rust_big.py`. It demonstrates the IRL family transition from tabular
maximum causal entropy estimation on the small Ziebart gridworld
(`load_taxi_gridworld`) to neural function approximation on the same
canonical setup at scale.

Reference:
    Ziebart, B. D. (2010). "Modeling Purposeful Adaptive Behavior with
    the Principle of Maximum Causal Entropy." PhD thesis,
    Carnegie Mellon University.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Eight-action move set: cardinal plus diagonals.
_MOVES = np.array(
    [
        (-1, 0),  # 0: north
        (1, 0),   # 1: south
        (0, -1),  # 2: west
        (0, 1),   # 3: east
        (-1, -1), # 4: northwest
        (-1, 1),  # 5: northeast
        (1, -1),  # 6: southwest
        (1, 1),   # 7: southeast
    ],
    dtype=np.int32,
)


def _rbf_basis_centers(grid_size: int, n_basis: int) -> np.ndarray:
    """Place radial basis function centers on a square sublattice.

    The centers are arranged on the smallest square grid that fits
    `n_basis` points and then trimmed to exactly `n_basis` centers.
    Each center is a (row, col) pair on the same coordinate system as
    the grid cells.
    """
    side = int(np.ceil(np.sqrt(n_basis)))
    spacing = grid_size / (side + 1)
    centers = []
    for i in range(side):
        for j in range(side):
            r = (i + 1) * spacing - 0.5
            c = (j + 1) * spacing - 0.5
            centers.append((r, c))
    return np.array(centers[:n_basis], dtype=np.float32)


def _rbf_features(
    rows: np.ndarray, cols: np.ndarray, centers: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Evaluate RBF features at a batch of grid coordinates.

    Args:
        rows, cols: Integer arrays of shape (N,) with grid coordinates.
        centers: Float array of shape (K, 2) with RBF center coordinates.
        bandwidth: Gaussian bandwidth in grid units.

    Returns:
        Array of shape (N, K) with RBF activations.
    """
    coords = np.stack([rows, cols], axis=1).astype(np.float32)
    diffs = coords[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diffs**2, axis=2)
    return np.exp(-dist_sq / (2.0 * bandwidth**2))


def _build_transition_matrix(
    grid_size: int, slip_prob: float
) -> np.ndarray:
    """Build the (A, S, S) transition tensor for the 8-action grid.

    With probability `1 - slip_prob` the move succeeds. With probability
    `slip_prob` an alternative direction is drawn uniformly at random
    from the seven other actions. Moves that would leave the grid keep
    the agent in place.
    """
    n_states = grid_size * grid_size
    n_actions = len(_MOVES)
    transitions = np.zeros((n_actions, n_states, n_states), dtype=np.float32)

    for s in range(n_states):
        r, c = divmod(s, grid_size)
        for a in range(n_actions):
            for a_realized in range(n_actions):
                if a == a_realized:
                    p = 1.0 - slip_prob
                else:
                    p = slip_prob / (n_actions - 1)
                dr, dc = _MOVES[a_realized]
                nr = max(0, min(grid_size - 1, r + dr))
                nc = max(0, min(grid_size - 1, c + dc))
                ns = nr * grid_size + nc
                transitions[a, s, ns] += p

    return transitions


def _soft_value_iteration(
    rewards: np.ndarray,
    transitions: np.ndarray,
    discount_factor: float,
    scale_parameter: float,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the soft Bellman equation by contraction.

    Returns the integrated value `V(s)` and the soft policy
    `pi(a|s)` over the joint state-action space.
    """
    n_actions, n_states, _ = transitions.shape
    V = np.zeros(n_states, dtype=np.float32)
    for _ in range(max_iter):
        # Q[a, s] = r(s) + beta sum_{s'} P(s'|s,a) V(s')
        EV = np.einsum("ass,s->as", transitions, V)
        Q = rewards[None, :] + discount_factor * EV
        Q_T = Q.T  # shape (S, A)
        max_q = np.max(Q_T, axis=1, keepdims=True)
        V_new = (
            scale_parameter
            * (max_q.squeeze(-1) + np.log(
                np.sum(np.exp((Q_T - max_q) / scale_parameter), axis=1)
            ))
        )
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    EV = np.einsum("ass,s->as", transitions, V)
    Q = rewards[None, :] + discount_factor * EV
    Q_T = Q.T
    max_q = np.max(Q_T, axis=1, keepdims=True)
    policy = np.exp((Q_T - max_q) / scale_parameter)
    policy = policy / policy.sum(axis=1, keepdims=True)
    return V, policy


def _generate_ziebart_big_data(
    grid_size: int,
    n_actions: int,
    slip_prob: float,
    n_basis: int,
    n_trajectories: int,
    trajectory_length: int,
    discount_factor: float,
    scale_parameter: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Generate the bundled CSV and metadata for the dataset.

    Returns the trajectory DataFrame and a metadata dictionary that
    declares the true reward coefficients used for sampling.
    """
    if n_actions != len(_MOVES):
        raise ValueError(
            f"Only the 8-action move set is supported, got n_actions={n_actions}"
        )

    rng = np.random.default_rng(seed)
    n_states = grid_size * grid_size

    # Build the basis and the true reward.
    centers = _rbf_basis_centers(grid_size, n_basis)
    bandwidth = grid_size / (np.sqrt(n_basis) * 2.0)

    rows = np.repeat(np.arange(grid_size), grid_size).astype(np.int32)
    cols = np.tile(np.arange(grid_size), grid_size).astype(np.int32)
    feature_matrix = _rbf_features(rows, cols, centers, bandwidth)

    true_theta = rng.normal(loc=0.0, scale=1.0, size=n_basis).astype(np.float32)
    rewards = feature_matrix @ true_theta

    # Solve the soft Bellman equation under the true reward.
    transitions = _build_transition_matrix(grid_size, slip_prob)
    _, policy = _soft_value_iteration(
        rewards=rewards,
        transitions=transitions,
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
    )

    # Sample trajectories.
    records = []
    initial_states = rng.integers(0, n_states, size=n_trajectories)
    for traj_id in range(n_trajectories):
        s = int(initial_states[traj_id])
        for t in range(trajectory_length):
            a = int(rng.choice(n_actions, p=policy[s]))
            ns = int(rng.choice(n_states, p=transitions[a, s]))
            r, c = divmod(s, grid_size)
            nr, nc = divmod(ns, grid_size)
            records.append(
                {
                    "trajectory_id": traj_id,
                    "period": t,
                    "state": s,
                    "action": a,
                    "next_state": ns,
                    "row": r,
                    "col": c,
                    "next_row": nr,
                    "next_col": nc,
                }
            )
            s = ns

    metadata = {
        "true_theta": true_theta.tolist(),
        "basis_centers": centers.tolist(),
        "basis_bandwidth": float(bandwidth),
        "n_basis": n_basis,
        "grid_size": grid_size,
        "n_actions": n_actions,
        "n_states": n_states,
        "slip_prob": slip_prob,
        "discount_factor": discount_factor,
        "scale_parameter": scale_parameter,
        "n_trajectories": n_trajectories,
        "trajectory_length": trajectory_length,
        "seed": seed,
    }
    return pd.DataFrame(records), metadata


def load_ziebart_big(
    grid_size: int = 50,
    n_actions: int = 8,
    slip_prob: float = 0.1,
    n_basis: int = 16,
    n_trajectories: int = 50000,
    trajectory_length: int = 30,
    discount_factor: float = 0.99,
    scale_parameter: float = 1.0,
    seed: int = 42,
    as_panel: bool = False,
) -> pd.DataFrame:
    """Load or generate the large Ziebart gridworld dataset.

    A 50-by-50 stochastic gridworld with 8 actions and a slip
    probability of 0.1. The reward is a linear combination of 16
    radial basis functions on the grid coordinates with coefficients
    drawn once at generation time and stored alongside the panel.
    50000 trajectories of length 30 are sampled under the maximum
    causal entropy optimal policy at the true reward.

    Args:
        grid_size: Side length of the square grid. Default 50 produces
            a 2500-cell state space.
        n_actions: Move set size. Only the 8-action diagonal move set
            is supported.
        slip_prob: Probability that a move resolves to a uniformly
            chosen alternative direction.
        n_basis: Number of radial basis functions used in the reward.
        n_trajectories: Number of trajectories to sample.
        trajectory_length: Number of state-action pairs per trajectory.
        discount_factor: Time discount factor for the optimal policy.
        scale_parameter: Logit scale parameter.
        seed: Random seed for reproducibility.
        as_panel: If True, return a Panel object. If False, return a
            pandas DataFrame.

    Returns:
        DataFrame with trajectory records or Panel object whose
        metadata field carries the true reward coefficients.
    """
    csv_path = Path(__file__).parent / "ziebart_big_data.csv"
    meta_path = Path(__file__).parent / "ziebart_big_metadata.json"

    use_cache = (
        csv_path.exists()
        and meta_path.exists()
        and grid_size == 50
        and n_actions == 8
        and slip_prob == 0.1
        and n_basis == 16
        and n_trajectories == 50000
        and trajectory_length == 30
        and seed == 42
    )

    if use_cache:
        import json

        df = pd.read_csv(csv_path)
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        df, metadata = _generate_ziebart_big_data(
            grid_size=grid_size,
            n_actions=n_actions,
            slip_prob=slip_prob,
            n_basis=n_basis,
            n_trajectories=n_trajectories,
            trajectory_length=trajectory_length,
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import jax.numpy as jnp

        trajectories = []
        for tid, group in df.groupby("trajectory_id"):
            group = group.sort_values("period")
            states = jnp.array(group["state"].values, dtype=jnp.int32)
            actions = jnp.array(group["action"].values, dtype=jnp.int32)
            next_states = jnp.array(group["next_state"].values, dtype=jnp.int32)
            trajectories.append(
                Trajectory(
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    individual_id=int(tid),
                )
            )
        return Panel(trajectories=trajectories, metadata=metadata)

    return df


def get_ziebart_big_info() -> dict:
    """Return metadata about the large gridworld dataset.

    Returns the true reward coefficients used by the data-generating
    process if the bundled metadata is available; otherwise returns
    only the static descriptive fields. Validation tests check
    recovered coefficients against the `true_theta` field.
    """
    meta_path = Path(__file__).parent / "ziebart_big_metadata.json"
    static = {
        "name": "Ziebart gridworld at scale (50x50, 8 actions, RBF reward)",
        "grid_size": 50,
        "n_actions": 8,
        "n_states": 2500,
        "slip_prob": 0.1,
        "n_basis": 16,
        "discount_factor": 0.99,
        "scale_parameter": 1.0,
        "n_trajectories": 50000,
        "trajectory_length": 30,
        "ground_truth": True,
        "use_case": "Maximum causal entropy IRL scalability benchmark",
        "reference": (
            "Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior "
            "with the Principle of Maximum Causal Entropy. PhD thesis, "
            "Carnegie Mellon University."
        ),
    }
    if meta_path.exists():
        import json

        with open(meta_path) as f:
            dynamic = json.load(f)
        static.update(
            {
                "true_theta": dynamic.get("true_theta"),
                "basis_centers": dynamic.get("basis_centers"),
                "basis_bandwidth": dynamic.get("basis_bandwidth"),
                "seed": dynamic.get("seed"),
            }
        )
    return static
