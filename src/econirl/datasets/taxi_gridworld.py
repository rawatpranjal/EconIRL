"""
Taxi Gridworld Dataset for MaxEnt / MCE IRL Benchmarking.

This module generates a taxi route-choice dataset on a grid, inspired by
Ziebart et al. (2008)'s Pittsburgh taxi driver route preference learning.

The dataset models a taxi navigating an N x N grid toward a destination,
making directional choices at each intersection. This is the canonical
IRL benchmark used to test MaxEnt IRL and MCE IRL estimators.

State space:
    N^2 grid cells indexed as row * N + col.
    Destination cell at (N-1, N-1) is absorbing.

Action space:
    5 actions: Left (0), Right (1), Up (2), Down (3), Stay (4).

True reward:
    The taxi prefers shorter routes (negative distance cost) and receives
    a terminal reward at the destination. Known ground truth parameters
    enable parameter recovery benchmarking.

Reference:
    Ziebart, B.D., Maas, A., Bagnell, J.A., & Dey, A.K. (2008).
    "Maximum Entropy Inverse Reinforcement Learning." AAAI.
"""

from typing import Optional

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.gridworld import GridworldEnvironment
from econirl.simulation.synthetic import simulate_panel


def load_taxi_gridworld(
    grid_size: int = 10,
    n_individuals: int = 200,
    n_periods: int = 100,
    as_panel: bool = False,
    seed: int = 2008,
    step_penalty: float = -0.1,
    terminal_reward: float = 10.0,
    distance_weight: float = 0.1,
    discount_factor: float = 0.99,
) -> pd.DataFrame:
    """
    Generate a taxi-gridworld dataset for MCE IRL benchmarking.

    Creates synthetic taxi trajectories on an N x N grid where drivers
    navigate toward a destination at (N-1, N-1). The true reward parameters
    are known, enabling parameter recovery evaluation.

    This replicates the canonical setup from Ziebart et al. (2008) where
    MaxEnt IRL was first applied to taxi route preferences.

    Args:
        grid_size: Size of the grid (N x N). Default 10 gives 100 states.
        n_individuals: Number of taxi trajectories to simulate.
        n_periods: Number of steps per trajectory.
        as_panel: If True, return as Panel object for econirl estimators.
        seed: Random seed for reproducibility.
        step_penalty: Cost per non-terminal step (default -0.1).
        terminal_reward: Reward for reaching destination (default 10.0).
        distance_weight: Weight on distance-to-destination feature (default 0.1).
        discount_factor: Time discount factor (default 0.99).

    Returns:
        DataFrame with columns: taxi_id, period, state, action, next_state,
        row, col, next_row, next_col, manhattan_distance

        If as_panel=True, returns Panel object.
    """
    env = GridworldEnvironment(
        grid_size=grid_size,
        step_penalty=step_penalty,
        terminal_reward=terminal_reward,
        distance_weight=distance_weight,
        discount_factor=discount_factor,
        seed=seed,
    )

    panel = simulate_panel(
        env,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )

    if as_panel:
        return panel

    # Convert to DataFrame
    records = []
    for traj in panel.trajectories:
        tid = traj.individual_id
        states = traj.states
        actions = traj.actions
        next_states = traj.next_states

        for t in range(len(states)):
            s = int(states[t])
            row, col = s // grid_size, s % grid_size
            ns = int(next_states[t])
            nrow, ncol = ns // grid_size, ns % grid_size
            dist = abs(row - (grid_size - 1)) + abs(col - (grid_size - 1))

            records.append({
                "taxi_id": tid,
                "period": t,
                "state": s,
                "action": int(actions[t]),
                "next_state": ns,
                "row": row,
                "col": col,
                "next_row": nrow,
                "next_col": ncol,
                "manhattan_distance": dist,
            })

    return pd.DataFrame(records)


def get_taxi_gridworld_info(grid_size: int = 10) -> dict:
    """Return metadata about the taxi gridworld dataset."""
    return {
        "name": "Taxi Gridworld (Ziebart-style)",
        "description": f"Synthetic taxi route choice on {grid_size}x{grid_size} grid",
        "source": "Simulated from GridworldEnvironment (inspired by Ziebart et al. 2008)",
        "n_states": grid_size * grid_size,
        "n_actions": 5,
        "n_features": 3,
        "state_description": "(row, col) on grid",
        "action_description": "Left/Right/Up/Down/Stay",
        "true_parameters": {
            "step_penalty": -0.1,
            "terminal_reward": 10.0,
            "distance_weight": 0.1,
        },
        "ground_truth": True,
        "use_case": "MCE IRL parameter recovery benchmark",
    }
