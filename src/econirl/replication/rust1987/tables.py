"""Rust (1987) table replication functions."""

from __future__ import annotations

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

from econirl.datasets import load_rust_bus
from econirl.estimation.transitions import estimate_transition_probs_by_group
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.core.types import Panel, Trajectory


def table_ii_descriptives(
    df: Optional[pd.DataFrame] = None,
    original: bool = True,
) -> pd.DataFrame:
    """Replicate Table II: Descriptive Statistics on Odometer Readings.

    This function computes descriptive statistics matching Table II from
    Rust (1987), including:
    - Number of buses per group
    - Number of engine replacements
    - Mean and std of mileage at replacement
    - Mean and std of final observed mileage

    Args:
        df: DataFrame with bus data. If None, loads from load_rust_bus().
        original: If df is None, whether to load original (True) or synthetic data.

    Returns:
        DataFrame indexed by group with descriptive statistics columns.

    Example:
        >>> from econirl.replication.rust1987 import table_ii_descriptives
        >>> table = table_ii_descriptives()
        >>> print(table)
    """
    if df is None:
        df = load_rust_bus(original=original)

    results = []

    for group in sorted(df['group'].unique()):
        group_df = df[df['group'] == group]

        n_buses = group_df['bus_id'].nunique()
        n_replacements = group_df['replaced'].sum()

        # Mileage at replacement
        replacement_obs = group_df[group_df['replaced'] == 1]
        if len(replacement_obs) > 0:
            mean_mileage = replacement_obs['mileage'].mean()
            std_mileage = replacement_obs['mileage'].std()
        else:
            mean_mileage = np.nan
            std_mileage = np.nan

        # Final mileage (last observation per bus)
        final_obs = group_df.groupby('bus_id').last()
        mean_final = final_obs['mileage'].mean()
        std_final = final_obs['mileage'].std()

        results.append({
            'group': group,
            'n_buses': n_buses,
            'n_replacements': int(n_replacements),
            'mean_mileage': mean_mileage,
            'std_mileage': std_mileage,
            'mean_final_mileage': mean_final,
            'std_final_mileage': std_final,
        })

    return pd.DataFrame(results).set_index('group')


def table_iv_transitions(
    df: Optional[pd.DataFrame] = None,
    original: bool = True,
) -> pd.DataFrame:
    """Replicate Table IV: Transition Probability Estimates.

    This function computes the mileage transition probability estimates
    (theta_0, theta_1, theta_2) for each bus group, matching Table IV
    from Rust (1987).

    Args:
        df: DataFrame with bus data. If None, loads from load_rust_bus().
        original: If df is None, whether to load original (True) or synthetic data.

    Returns:
        DataFrame indexed by group with transition probability columns:
        - theta_0: P(mileage increase = 0 bins)
        - theta_1: P(mileage increase = 1 bin)
        - theta_2: P(mileage increase = 2 bins)
        - n_transitions: Number of transitions used in estimation

    Example:
        >>> from econirl.replication.rust1987 import table_iv_transitions
        >>> table = table_iv_transitions()
        >>> print(table)
    """
    if df is None:
        df = load_rust_bus(original=original)

    probs_by_group = estimate_transition_probs_by_group(df)

    results = []
    for group, probs in probs_by_group.items():
        results.append({
            'group': group,
            'theta_0': probs[0],
            'theta_1': probs[1],
            'theta_2': probs[2],
            'n_transitions': len(df[(df['group'] == group) & (df['replaced'] == 0)]),
        })

    return pd.DataFrame(results).set_index('group')


def _df_to_panel(df: pd.DataFrame) -> Panel:
    """Convert DataFrame to Panel format for estimation.

    Args:
        df: DataFrame with columns bus_id, period, mileage_bin, replaced

    Returns:
        Panel object with trajectories for each bus
    """
    trajectories = []

    for bus_id in df['bus_id'].unique():
        bus_data = df[df['bus_id'] == bus_id].sort_values('period')

        states = jnp.array(bus_data['mileage_bin'].values, dtype=jnp.int32)
        actions = jnp.array(bus_data['replaced'].values, dtype=jnp.int32)

        # Compute next_states: shift states by 1, last one is 0 (placeholder)
        next_states = jnp.concatenate([states[1:], jnp.array([0])])

        traj = Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=int(bus_id),
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


def table_v_structural(
    df: Optional[pd.DataFrame] = None,
    groups: Optional[list[int]] = None,
    estimators: list[str] = ["NFXP", "Hotz-Miller", "NPL"],
    discount_factor: float = 0.9999,
    n_bootstrap: int = 0,
    original: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Replicate Table V: Structural Parameter Estimates.

    Table V reports estimates of the cost parameters:
    - theta_c: Operating cost (cost per mileage unit)
    - RC: Replacement cost

    for different specifications and estimators.

    Args:
        df: DataFrame with bus data. If None, loads from load_rust_bus().
        groups: List of bus groups to estimate. Default is [4] (Rust's main group).
        estimators: List of estimation methods. Options: "NFXP", "Hotz-Miller", "NPL".
        discount_factor: Time discount factor beta. Default is 0.9999 (Rust's value).
        n_bootstrap: Number of bootstrap replications for SEs. Default 0 (use asymptotic).
        original: If df is None, whether to load original (True) or synthetic data.
        verbose: Whether to print progress messages.

    Returns:
        DataFrame with columns:
        - group: Bus group number
        - estimator: Estimation method used
        - theta_c: Operating cost parameter estimate
        - theta_c_se: Standard error for theta_c
        - RC: Replacement cost parameter estimate
        - RC_se: Standard error for RC
        - log_likelihood: Maximized log-likelihood
        - converged: Whether estimation converged

    Example:
        >>> from econirl.replication.rust1987 import table_v_structural
        >>> table = table_v_structural(groups=[4], estimators=["NFXP"])
        >>> print(table)
    """
    if df is None:
        df = load_rust_bus(original=original)

    if groups is None:
        groups = [4]  # Rust focuses on Group 4

    # Get transition probabilities from first stage
    probs_by_group = estimate_transition_probs_by_group(df)

    results = []

    for group in groups:
        group_df = df[df['group'] == group]
        panel = _df_to_panel(group_df)

        # Set up environment with estimated transitions
        trans_probs = tuple(probs_by_group[group])

        env = RustBusEnvironment(
            operating_cost=0.001,  # Initial guess
            replacement_cost=3.0,  # Initial guess
            mileage_transition_probs=trans_probs,
            discount_factor=discount_factor,
        )

        utility = LinearUtility.from_environment(env)
        problem = env.problem_spec
        transitions = env.transition_matrices

        for est_name in estimators:
            if verbose:
                print(f"Estimating Group {group} with {est_name}...")

            if est_name == "NFXP":
                estimator = NFXPEstimator(verbose=False, outer_max_iter=200)
            elif est_name == "Hotz-Miller":
                estimator = CCPEstimator(num_policy_iterations=1, verbose=False)
            elif est_name == "NPL":
                estimator = CCPEstimator(num_policy_iterations=10, verbose=False)
            else:
                raise ValueError(f"Unknown estimator: {est_name}")

            result = estimator.estimate(panel, utility, problem, transitions)

            # Extract results
            theta_c = result.parameters[0].item()
            RC = result.parameters[1].item()

            # Get standard errors
            theta_c_se = result.standard_errors[0].item() if result.standard_errors is not None else np.nan
            RC_se = result.standard_errors[1].item() if result.standard_errors is not None else np.nan

            results.append({
                'group': group,
                'estimator': est_name,
                'theta_c': theta_c,
                'theta_c_se': theta_c_se,
                'RC': RC,
                'RC_se': RC_se,
                'log_likelihood': result.log_likelihood,
                'converged': result.converged,
            })

    return pd.DataFrame(results)
