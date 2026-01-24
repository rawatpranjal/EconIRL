"""Monte Carlo parameter recovery validation.

This module provides tools for validating estimator performance through
Monte Carlo simulation studies. The approach:

1. Generate data from known parameters
2. Estimate parameters using specified estimators
3. Compare estimated vs. true parameters
4. Compute summary statistics (bias, RMSE, coverage)

This is the standard approach for validating structural estimators
in econometrics.

References:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines"
    Aguirregabiria, V. and Mira, P. (2002). "Swapping the Nested Fixed Point"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Optional

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.simulation.synthetic import simulate_panel


def run_monte_carlo(
    n_simulations: int = 100,
    n_individuals: int = 500,
    n_periods: int = 100,
    true_operating_cost: float = 0.01,
    true_replacement_cost: float = 2.0,
    num_mileage_bins: int = 20,
    discount_factor: float = 0.99,
    estimators: list[str] = ["NFXP", "Hotz-Miller", "NPL"],
    seed: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run Monte Carlo parameter recovery experiment.

    Simulates data from the Rust (1987) bus engine replacement model
    with known parameters, estimates parameters using specified
    estimators, and collects results for analysis.

    The default parameters are set for fast and reliable estimation
    in finite samples. For the original Rust (1987) parameters,
    use operating_cost=0.001, replacement_cost=3.0, num_mileage_bins=90,
    discount_factor=0.9999.

    Args:
        n_simulations: Number of Monte Carlo replications
        n_individuals: Number of individuals per simulated panel
        n_periods: Number of time periods per individual
        true_operating_cost: True operating cost parameter (theta_c)
        true_replacement_cost: True replacement cost parameter (RC)
        num_mileage_bins: Number of mileage discretization bins
        discount_factor: Discount factor for the environment
        estimators: List of estimator names. Options:
            - "NFXP": Nested Fixed Point (full MLE)
            - "Hotz-Miller": CCP with K=1 (fast, consistent)
            - "NPL": CCP with K>1 (iterates toward MLE)
        seed: Random seed for reproducibility
        verbose: Whether to print progress messages

    Returns:
        DataFrame with columns:
            - simulation: Simulation index
            - estimator: Estimator name
            - theta_c: Estimated operating cost
            - theta_c_se: Standard error for theta_c
            - RC: Estimated replacement cost
            - RC_se: Standard error for RC
            - log_likelihood: Log-likelihood at estimates
            - converged: Whether optimization converged
            - true_theta_c: True operating cost
            - true_RC: True replacement cost

    Example:
        >>> results = run_monte_carlo(
        ...     n_simulations=100,
        ...     n_individuals=500,
        ...     n_periods=100,
        ...     estimators=["Hotz-Miller", "NPL"],
        ...     seed=42,
        ... )
        >>> summary = summarize_monte_carlo(results)
        >>> print(summary)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create environment with true parameters
    env = RustBusEnvironment(
        operating_cost=true_operating_cost,
        replacement_cost=true_replacement_cost,
        num_mileage_bins=num_mileage_bins,
        discount_factor=discount_factor,
    )

    # Get utility specification and problem
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    results = []

    for sim in range(n_simulations):
        if verbose:
            print(f"Simulation {sim + 1}/{n_simulations}")

        # Set seed for this simulation
        sim_seed = seed + sim if seed is not None else None

        # Simulate panel data
        panel = simulate_panel(
            env,
            n_individuals=n_individuals,
            n_periods=n_periods,
            seed=sim_seed,
        )

        # Estimate with each estimator
        for est_name in estimators:
            if est_name == "NFXP":
                estimator = NFXPEstimator(verbose=False, outer_max_iter=200)
            elif est_name == "Hotz-Miller":
                estimator = CCPEstimator(num_policy_iterations=1, verbose=False)
            elif est_name == "NPL":
                estimator = CCPEstimator(num_policy_iterations=10, verbose=False)
            else:
                raise ValueError(f"Unknown estimator: {est_name}")

            try:
                result = estimator.estimate(panel, utility, problem, transitions)

                results.append({
                    'simulation': sim,
                    'estimator': est_name,
                    'theta_c': result.parameters[0].item(),
                    'theta_c_se': result.standard_errors[0].item() if result.standard_errors is not None else np.nan,
                    'RC': result.parameters[1].item(),
                    'RC_se': result.standard_errors[1].item() if result.standard_errors is not None else np.nan,
                    'log_likelihood': result.log_likelihood,
                    'converged': result.converged,
                    'true_theta_c': true_operating_cost,
                    'true_RC': true_replacement_cost,
                })
            except Exception as e:
                if verbose:
                    print(f"Sim {sim}, {est_name} failed: {e}")
                results.append({
                    'simulation': sim,
                    'estimator': est_name,
                    'theta_c': np.nan,
                    'theta_c_se': np.nan,
                    'RC': np.nan,
                    'RC_se': np.nan,
                    'log_likelihood': np.nan,
                    'converged': False,
                    'true_theta_c': true_operating_cost,
                    'true_RC': true_replacement_cost,
                })

    return pd.DataFrame(results)


def summarize_monte_carlo(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo results with bias, RMSE, coverage.

    Computes standard summary statistics for Monte Carlo experiments:
    - Mean estimate
    - Bias (mean estimate - true value)
    - Standard deviation of estimates
    - RMSE (root mean squared error)
    - 95% CI coverage rate

    Args:
        results: DataFrame from run_monte_carlo()

    Returns:
        DataFrame with summary statistics per estimator and parameter

    Example:
        >>> results = run_monte_carlo(n_simulations=100, seed=42)
        >>> summary = summarize_monte_carlo(results)
        >>> print(summary[['estimator', 'parameter', 'bias', 'rmse', 'coverage_95']])
    """
    summary = []

    for est_name in results['estimator'].unique():
        est_results = results[results['estimator'] == est_name]
        converged = est_results[est_results['converged'] == True]

        for param in ['theta_c', 'RC']:
            if len(converged) == 0:
                continue

            true_val = converged[f'true_{param}'].iloc[0]
            estimates = converged[param].dropna()

            if len(estimates) == 0:
                continue

            bias = estimates.mean() - true_val
            rmse = np.sqrt(((estimates - true_val) ** 2).mean())
            std = estimates.std()

            # Coverage: fraction of 95% CIs containing true value
            if f'{param}_se' in converged.columns:
                ses = converged[f'{param}_se'].dropna()
                valid_mask = ~estimates.isna() & ~ses.isna()
                valid_estimates = estimates[valid_mask]
                valid_ses = ses[valid_mask]

                if len(valid_ses) > 0 and len(valid_ses) == len(valid_estimates):
                    lower = valid_estimates - 1.96 * valid_ses
                    upper = valid_estimates + 1.96 * valid_ses
                    coverage = ((lower <= true_val) & (true_val <= upper)).mean()
                else:
                    coverage = np.nan
            else:
                coverage = np.nan

            summary.append({
                'estimator': est_name,
                'parameter': param,
                'true_value': true_val,
                'mean_estimate': estimates.mean(),
                'bias': bias,
                'std': std,
                'rmse': rmse,
                'coverage_95': coverage,
                'n_converged': len(estimates),
                'n_total': len(est_results),
            })

    return pd.DataFrame(summary)
