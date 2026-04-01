"""Citibike Route Choice: IRL estimation of destination preferences.

Demonstrates NFXP and MCE-IRL on Citibike station-to-station
destination choice. Falls back to synthetic data if the real
Citibike data has not been downloaded.

Usage:
    python examples/citibike-route/run_estimation.py
"""

import time

import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.datasets.citibike_route import load_citibike_route
from econirl.environments.citibike_route import CitibikeRouteEnvironment
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.inference import etable
from econirl.inference.fit_metrics import brier_score, kl_divergence
from econirl.preferences.linear import LinearUtility


def main():
    print("=" * 65)
    print("Citibike Route Choice (80 states, 20 actions)")
    print("=" * 65)

    env = CitibikeRouteEnvironment(discount_factor=0.95)
    panel = load_citibike_route(as_panel=True, n_individuals=1000, n_periods=50)

    cutoff = int(panel.num_individuals * 0.8)
    train = Panel(trajectories=panel.trajectories[:cutoff])
    test = Panel(trajectories=panel.trajectories[cutoff:])
    print(f"  Train: {train.num_individuals} riders, {train.num_observations} obs")
    print(f"  Test:  {test.num_individuals} riders, {test.num_observations} obs")

    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    results = {}

    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust")
    results["NFXP"] = nfxp.estimate(train, utility, problem, transitions)
    print(f"\nNFXP: {time.time() - t0:.1f}s")
    print(results["NFXP"].summary())

    t0 = time.time()
    mce_config = MCEIRLConfig(learning_rate=0.05, outer_max_iter=500)
    mce = MCEIRLEstimator(config=mce_config)
    results["MCE-IRL"] = mce.estimate(train, utility, problem, transitions)
    print(f"\nMCE-IRL: {time.time() - t0:.1f}s")
    print(results["MCE-IRL"].summary())

    # Diagnostics
    print("\n" + "=" * 65)
    print("Post-Estimation Diagnostics")
    print("=" * 65)

    print("\n--- etable() ---")
    print(etable(results["NFXP"], results["MCE-IRL"]))

    obs_states = jnp.array(train.get_all_states())
    obs_actions = jnp.array(train.get_all_actions())

    print("\n--- Brier Scores ---")
    for name, r in results.items():
        bs = brier_score(r.policy, obs_states, obs_actions)
        print(f"  {name}: {bs['brier_score']:.4f}")

    sufficient = train.sufficient_stats(env.num_states, env.num_actions)
    data_ccps = jnp.array(sufficient.empirical_ccps)
    print("\n--- KL Divergence ---")
    for name, r in results.items():
        kl = kl_divergence(data_ccps, r.policy)
        print(f"  {name}: {kl['kl_divergence']:.6f}")


if __name__ == "__main__":
    main()
