"""Supermarket Pricing/Inventory: Structural estimation on Aguirregabiria (1999) data.

Demonstrates NFXP, CCP, and MCE-IRL on a retailer pricing and inventory
DDC problem using real data from a Spanish supermarket. 534 products
observed over 29 months.

Usage:
    python examples/supermarket/run_estimation.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.datasets.supermarket import load_supermarket
from econirl.environments.supermarket import SupermarketEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.inference import etable
from econirl.inference.fit_metrics import brier_score, kl_divergence
from econirl.inference.hypothesis_tests import vuong_test
from econirl.preferences.linear import LinearUtility
from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis


def main():
    print("=" * 65)
    print("Aguirregabiria (1999) Supermarket (10 states, 4 actions)")
    print("=" * 65)

    env = SupermarketEnvironment(discount_factor=0.95)
    panel = load_supermarket(as_panel=True)

    cutoff = int(panel.num_individuals * 0.8)
    train = Panel(trajectories=panel.trajectories[:cutoff])
    test = Panel(trajectories=panel.trajectories[cutoff:])
    print(f"  Train: {train.num_individuals} products, {train.num_observations} obs")
    print(f"  Test:  {test.num_individuals} products, {test.num_observations} obs")

    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    # Estimation
    results = {}

    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust")
    results["NFXP"] = nfxp.estimate(train, utility, problem, transitions)
    print(f"\nNFXP: {time.time() - t0:.1f}s")
    print(results["NFXP"].summary())

    t0 = time.time()
    ccp = CCPEstimator(num_policy_iterations=20, se_method="robust")
    results["CCP"] = ccp.estimate(train, utility, problem, transitions)
    print(f"\nCCP: {time.time() - t0:.1f}s")
    print(results["CCP"].summary())

    t0 = time.time()
    mce_config = MCEIRLConfig(learning_rate=0.1, outer_max_iter=300)
    mce = MCEIRLEstimator(config=mce_config)
    results["MCE-IRL"] = mce.estimate(train, utility, problem, transitions)
    print(f"\nMCE-IRL: {time.time() - t0:.1f}s")
    print(results["MCE-IRL"].summary())

    # Post-estimation diagnostics
    print("\n" + "=" * 65)
    print("Post-Estimation Diagnostics")
    print("=" * 65)

    print("\n--- etable() ---")
    print(etable(results["NFXP"], results["CCP"], results["MCE-IRL"]))

    obs_states = jnp.array(train.get_all_states())
    obs_actions = jnp.array(train.get_all_actions())

    print("\n--- Brier Scores ---")
    for name, r in results.items():
        bs = brier_score(r.policy, obs_states, obs_actions)
        print(f"  {name}: {bs['brier_score']:.4f}")

    print("\n--- Vuong Test (NFXP vs MCE-IRL) ---")
    vt = vuong_test(results["NFXP"].policy, results["MCE-IRL"].policy, obs_states, obs_actions)
    print(f"  Z-statistic: {vt['statistic']:.3f}")
    print(f"  P-value: {vt['p_value']:.4f}")
    print(f"  Direction: {vt['direction']}")

    sufficient = train.sufficient_stats(env.num_states, env.num_actions)
    data_ccps = jnp.array(sufficient.empirical_ccps)
    print("\n--- KL Divergence ---")
    for name, r in results.items():
        kl = kl_divergence(data_ccps, r.policy)
        print(f"  {name}: {kl['kl_divergence']:.6f}")

    # Counterfactual: halve stockout penalty
    print("\n" + "=" * 65)
    print("Counterfactual: Halve Stockout Penalty")
    print("=" * 65)
    best = results["MCE-IRL"]
    stockout_idx = env.parameter_names.index("stockout_penalty")
    new_params = best.parameters.at[stockout_idx].set(best.parameters[stockout_idx] / 2)
    cf = counterfactual_policy(best, new_params, utility, problem, transitions)
    print(f"Welfare change: {float(cf.welfare_change):+.3f}")

    # Elasticity on stockout_penalty
    print("\n--- Stockout Penalty Elasticity ---")
    ea = elasticity_analysis(
        best, utility, problem, transitions,
        parameter_name="stockout_penalty",
        pct_changes=[-0.50, -0.25, 0.25, 0.50, 1.0],
    )
    print(f"{'% Change':>10} {'Welfare':>12} {'Avg Policy':>12}")
    print("-" * 36)
    for i, pct in enumerate(ea["pct_changes"]):
        wc = ea["welfare_changes"][i]
        pc = ea["policy_changes"][i]
        print(f"{pct:>+10.0%} {float(wc):>+12.3f} {float(pc):>12.3f}")

    # Save results to JSON
    out = {
        "parameters": {},
        "standard_errors": {},
        "log_likelihoods": {},
    }
    for name, r in results.items():
        out["parameters"][name] = {
            pname: float(r.parameters[i])
            for i, pname in enumerate(env.parameter_names)
        }
        out["standard_errors"][name] = {
            pname: float(r.standard_errors[i])
            for i, pname in enumerate(env.parameter_names)
        }
        out["log_likelihoods"][name] = float(r.log_likelihood)
    out["counterfactual"] = {"welfare_change": float(cf.welfare_change)}
    out["elasticity"] = {
        "pct_changes": [float(p) for p in ea["pct_changes"]],
        "welfare_changes": [float(w) for w in ea["welfare_changes"]],
        "policy_changes": [float(p) for p in ea["policy_changes"]],
    }
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
