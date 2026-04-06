"""Bayesian IRL: Posterior uncertainty without asymptotics.

Demonstrates the key contribution of Ramachandran and Amir (2007, IJCAI):
1. All other IRL methods give point estimates with asymptotic confidence
   intervals that can be unreliable in small samples.
2. Bayesian IRL gives the full posterior over reward parameters via MCMC,
   providing valid uncertainty quantification at any sample size.
3. At N=10, asymptotic SEs may be unreliable. The posterior is always valid.
4. At N=200, the posterior concentrates and agrees with the MLE.

The showcase compares Bayesian IRL posteriors against MCE-IRL point
estimates at three sample sizes on the Frozen Lake environment.

Usage:
    python examples/frozen-lake/bayesian_irl_uncertainty_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.frozen_lake import FrozenLakeEnvironment
from econirl.contrib.bayesian_irl import BayesianIRLEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.linear import LinearUtility
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


SAMPLE_SIZES = [10, 50, 200]


def run_diagnostics(panel, problem, utility):
    """Print pre-estimation diagnostics per CLAUDE.md."""
    features = np.array(utility.feature_matrix)
    n_states = problem.num_states
    n_actions = problem.num_actions
    n_features = features.shape[2]

    flat_features = features.reshape(n_states * n_actions, n_features)
    rank = np.linalg.matrix_rank(flat_features)
    nonzero_mask = np.any(flat_features != 0, axis=1)
    cond = np.linalg.cond(flat_features[nonzero_mask])

    all_states = panel.get_all_states()
    all_actions = panel.get_all_actions()
    covered = len(np.unique(all_states))
    single_action = 0
    for s in range(n_states):
        actions_at_s = all_actions[all_states == s]
        if len(actions_at_s) > 0 and len(np.unique(actions_at_s)) == 1:
            single_action += 1

    print(f"\n  Pre-estimation diagnostics:")
    print(f"    Feature matrix rank: {rank} / {n_features}")
    print(f"    Condition number: {cond:.1f}")
    print(f"    State coverage: {covered} / {n_states}")
    print(f"    Single-action states: {single_action}")


def main():
    print("=" * 70)
    print("Bayesian IRL: Posterior Uncertainty Without Asymptotics")
    print("Ramachandran and Amir (2007, IJCAI)")
    print("=" * 70)
    print()
    print("All other IRL methods rely on asymptotic SEs.")
    print("Bayesian IRL gives exact finite-sample posteriors via MCMC.")
    print(f"Comparing across sample sizes: {SAMPLE_SIZES}")

    env = FrozenLakeEnvironment(discount_factor=0.95)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = env.get_true_parameter_vector()

    print(f"\n  States: {problem.num_states}, Actions: {problem.num_actions}")
    print(f"  True parameters: {[f'{float(p):.4f}' for p in true_params]}")
    print(f"  Parameter names: {list(env.parameter_names)}")

    results = {}

    for n_ind in SAMPLE_SIZES:
        print(f"\n{'=' * 70}")
        print(f"  N = {n_ind} individuals, 50 periods each")
        print(f"{'=' * 70}")

        panel = env.generate_panel(
            n_individuals=n_ind, n_periods=50, seed=42
        )
        n_obs = panel.num_observations
        print(f"  Observations: {n_obs}")

        if n_ind == SAMPLE_SIZES[0]:
            run_diagnostics(panel, problem, utility)

        # ---- MCE-IRL (point estimate + asymptotic SEs) ----
        print(f"\n  --- MCE-IRL (point estimate) ---")
        t0 = time.time()
        mce = MCEIRLEstimator(config=MCEIRLConfig(
            learning_rate=0.1,
            outer_max_iter=500,
            verbose=False,
        ))
        try:
            mce_result = mce.estimate(panel, utility, problem, transitions)
            mce_time = time.time() - t0
            mce_params = np.array(mce_result.parameters)
            mce_se = np.array(mce_result.standard_errors)
            print(f"    Time: {mce_time:.1f}s")
            for i, name in enumerate(env.parameter_names):
                se_str = f"{mce_se[i]:.4f}" if not np.isnan(mce_se[i]) else "NaN"
                print(f"    {name}: {mce_params[i]:.4f} (SE: {se_str})")
        except Exception as e:
            print(f"    MCE-IRL failed: {e}")
            mce_params = np.full(len(true_params), np.nan)
            mce_se = np.full(len(true_params), np.nan)
            mce_time = time.time() - t0

        # ---- Bayesian IRL (posterior) ----
        print(f"\n  --- Bayesian IRL (MCMC posterior) ---")
        t0 = time.time()
        birl = BayesianIRLEstimator(
            n_samples=3000,
            burnin=1000,
            proposal_sigma=0.05,
            prior_sigma=5.0,
            confidence=1.0,
            verbose=False,
        )
        try:
            birl_result = birl.estimate(panel, utility, problem, transitions)
            birl_time = time.time() - t0
            birl_params = np.array(birl_result.parameters)
            birl_se = np.array(birl_result.standard_errors)
            accept_rate = birl_result.metadata.get("acceptance_rate", float("nan"))
            print(f"    Time: {birl_time:.1f}s")
            print(f"    Acceptance rate: {accept_rate:.2%}")
            for i, name in enumerate(env.parameter_names):
                se_str = f"{birl_se[i]:.4f}" if not np.isnan(birl_se[i]) else "NaN"
                print(f"    {name}: {birl_params[i]:.4f} (posterior SD: {se_str})")
        except Exception as e:
            print(f"    Bayesian IRL failed: {e}")
            birl_params = np.full(len(true_params), np.nan)
            birl_se = np.full(len(true_params), np.nan)
            birl_time = time.time() - t0
            accept_rate = float("nan")

        # ---- Compare ----
        print(f"\n  --- Comparison ---")
        print(f"    {'Param':>15} {'True':>8} {'MCE':>8} {'MCE SE':>8} "
              f"{'BIRL':>8} {'Post SD':>8}")
        print(f"    {'-'*55}")
        for i, name in enumerate(env.parameter_names):
            print(f"    {name:>15} {float(true_params[i]):>8.4f} "
                  f"{mce_params[i]:>8.4f} {mce_se[i]:>8.4f} "
                  f"{birl_params[i]:>8.4f} {birl_se[i]:>8.4f}")

        results[str(n_ind)] = {
            "n_obs": n_obs,
            "mce": {
                "params": mce_params.tolist(),
                "se": mce_se.tolist(),
                "time": mce_time,
            },
            "birl": {
                "params": birl_params.tolist(),
                "posterior_sd": birl_se.tolist(),
                "time": birl_time,
                "acceptance_rate": float(accept_rate),
            },
        }

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Uncertainty Quantification Across Sample Sizes")
    print("=" * 70)

    print(f"\n{'N':>6} {'MCE mean SE':>14} {'BIRL mean SD':>14} {'Ratio':>8}")
    print("-" * 45)
    for n_str, r in results.items():
        mce_mean_se = np.nanmean(r["mce"]["se"])
        birl_mean_sd = np.nanmean(r["birl"]["posterior_sd"])
        ratio = mce_mean_se / birl_mean_sd if birl_mean_sd > 0 else float("nan")
        print(f"{n_str:>6} {mce_mean_se:>14.4f} {birl_mean_sd:>14.4f} {ratio:>8.2f}")

    print(f"""
Bayesian IRL provides valid uncertainty at any sample size. The
posterior standard deviation shrinks with more data, naturally
reflecting the information content. Asymptotic standard errors from
MCE-IRL assume a well-conditioned Hessian, which may not hold at
small N. When the posterior SD and asymptotic SE agree, the asymptotic
approximation is working. When they diverge, trust the posterior.""")

    # Save results
    out = {
        "true_params": [float(p) for p in true_params],
        "parameter_names": list(env.parameter_names),
        "results_by_n": results,
    }
    path = Path(__file__).parent / "bayesian_irl_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
