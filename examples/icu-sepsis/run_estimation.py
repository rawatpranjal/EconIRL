"""ICU-Sepsis: Clinical Treatment IRL on Real Patient Data

Demonstrates inverse reinforcement learning on the ICU-Sepsis benchmark
MDP derived from MIMIC-III records. The goal is to recover the implicit
reward function driving ICU clinicians' decisions about IV fluid and
vasopressor dosing for sepsis patients.

Estimators:
  NFXP      -- structural, model-based, linear utility
  CCP       -- reduced-form, model-based, linear utility
  NeuralGLADIUS -- model-free neural IRL, projects onto linear spec

Counterfactual analysis shows what happens when vasopressor cost doubles.

Run:
    python examples/icu-sepsis/run_estimation.py
"""

import json
import time
from pathlib import Path

import econirl._jax_config  # enable float64 before any JAX ops
import jax.numpy as jnp
import numpy as np
import torch

from econirl.core.types import DDCProblem, Panel
from econirl.datasets.icu_sepsis import load_icu_sepsis
from econirl.environments.icu_sepsis import ICUSepsisEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimators.neural_gladius import NeuralGLADIUS
from econirl.inference import etable
from econirl.inference.fit_metrics import brier_score, kl_divergence
from econirl.inference.hypothesis_tests import vuong_test
from econirl.preferences.linear import LinearUtility
from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis


def main():
    print("=" * 65)
    print("ICU-Sepsis: Clinical Treatment IRL (716 states, 25 actions)")
    print("=" * 65)

    env = ICUSepsisEnvironment(discount_factor=0.99)

    print("Generating expert demonstrations from clinician policy...")
    panel = load_icu_sepsis(n_individuals=2000, max_steps=20, as_panel=True, seed=42)
    cutoff = int(panel.num_individuals * 0.8)
    train = Panel(trajectories=panel.trajectories[:cutoff])
    test = Panel(trajectories=panel.trajectories[cutoff:])
    print(f"  Train: {train.num_individuals} patients, {train.num_observations} obs")
    print(f"  Test:  {test.num_individuals} patients, {test.num_observations} obs")

    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    # ── Structural estimators (model-based) ────────────────────────────

    results = {}
    for name, EstCls in [("NFXP", NFXPEstimator), ("CCP", CCPEstimator)]:
        t0 = time.time()
        est = EstCls()
        r = est.estimate(train, utility, problem, transitions)
        dt = time.time() - t0
        results[name] = r
        print(f"\n{name}: {dt:.1f}s, converged={r.converged}")
        for pname, val in zip(env.parameter_names, r.parameters):
            print(f"  {pname}: {float(val):.4f}")

    # ── NeuralGLADIUS (model-free IRL) ─────────────────────────────────
    # State encoder: map discrete state index to its SOFA severity score.
    # SOFA scores (0-1) are more informative than raw normalized indices
    # because they capture patient severity rather than cluster ordering.

    mdp_path = Path(__file__).parent.parent.parent / "src" / "econirl" / "datasets" / "icu_sepsis_mdp.npz"
    sofa_scores = torch.tensor(np.load(mdp_path)["sofa_scores"], dtype=torch.float32)
    sofa_encoder = lambda s: sofa_scores[s].unsqueeze(-1)  # (B,) → (B, 1)

    feat_tensor = torch.tensor(np.array(env.feature_matrix), dtype=torch.float32)

    t0 = time.time()
    gladius = NeuralGLADIUS(
        n_actions=env.num_actions,
        discount=0.99,
        q_hidden_dim=64,
        q_num_layers=2,
        ev_hidden_dim=64,
        ev_num_layers=2,
        max_epochs=300,
        batch_size=512,
        lr=5e-4,
        patience=40,
        state_encoder=sofa_encoder,
        state_dim=1,
        feature_names=env.parameter_names,
        verbose=True,
    )
    gladius.fit(data=train, features=feat_tensor)
    dt = time.time() - t0
    print(f"\nNeuralGLADIUS: {dt:.1f}s, converged={gladius.converged_}, "
          f"epochs={gladius.n_epochs_}, projection R²={gladius.projection_r2_:.3f}")
    for pname, val in (gladius.params_ or {}).items():
        print(f"  {pname}: {val:.4f}")

    # ── Standard errors (NFXP) ─────────────────────────────────────────

    print("\nStandard errors (NFXP):")
    se = results["NFXP"].standard_errors
    if se is not None:
        for pname, val in zip(env.parameter_names, se):
            print(f"  {pname}: {float(val):.4f}")

    # ── Post-estimation diagnostics ────────────────────────────────────

    print("\n" + "=" * 65)
    print("Post-Estimation Diagnostics")
    print("=" * 65)

    print("\n--- etable() ---")
    print(etable(results["NFXP"], results["CCP"]))

    obs_states = jnp.array(train.get_all_states())
    obs_actions = jnp.array(train.get_all_actions())

    print("\n--- Brier Scores ---")
    for name, r in results.items():
        bs = brier_score(r.policy, obs_states, obs_actions)
        print(f"  {name}: {bs['brier_score']:.4f}")

    gladius_policy = jnp.array(gladius.policy_)
    bs_g = brier_score(gladius_policy, obs_states, obs_actions)
    print(f"  NeuralGLADIUS: {bs_g['brier_score']:.4f}")

    print("\n--- Vuong Test (NFXP vs CCP) ---")
    vt = vuong_test(results["NFXP"].policy, results["CCP"].policy, obs_states, obs_actions)
    print(f"  Z-statistic: {vt['statistic']:.3f}, P-value: {vt['p_value']:.4f}, "
          f"Direction: {vt['direction']}")

    # ── Counterfactual: Double vasopressor weight ──────────────────────

    print("\n" + "=" * 65)
    print("Counterfactual: Double vasopressor weight")
    print("=" * 65)

    best = results["NFXP"]
    vaso_idx = env.parameter_names.index("vaso_weight")
    new_params = best.parameters.at[vaso_idx].set(best.parameters[vaso_idx] * 2)
    cf = counterfactual_policy(best, new_params, utility, problem, transitions)
    print(f"Welfare change: {float(cf.welfare_change):+.4f}")

    baseline_mean = np.array(cf.baseline_policy[:713]).mean(axis=0)
    counter_mean = np.array(cf.counterfactual_policy[:713]).mean(axis=0)

    print(f"\n{'Vaso Level':>12} {'Baseline':>10} {'Counter':>10} {'Change':>10}")
    print("-" * 44)
    for vl in range(5):
        bp = sum(baseline_mean[fl * 5 + vl] for fl in range(5))
        cp = sum(counter_mean[fl * 5 + vl] for fl in range(5))
        print(f"{'Level ' + str(vl):>12} {bp:>10.3f} {cp:>10.3f} {cp - bp:>+10.3f}")

    # ── Elasticity ─────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("Elasticity: SOFA weight sensitivity")
    print("=" * 65)
    ea = elasticity_analysis(
        best, utility, problem, transitions,
        parameter_name="sofa_weight",
        pct_changes=[-0.50, -0.25, 0.25, 0.50],
    )
    print(f"Baseline sofa_weight: {float(ea['baseline_value']):.4f}")
    print(f"{'% Change':>10} {'Welfare Δ':>12} {'Avg Policy Δ':>14}")
    print("-" * 38)
    for i, pct in enumerate(ea["pct_changes"]):
        wc = ea["welfare_changes"][i]
        pc = ea["policy_changes"][i]
        print(f"{pct:>+10.0%} {float(wc):>12.4f} {float(pc):>14.4f}")

    # ── Save results ───────────────────────────────────────────────────

    out = {
        "parameters": {},
        "standard_errors": {},
        "log_likelihoods": {},
        "gladius": {
            "params": gladius.params_,
            "projection_r2": gladius.projection_r2_,
            "n_epochs": gladius.n_epochs_,
            "converged": gladius.converged_,
        },
        "counterfactual": {"welfare_change": float(cf.welfare_change)},
        "elasticity": {
            "baseline_sofa_weight": float(ea["baseline_value"]),
            "pct_changes": [float(p) for p in ea["pct_changes"]],
            "welfare_changes": [float(w) for w in ea["welfare_changes"]],
        },
    }
    for name, r in results.items():
        out["parameters"][name] = {
            pname: float(r.parameters[i])
            for i, pname in enumerate(env.parameter_names)
        }
        out["standard_errors"][name] = {
            pname: float(r.standard_errors[i])
            for i, pname in enumerate(env.parameter_names)
        } if r.standard_errors is not None else {}
        out["log_likelihoods"][name] = float(r.log_likelihood)
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
