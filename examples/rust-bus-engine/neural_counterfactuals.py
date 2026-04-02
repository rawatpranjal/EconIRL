#!/usr/bin/env python3
"""
Neural Counterfactuals on the Rust Bus Engine
==============================================

Part I uses simulated data where the ground truth is linear. The
neural estimator should recover the same counterfactual predictions
as the structural model. Any gap is estimation error, not model
misspecification.

Part II uses real Rust (1987) data where the true reward may have
nonlinearity. Here the neural model is free to differ from the
structural model, and the gap reflects genuine flexibility.

Usage:
    python examples/rust-bus-engine/neural_counterfactuals.py
"""

import numpy as np
import jax.numpy as jnp

from econirl import NFXP
from econirl.estimators.neural_gladius import NeuralGLADIUS
from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.simulation.synthetic import simulate_panel
from econirl.core.types import DDCProblem
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.simulation import (
    neural_global_perturbation,
    neural_local_perturbation,
    neural_transition_counterfactual,
    neural_choice_set_counterfactual,
    neural_sieve_compression,
    neural_policy_jacobian,
    neural_perturbation_sweep,
)


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def fit_models(df, state_col, action_col, id_col, discount, label):
    """Fit NFXP and NeuralGLADIUS on a DataFrame, return rewards and problem."""
    print(f"Fitting NFXP ({label}) ...")
    nfxp = NFXP(n_states=90, n_actions=2, discount=discount, verbose=False)
    nfxp.fit(data=df, state=state_col, action=action_col, id=id_col)
    print(f"  theta_c = {nfxp.params_['theta_c']:.6f}, RC = {nfxp.params_['RC']:.4f}")
    print(f"  P(replace|s=50) = {nfxp.policy_[50, 1]:.4f}")
    print()

    print(f"Fitting NeuralGLADIUS ({label}) ...")
    gladius = NeuralGLADIUS(
        n_actions=2, discount=discount,
        q_hidden_dim=64, q_num_layers=2,
        ev_hidden_dim=64, ev_num_layers=2,
        max_epochs=500, batch_size=256, patience=100,
        bellman_weight=1.0, alternating_updates=True, verbose=True,
    )
    gladius.fit(data=df, state=state_col, action=action_col, id=id_col)
    gladius._n_states = 90  # ensure full state space coverage
    print(f"  Epochs: {gladius.n_epochs_}")
    print(f"  P(replace|s=50) = {gladius.policy_[50, 1]:.4f}")
    print()

    # Reward matrices
    nfxp_reward = jnp.array(nfxp.reward_matrix_)
    neural_reward = jnp.array(gladius.reward_matrix_)

    # Transition tensor
    keep_trans = nfxp.transitions_
    transitions = np.zeros((2, 90, 90))
    transitions[0] = keep_trans
    for s in range(90):
        transitions[1, s, :] = keep_trans[0, :]
    transitions = jnp.array(transitions)

    problem = DDCProblem(
        num_states=90, num_actions=2,
        discount_factor=discount, scale_parameter=1.0,
    )

    return nfxp, gladius, nfxp_reward, neural_reward, transitions, problem


def run_counterfactuals(nfxp_reward, neural_reward, transitions, problem,
                        true_params=None):
    """Run all counterfactual types on both reward surfaces and print comparison."""

    # Feature matrix for sieve compression
    mileage = jnp.arange(90, dtype=jnp.float32)
    features = jnp.zeros((90, 2, 2))
    features = features.at[:, 0, 0].set(-mileage)
    features = features.at[:, 1, 1].set(-1.0)

    # --- A. Global perturbation sweep ---
    banner("Global Perturbation: Penalize Replacement")
    deltas = [0.0, 1.0, 2.0, 5.0, 10.0]
    sweep_n = neural_perturbation_sweep(
        neural_reward, action=1, delta_grid=jnp.array(deltas),
        problem=problem, transitions=transitions,
    )
    sweep_s = neural_perturbation_sweep(
        nfxp_reward, action=1, delta_grid=jnp.array(deltas),
        problem=problem, transitions=transitions,
    )

    print(f"{'Delta':>6s}  {'Neural P(r)':>12s}  {'Struct P(r)':>12s}  {'Gap':>8s}")
    print(f"{'-----':>6s}  {'-----------':>12s}  {'-----------':>12s}  {'---':>8s}")
    for i, d in enumerate(deltas):
        n = sweep_n['mean_action_prob'][i]
        s = sweep_s['mean_action_prob'][i]
        print(f"{d:6.0f}  {n:12.4f}  {s:12.4f}  {n-s:+8.4f}")

    # --- B. Local perturbation (high mileage) ---
    banner("Local Perturbation: Penalize Keep at s > 60")
    mask = jnp.arange(90) > 60
    affected = jnp.where(mask)[0]
    print(f"{'Delta':>6s}  {'Neural P(r|s>60)':>18s}  {'Struct P(r|s>60)':>18s}")
    print(f"{'-----':>6s}  {'----------------':>18s}  {'----------------':>18s}")
    for d in [0.0, 1.0, 2.0, 5.0]:
        rn = neural_local_perturbation(
            neural_reward, action=0, delta=d, state_mask=mask,
            problem=problem, transitions=transitions)
        rs = neural_local_perturbation(
            nfxp_reward, action=0, delta=d, state_mask=mask,
            problem=problem, transitions=transitions)
        np_ = float(rn.counterfactual_policy[affected, 1].mean())
        sp_ = float(rs.counterfactual_policy[affected, 1].mean())
        print(f"{d:6.0f}  {np_:18.4f}  {sp_:18.4f}")

    # --- C. Transition counterfactual ---
    banner("Transition Counterfactual: Faster Depreciation")
    print("Mileage increment probs: (0.39, 0.60, 0.01) -> (0.20, 0.50, 0.30)")
    print()
    new_trans = jnp.zeros_like(transitions)
    new_probs = [0.20, 0.50, 0.30]
    for a in range(2):
        for s in range(90):
            row = jnp.zeros(90)
            base = 0 if a == 1 else s
            for i, p in enumerate(new_probs):
                row = row.at[min(base + i, 89)].add(p)
            new_trans = new_trans.at[a, s, :].set(row)

    rn = neural_transition_counterfactual(neural_reward, new_trans, problem, transitions)
    rs = neural_transition_counterfactual(nfxp_reward, new_trans, problem, transitions)

    print(f"  {'':>30s}  {'Neural':>10s}  {'Struct':>10s}")
    print(f"  {'Baseline P(replace)':>30s}  "
          f"{float(rn.baseline_policy[:, 1].mean()):10.4f}  "
          f"{float(rs.baseline_policy[:, 1].mean()):10.4f}")
    print(f"  {'Counterfactual P(replace)':>30s}  "
          f"{float(rn.counterfactual_policy[:, 1].mean()):10.4f}  "
          f"{float(rs.counterfactual_policy[:, 1].mean()):10.4f}")
    print(f"  {'Welfare change':>30s}  "
          f"{rn.welfare_change:10.2f}  {rs.welfare_change:10.2f}")

    # --- D. Choice set ---
    banner("Choice Set: Mandatory Replace s>80, Warranty s<10")
    mask_cs = jnp.ones((90, 2), dtype=jnp.bool_)
    mask_cs = mask_cs.at[81:, 0].set(False)  # block keep above 80
    mask_cs = mask_cs.at[:10, 1].set(False)  # block replace below 10

    rn = neural_choice_set_counterfactual(neural_reward, mask_cs, problem, transitions)
    rs = neural_choice_set_counterfactual(nfxp_reward, mask_cs, problem, transitions)

    print(f"  {'State':>6s}  {'Neural':>8s}  {'N-CF':>8s}  {'Struct':>8s}  {'S-CF':>8s}  Note")
    print(f"  {'-----':>6s}  {'------':>8s}  {'----':>8s}  {'------':>8s}  {'----':>8s}  ----")
    for s in [5, 9, 10, 50, 79, 80, 85]:
        note = "warranty" if s < 10 else ("mandatory" if s > 80 else "")
        print(f"  {s:6d}  {float(rn.baseline_policy[s,1]):8.4f}  "
              f"{float(rn.counterfactual_policy[s,1]):8.4f}  "
              f"{float(rs.baseline_policy[s,1]):8.4f}  "
              f"{float(rs.counterfactual_policy[s,1]):8.4f}  {note}")

    # --- E. Sieve compression ---
    banner("Sieve Compression: Project Neural Reward onto Linear Features")
    sieve = neural_sieve_compression(
        neural_reward, features, parameter_names=["theta_c", "RC"])

    print(f"  Neural projected: theta_c = {sieve['theta'][0]:.6f}, "
          f"RC = {abs(sieve['theta'][1]):.4f}")
    print(f"  R-squared: {sieve['r_squared']:.4f}")
    if true_params is not None:
        print(f"  True params:      theta_c = {true_params[0]:.6f}, "
              f"RC = {true_params[1]:.4f}")
    print()

    return sieve


def main():
    # ============================================================
    #  PART I: SIMULATED DATA (ground truth is linear)
    # ============================================================
    banner("PART I: SIMULATED DATA — Ground Truth is Linear")
    print("We simulate from the Rust bus DGP with known parameters")
    print("theta_c = 0.001, RC = 3.0. Both NFXP and NeuralGLADIUS")
    print("should recover the same counterfactual predictions because")
    print("the true reward IS linear. Any gap is estimation error.")
    print()

    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0,
        num_mileage_bins=90, discount_factor=0.9999, seed=42,
    )
    panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)
    sim_df = panel.to_dataframe()
    sim_df = sim_df.rename(columns={"state": "mileage_bin", "action": "replaced"})
    n_obs = len(sim_df)
    repl_rate = sim_df["replaced"].mean()
    print(f"Simulated: {n_obs:,} obs, {panel.num_individuals} buses, "
          f"replacement rate {repl_rate:.4f}")
    print()

    nfxp_sim, gladius_sim, nfxp_r_sim, neural_r_sim, trans_sim, prob_sim = \
        fit_models(sim_df, "mileage_bin", "replaced", "id", 0.9999, "simulated")

    # Policy comparison
    banner("PART I — Policy Validation (simulated)")
    operator = SoftBellmanOperator(prob_sim, trans_sim)
    neural_sol = value_iteration(operator, neural_r_sim)
    print(f"  {'State':>6s}  {'NFXP':>8s}  {'Neural':>8s}  {'Diff':>8s}")
    print(f"  {'-----':>6s}  {'----':>8s}  {'------':>8s}  {'----':>8s}")
    for s in [0, 10, 20, 30, 40, 50, 60, 70, 80, 89]:
        np_ = float(nfxp_sim.policy_[s, 1])
        nn_ = float(neural_sol.policy[s, 1])
        print(f"  {s:6d}  {np_:8.4f}  {nn_:8.4f}  {nn_-np_:+8.4f}")
    corr = float(np.corrcoef(nfxp_sim.policy_[:, 1],
                              np.asarray(neural_sol.policy[:, 1]))[0, 1])
    mae = float(np.abs(nfxp_sim.policy_[:, 1] -
                        np.asarray(neural_sol.policy[:, 1])).mean())
    print(f"\n  Correlation: {corr:.4f}    MAE: {mae:.4f}")

    # Counterfactuals
    sieve_sim = run_counterfactuals(
        nfxp_r_sim, neural_r_sim, trans_sim, prob_sim,
        true_params=[0.001, 3.0])

    # ============================================================
    #  PART II: REAL DATA (neural reward may differ)
    # ============================================================
    banner("PART II: REAL DATA — Neural Reward May Capture Nonlinearity")
    print("On real Rust (1987) data the true reward may not be exactly")
    print("linear. The neural model is free to learn a different reward")
    print("surface. Gaps between neural and structural counterfactuals")
    print("reflect genuine model flexibility, not estimation error.")
    print()

    real_df = load_rust_bus()
    n_obs = len(real_df)
    repl_rate = real_df["replaced"].mean()
    print(f"Real data: {n_obs:,} obs, {real_df['bus_id'].nunique()} buses, "
          f"replacement rate {repl_rate:.4f}")
    print()

    nfxp_real, gladius_real, nfxp_r_real, neural_r_real, trans_real, prob_real = \
        fit_models(real_df, "mileage_bin", "replaced", "bus_id", 0.9999, "real")

    # Policy comparison
    banner("PART II — Policy Validation (real)")
    operator = SoftBellmanOperator(prob_real, trans_real)
    neural_sol = value_iteration(operator, neural_r_real)
    print(f"  {'State':>6s}  {'NFXP':>8s}  {'Neural':>8s}  {'Diff':>8s}")
    print(f"  {'-----':>6s}  {'----':>8s}  {'------':>8s}  {'----':>8s}")
    for s in [0, 10, 20, 30, 40, 50, 60, 70, 80, 89]:
        np_ = float(nfxp_real.policy_[s, 1])
        nn_ = float(neural_sol.policy[s, 1])
        print(f"  {s:6d}  {np_:8.4f}  {nn_:8.4f}  {nn_-np_:+8.4f}")
    corr = float(np.corrcoef(nfxp_real.policy_[:, 1],
                              np.asarray(neural_sol.policy[:, 1]))[0, 1])
    mae = float(np.abs(nfxp_real.policy_[:, 1] -
                        np.asarray(neural_sol.policy[:, 1])).mean())
    print(f"\n  Correlation: {corr:.4f}    MAE: {mae:.4f}")

    # Counterfactuals
    sieve_real = run_counterfactuals(
        nfxp_r_real, neural_r_real, trans_real, prob_real)

    # ============================================================
    #  Summary
    # ============================================================
    banner("SUMMARY")
    print(f"  Part I  (sim):  Sieve R^2 = {sieve_sim['r_squared']:.4f}  "
          f"(should be ~1.0 if neural recovers the linear truth)")
    print(f"  Part II (real): Sieve R^2 = {sieve_real['r_squared']:.4f}  "
          f"(lower means neural found nonlinearity)")
    print()
    print("When the true reward is linear (Part I), the neural and")
    print("structural counterfactuals should agree closely. On real")
    print("data (Part II), differences reflect the neural model's")
    print("ability to capture reward nonlinearity that the linear")
    print("specification misses.")


if __name__ == "__main__":
    main()
