#!/usr/bin/env python3
"""Counterfactual analysis on simulated Rust bus data.

The ground truth is linear (theta_c=0.001, RC=3.0). Both NFXP and
NeuralGLADIUS should recover the same counterfactual predictions.
Any gap is estimation error, not model misspecification.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from econirl import NFXP
from econirl.estimators.neural_gladius import NeuralGLADIUS
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

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "_static"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN = "#1f77b4", "#ff7f0e", "#2ca02c"


def banner(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def main():
    # ── 1. Simulate data ─────────────────────────────────────────────
    banner("1. SIMULATE DATA AND FIT MODELS")
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0,
        num_mileage_bins=90, discount_factor=0.9999, seed=42)
    panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)
    df = panel.to_dataframe().rename(columns={"state": "mileage_bin", "action": "replaced"})
    print(f"Simulated: {len(df):,} obs, {panel.num_individuals} buses, "
          f"replacement rate {df['replaced'].mean():.4f}")

    # Fit NFXP
    nfxp = NFXP(n_states=90, n_actions=2, discount=0.9999, verbose=False)
    nfxp.fit(df, state="mileage_bin", action="replaced", id="id")
    print(f"NFXP:  theta_c={nfxp.params_['theta_c']:.6f}, RC={nfxp.params_['RC']:.4f}")

    # Fit NeuralGLADIUS
    gladius = NeuralGLADIUS(
        n_actions=2, discount=0.9999,
        q_hidden_dim=64, q_num_layers=2,
        ev_hidden_dim=64, ev_num_layers=2,
        max_epochs=500, batch_size=256, patience=100,
        bellman_weight=1.0, alternating_updates=True, verbose=True)
    gladius.fit(df, state="mileage_bin", action="replaced", id="id")
    gladius._n_states = 90
    print(f"GLADIUS: epochs={gladius.n_epochs_}")
    print(f"True:  theta_c=0.001000, RC=3.0000")

    # Build shared objects
    nfxp_r = jnp.array(nfxp.reward_matrix_)
    neural_r = jnp.array(gladius.reward_matrix_)
    keep_t = nfxp.transitions_
    trans = np.zeros((2, 90, 90))
    trans[0] = keep_t
    for s in range(90):
        trans[1, s, :] = keep_t[0, :]
    trans = jnp.array(trans)
    prob = DDCProblem(num_states=90, num_actions=2, discount_factor=0.9999, scale_parameter=1.0)

    # True policy from environment
    true_params = jnp.array([env.true_parameters["operating_cost"],
                              env.true_parameters["replacement_cost"]])
    true_r = jnp.einsum("sak,k->sa", jnp.array(env.feature_matrix), true_params)
    op = SoftBellmanOperator(prob, jnp.array(env.transition_matrices))
    true_sol = value_iteration(op, true_r)
    true_pi = true_sol.policy

    # Neural policy from Bellman solve on neural reward
    op2 = SoftBellmanOperator(prob, trans)
    neural_sol = value_iteration(op2, neural_r)
    nfxp_sol = value_iteration(op2, nfxp_r)

    # ── 2. Reward heatmap ────────────────────────────────────────────
    banner("2. REWARD HEATMAP")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = ["NFXP (structural)", "NeuralGLADIUS", "Difference"]
    data = [np.asarray(nfxp_r), np.asarray(neural_r),
            np.asarray(neural_r - nfxp_r)]
    for i, (ax, d, lab) in enumerate(zip(axes, data, labels)):
        vmax = max(abs(d.min()), abs(d.max())) if i == 2 else None
        cmap = "RdBu_r" if i == 2 else "viridis"
        kwargs = dict(cmap=cmap)
        if i == 2:
            kwargs["vmin"], kwargs["vmax"] = -vmax, vmax
        im = ax.imshow(d, aspect="auto", **kwargs)
        ax.set_xlabel("Action")
        ax.set_ylabel("Mileage bin")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Keep", "Replace"])
        ax.set_title(lab)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Reward surfaces: simulated data (true reward is linear)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_reward_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_reward_heatmap.png")

    # ── 3. Policy comparison ─────────────────────────────────────────
    banner("3. POLICY COMPARISON")
    states = np.arange(90)
    corr = float(np.corrcoef(nfxp.policy_[:, 1], np.asarray(neural_sol.policy[:, 1]))[0, 1])
    mae = float(np.abs(nfxp.policy_[:, 1] - np.asarray(neural_sol.policy[:, 1])).mean())
    print(f"Policy correlation: {corr:.4f}")
    print(f"Mean absolute error: {mae:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(states, np.asarray(true_pi[:, 1]), "--", color=GREEN, lw=2, label="True")
    ax.plot(states, nfxp.policy_[:, 1], color=BLUE, lw=1.5, label=f"NFXP (corr={corr:.2f})")
    ax.plot(states, np.asarray(neural_sol.policy[:, 1]), color=ORANGE, lw=1.5, label="Neural")
    ax.set_xlabel("Mileage bin")
    ax.set_ylabel("P(replace)")
    ax.set_title("Replacement probability: simulated data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_policy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_policy_comparison.png")

    # ── 4. Global perturbation sweep ─────────────────────────────────
    banner("4. GLOBAL PERTURBATION SWEEP")
    deltas = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    sw_n = neural_perturbation_sweep(neural_r, action=1, delta_grid=deltas, problem=prob, transitions=trans)
    sw_s = neural_perturbation_sweep(nfxp_r, action=1, delta_grid=deltas, problem=prob, transitions=trans)

    print(f"{'Delta':>6s}  {'Neural':>10s}  {'Struct':>10s}  {'Gap':>8s}")
    for i, d in enumerate(np.asarray(deltas)):
        n, s = sw_n['mean_action_prob'][i], sw_s['mean_action_prob'][i]
        print(f"{d:6.1f}  {n:10.4f}  {s:10.4f}  {n-s:+8.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(np.asarray(deltas), sw_s['mean_action_prob'], 'o-', color=BLUE, label="Structural")
    ax1.plot(np.asarray(deltas), sw_n['mean_action_prob'], 's-', color=ORANGE, label="Neural")
    ax1.set_xlabel("Replacement penalty (delta)")
    ax1.set_ylabel("Mean P(replace)")
    ax1.set_title("Policy response to replacement cost increase")
    ax1.legend()

    ax2.plot(np.asarray(deltas), sw_s['welfare'], 'o-', color=BLUE, label="Structural")
    ax2.plot(np.asarray(deltas), sw_n['welfare'], 's-', color=ORANGE, label="Neural")
    ax2.set_xlabel("Replacement penalty (delta)")
    ax2.set_ylabel("Welfare")
    ax2.set_title("Welfare response")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_global_perturbation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_global_perturbation.png")

    # ── 5. Local perturbation ────────────────────────────────────────
    banner("5. LOCAL PERTURBATION (states > 60)")
    mask = jnp.arange(90) > 60
    affected = jnp.where(mask)[0]
    loc_deltas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    n_probs, s_probs = [], []
    for d in loc_deltas:
        rn = neural_local_perturbation(neural_r, 0, d, mask, prob, trans)
        rs = neural_local_perturbation(nfxp_r, 0, d, mask, prob, trans)
        n_probs.append(float(rn.counterfactual_policy[affected, 1].mean()))
        s_probs.append(float(rs.counterfactual_policy[affected, 1].mean()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loc_deltas, s_probs, 'o-', color=BLUE, label="Structural")
    ax.plot(loc_deltas, n_probs, 's-', color=ORANGE, label="Neural")
    ax.set_xlabel("Operating cost penalty at s > 60")
    ax.set_ylabel("P(replace | s > 60)")
    ax.set_title("Local perturbation: high-mileage operating cost increase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_local_perturbation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_local_perturbation.png")

    # ── 6. Transition counterfactual ─────────────────────────────────
    banner("6. TRANSITION COUNTERFACTUAL")
    new_trans = jnp.zeros_like(trans)
    for a in range(2):
        for s in range(90):
            row = jnp.zeros(90)
            base = 0 if a == 1 else s
            for i, p in enumerate([0.20, 0.50, 0.30]):
                row = row.at[min(base + i, 89)].add(p)
            new_trans = new_trans.at[a, s, :].set(row)

    rn = neural_transition_counterfactual(neural_r, new_trans, prob, trans)
    rs = neural_transition_counterfactual(nfxp_r, new_trans, prob, trans)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(states, np.asarray(rs.baseline_policy[:, 1]), color=BLUE, lw=1.5, label="Struct baseline")
    ax.plot(states, np.asarray(rs.counterfactual_policy[:, 1]), '--', color=BLUE, lw=1.5, label="Struct CF")
    ax.plot(states, np.asarray(rn.baseline_policy[:, 1]), color=ORANGE, lw=1.5, label="Neural baseline")
    ax.plot(states, np.asarray(rn.counterfactual_policy[:, 1]), '--', color=ORANGE, lw=1.5, label="Neural CF")
    ax.set_xlabel("Mileage bin")
    ax.set_ylabel("P(replace)")
    ax.set_title("Transition counterfactual: faster depreciation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_transition_cf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_transition_cf.png")
    print(f"Neural welfare change: {rn.welfare_change:.2f}")
    print(f"Struct welfare change: {rs.welfare_change:.2f}")

    # ── 7. Choice set counterfactual ─────────────────────────────────
    banner("7. CHOICE SET COUNTERFACTUAL")
    mask_cs = jnp.ones((90, 2), dtype=jnp.bool_)
    mask_cs = mask_cs.at[81:, 0].set(False)   # mandatory replace above 80
    mask_cs = mask_cs.at[:10, 1].set(False)    # warranty below 10

    rn = neural_choice_set_counterfactual(neural_r, mask_cs, prob, trans)
    rs = neural_choice_set_counterfactual(nfxp_r, mask_cs, prob, trans)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, res, title in [(ax1, rs, "Structural"), (ax2, rn, "Neural")]:
        ax.plot(states, np.asarray(res.baseline_policy[:, 1]), lw=1.5, label="Baseline")
        ax.plot(states, np.asarray(res.counterfactual_policy[:, 1]), '--', lw=1.5, label="Constrained")
        ax.axvspan(0, 9, alpha=0.1, color="red", label="Warranty zone")
        ax.axvspan(81, 89, alpha=0.1, color="green", label="Mandatory zone")
        ax.set_xlabel("Mileage bin")
        ax.set_ylabel("P(replace)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Choice set counterfactual: warranty + mandatory replacement")
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_choice_set.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_choice_set.png")

    # ── 8. Sieve compression ────────────────────────────────────────
    banner("8. SIEVE COMPRESSION")
    mileage = jnp.arange(90, dtype=jnp.float32)
    features = jnp.zeros((90, 2, 2))
    features = features.at[:, 0, 0].set(-mileage)
    features = features.at[:, 1, 1].set(-1.0)

    sieve = neural_sieve_compression(neural_r, features, parameter_names=["theta_c", "RC"])
    print(f"Neural projected: theta_c={sieve['theta'][0]:.6f}, RC={abs(sieve['theta'][1]):.4f}")
    print(f"R-squared: {sieve['r_squared']:.4f}")
    print(f"True:      theta_c=0.001000, RC=3.0000")

    # Scatter: neural vs structural reward differences
    neural_diff = np.asarray(neural_r[:, 1] - neural_r[:, 0])
    struct_diff = np.asarray(nfxp_r[:, 1] - nfxp_r[:, 0])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(struct_diff, neural_diff, alpha=0.6, edgecolors="k", lw=0.5)
    lims = [min(struct_diff.min(), neural_diff.min()), max(struct_diff.max(), neural_diff.max())]
    ax.plot(lims, lims, '--', color="gray", label="45-degree line")
    ax.set_xlabel("Structural reward difference r(s,replace) - r(s,keep)")
    ax.set_ylabel("Neural reward difference")
    ax.set_title(f"Sieve compression (R² = {sieve['r_squared']:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_sieve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_sieve.png")

    # ── 9. Policy Jacobian ──────────────────────────────────────────
    banner("9. POLICY JACOBIAN")
    print("Computing Jacobian (180 Bellman solves) ...")
    J = neural_policy_jacobian(neural_r, prob, trans, target_action=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Show J[s, s', replace] — how r(s', replace) affects P(replace|s)
    im = ax.imshow(np.asarray(J[:, :, 1]), aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("Perturbed state s'")
    ax.set_ylabel("Affected state s")
    ax.set_title("Policy Jacobian: ∂P(replace|s) / ∂r(s', replace)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "sim_bus_jacobian.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sim_bus_jacobian.png")

    # ── Summary ─────────────────────────────────────────────────────
    banner("SUMMARY")
    print(f"Sieve R-squared:    {sieve['r_squared']:.4f}")
    print(f"Policy correlation: {corr:.4f}")
    print(f"Policy MAE:         {mae:.4f}")
    print()
    print("On simulated linear data, the neural model should match the")
    print("structural model. Any remaining gap is estimation error from")
    print("the neural network, not model misspecification.")


if __name__ == "__main__":
    main()
