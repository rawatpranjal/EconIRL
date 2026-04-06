"""SEES: Sieve compression of the state space.

Demonstrates the key contribution of Luo and Sang (2024):
1. NFXP requires O(S^2) transition matrices and O(S) inner VI per
   optimizer step. On large state spaces this is both memory-intensive
   and slow.
2. NNES replaces VI with a neural V-network, but still requires
   training the network.
3. SEES approximates V(s) with K closed-form sieve basis functions
   (Fourier or polynomial), making complexity O(K) independent of
   the state space size. No neural network training required.

The showcase runs SEES, NFXP, and NNES on a multi-component bus with
400 states (K=2, M=20), then sweeps basis dimension K to show the
accuracy-compression tradeoff.

Usage:
    python examples/multi-component-bus/sees_sieve_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.multi_component_bus import MultiComponentBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.nnes import NNESEstimator, NNESConfig
from econirl.estimation.sees import SEESEstimator, SEESConfig
from econirl.preferences.linear import LinearUtility


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
    print("SEES: Sieve Compression of the State Space")
    print("Luo and Sang (2024)")
    print("=" * 70)
    print()
    print("NFXP solves the full Bellman equation (exact, O(S) per step).")
    print("NNES approximates V with a neural network (fast, needs training).")
    print("SEES approximates V with K basis functions (fast, closed-form).")

    # Multi-component bus: K=2, M=20 = 400 states
    env = MultiComponentBusEnvironment(K=2, M=20, discount_factor=0.99)
    panel = env.generate_panel(n_individuals=500, n_periods=100, seed=42)
    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )
    true_params = env.get_true_parameter_vector()

    print(f"\n  Components: K=2, Bins per component: M=20")
    print(f"  State space: {env.num_states} states")
    print(f"  Compression: {env.num_states} states -> K basis functions")
    print(f"  True parameters: {env.true_parameters}")

    run_diagnostics(panel, problem, utility)

    results = {}

    # ---- NFXP (gold standard) ----
    print("\n--- NFXP (exact Bellman inner loop) ---")
    t0 = time.time()
    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-10,
        inner_max_iter=100000,
        se_method="robust",
        verbose=False,
    )
    nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    nfxp_p = np.array(nfxp_result.parameters)
    print(f"  Time: {nfxp_time:.1f}s")
    print(f"  LL: {nfxp_result.log_likelihood:.4f}")
    for name, val, true in zip(
        env.parameter_names, nfxp_p, np.array(true_params)
    ):
        print(f"  {name}: {val:.6f} (true: {true:.6f})")

    results["nfxp"] = {
        "params": nfxp_p.tolist(),
        "ll": float(nfxp_result.log_likelihood),
        "time": nfxp_time,
    }

    # ---- NNES (neural V-network) ----
    print("\n--- NNES (neural V-network, NPL-based) ---")
    t0 = time.time()
    nnes = NNESEstimator(config=NNESConfig(
        hidden_dim=64,
        num_layers=2,
        v_epochs=200,
        v_lr=0.001,
        verbose=False,
    ))
    nnes_result = nnes.estimate(panel, utility, problem, transitions)
    nnes_time = time.time() - t0
    nnes_p = np.array(nnes_result.parameters)
    print(f"  Time: {nnes_time:.1f}s")
    print(f"  LL: {nnes_result.log_likelihood:.4f}")
    for name, val in zip(env.parameter_names, nnes_p):
        print(f"  {name}: {val:.6f}")

    results["nnes"] = {
        "params": nnes_p.tolist(),
        "ll": float(nnes_result.log_likelihood),
        "time": nnes_time,
    }

    # ---- SEES (sieve basis) ----
    print("\n--- SEES (Fourier basis, K=8) ---")
    t0 = time.time()
    sees = SEESEstimator(
        basis_type="fourier",
        basis_dim=8,
        penalty_weight=10.0,
        se_method="asymptotic",
        verbose=False,
    )
    sees_result = sees.estimate(panel, utility, problem, transitions)
    sees_time = time.time() - t0
    sees_p = np.array(sees_result.parameters)
    print(f"  Time: {sees_time:.1f}s")
    print(f"  LL: {sees_result.log_likelihood:.4f}")
    print(f"  Compression: {env.num_states} states -> 8 basis functions "
          f"({env.num_states / 8:.0f}x)")
    for name, val in zip(env.parameter_names, sees_p):
        print(f"  {name}: {val:.6f}")

    results["sees_k8"] = {
        "params": sees_p.tolist(),
        "ll": float(sees_result.log_likelihood),
        "time": sees_time,
    }

    # ---- Basis dimension sweep ----
    print("\n" + "=" * 70)
    print("Basis Dimension Sweep: Accuracy vs Compression")
    print("=" * 70)

    basis_dims = [4, 6, 8, 12, 16, 20]
    sweep_results = {}

    for K in basis_dims:
        t0 = time.time()
        sees_k = SEESEstimator(
            basis_type="fourier",
            basis_dim=K,
            penalty_weight=10.0,
            verbose=False,
        )
        result_k = sees_k.estimate(panel, utility, problem, transitions)
        elapsed = time.time() - t0
        p_k = np.array(result_k.parameters)
        rmse = float(np.sqrt(np.mean((p_k - nfxp_p) ** 2)))
        sweep_results[K] = {
            "params": p_k.tolist(),
            "ll": float(result_k.log_likelihood),
            "time": elapsed,
            "rmse_vs_nfxp": rmse,
        }

    print(f"\n{'K':>5} {'Compress':>10} {'Time (s)':>10} {'LL':>12} "
          f"{'RMSE vs NFXP':>14}")
    print("-" * 55)
    for K, r in sweep_results.items():
        compress = f"{env.num_states / K:.0f}x"
        print(f"{K:>5} {compress:>10} {r['time']:>10.1f} {r['ll']:>12.2f} "
              f"{r['rmse_vs_nfxp']:>14.6f}")

    results["basis_sweep"] = sweep_results

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Three-Way Comparison at 400 States")
    print("=" * 70)

    print(f"\n{'Method':>12} {'Time (s)':>10} {'LL':>12} ", end="")
    for name in env.parameter_names:
        print(f"  {name:>12}", end="")
    print()
    print("-" * (36 + 14 * len(env.parameter_names)))

    for label, r in [("NFXP", results["nfxp"]),
                     ("NNES", results["nnes"]),
                     ("SEES (K=8)", results["sees_k8"])]:
        print(f"{label:>12} {r['time']:>10.1f} {r['ll']:>12.2f} ", end="")
        for val in r["params"]:
            print(f"  {val:>12.6f}", end="")
        print()

    print(f"\n{'True':>12} {'':>10} {'':>12} ", end="")
    for val in np.array(true_params):
        print(f"  {float(val):>12.6f}", end="")
    print()

    print(f"""
SEES compresses {env.num_states} states into K basis functions. With K=8 Fourier
terms, the value function is represented by 8 coefficients instead of
{env.num_states} state values. The basis dimension sweep shows the
accuracy-compression tradeoff: too few basis functions underfit V(s),
while too many lose the computational advantage over NFXP.""")

    # Save results
    path = Path(__file__).parent / "sees_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
