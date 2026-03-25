"""
Gridworld Estimation Comparison: MCE IRL vs MaxEnt IRL vs NFXP
===============================================================

Compares three estimation approaches on a gridworld with known reward:

  - MCE IRL (Ziebart 2010): Inverse RL via maximum causal entropy
  - MaxEnt IRL (Ziebart 2008): Inverse RL via maximum entropy
  - NFXP (Rust 1987): Forward MLE via nested fixed point

MCE IRL and MaxEnt IRL recover reward parameters from observed behavior
(inverse problem). NFXP estimates parameters by maximizing the conditional
choice probability likelihood (forward problem). NFXP serves as the gold
standard benchmark since it directly maximizes the data likelihood.

Usage:
    python run_gridworld.py
    python run_gridworld.py --grid-size 8 --n-traj 200 --save-results
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.environments.gridworld import GridworldEnvironment
from econirl.estimation.maxent_irl import MaxEntIRLEstimator
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel


def run_experiment(
    grid_size: int = 5,
    n_traj: int = 100,
    n_periods: int = 30,
    discount: float = 0.95,
    seed: int = 42,
    verbose: bool = True,
    save_results: bool = False,
    output_dir: str = "results",
) -> dict:
    """Run MCE IRL vs MaxEnt IRL vs NFXP on a gridworld.

    Args:
        grid_size: Side length of NxN grid.
        n_traj: Number of expert trajectories.
        n_periods: Periods per trajectory.
        discount: Discount factor.
        seed: Random seed.
        verbose: Print progress.
        save_results: Save results to JSON.
        output_dir: Directory for saved results.

    Returns:
        Dictionary with all results and metrics.
    """
    results = {
        "config": {
            "grid_size": grid_size,
            "n_traj": n_traj,
            "n_periods": n_periods,
            "discount": discount,
            "seed": seed,
        }
    }

    # --- Environment ---
    env = GridworldEnvironment(
        grid_size=grid_size,
        step_penalty=-0.1,
        terminal_reward=10.0,
        distance_weight=0.1,
        discount_factor=discount,
        seed=seed,
    )
    true_params = env.get_true_parameter_vector()

    if verbose:
        print("=" * 70)
        print(f"  Ziebart MCE IRL: {grid_size}x{grid_size} Gridworld")
        print("=" * 70)
        print(f"\nTrue parameters: {dict(zip(env.parameter_names, true_params.tolist()))}")
        print(f"States: {env.num_states}, Actions: {env.num_actions}")

    results["true_parameters"] = dict(zip(env.parameter_names, true_params.tolist()))

    # --- Generate demonstrations ---
    panel = simulate_panel(env=env, n_individuals=n_traj, n_periods=n_periods, seed=seed)

    all_states = panel.get_all_states()
    all_actions = panel.get_all_actions()
    action_names = ["Left", "Right", "Up", "Down", "Stay"]

    demo_stats = {
        "total_observations": panel.num_observations,
        "terminal_visits": int((all_states == env.terminal_state).sum().item()),
        "action_distribution": {
            action_names[a]: int((all_actions == a).sum().item())
            for a in range(env.num_actions)
        },
    }
    results["demo_stats"] = demo_stats

    if verbose:
        print(f"\nDemonstrations: {n_traj} trajectories, {n_periods} periods")
        print(f"Total observations: {demo_stats['total_observations']}")
        tv = demo_stats["terminal_visits"]
        print(f"Terminal visits: {tv} ({tv / demo_stats['total_observations'] * 100:.1f}%)")

    # --- Shared setup ---
    reward_fn = ActionDependentReward(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )
    problem = env.problem_spec
    transitions = env.transition_matrices

    # Compute true policy for comparison
    operator = SoftBellmanOperator(problem, transitions)
    true_reward = reward_fn.compute(true_params)
    true_sol = hybrid_iteration(operator, true_reward, tol=1e-10)
    true_policy = true_sol.policy

    # =====================================================================
    # MCE IRL (Ziebart 2010)
    # =====================================================================
    if verbose:
        print(f"\n{'=' * 50}")
        print("  MCE IRL (Ziebart 2010)")
        print(f"{'=' * 50}")

    config = MCEIRLConfig(
        optimizer="L-BFGS-B",
        outer_max_iter=500,
        outer_tol=1e-8,
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=5000,
        use_adam=False,
        compute_se=False,
        verbose=verbose,
    )

    t0 = time.time()
    mce_estimator = MCEIRLEstimator(config=config)
    mce_result = mce_estimator.estimate(
        panel=panel, utility=reward_fn, problem=problem,
        transitions=transitions, true_params=true_params,
    )
    mce_time = time.time() - t0

    mce_params = mce_result.parameters
    mce_policy = mce_result.policy

    # Metrics
    cos_sim_mce = torch.nn.functional.cosine_similarity(
        mce_params.unsqueeze(0), true_params.unsqueeze(0)
    ).item()
    rmse_mce = torch.sqrt(torch.mean((mce_params - true_params) ** 2)).item()

    eps = 1e-10
    kl_mce = (true_policy * torch.log((true_policy + eps) / (mce_policy + eps))).sum(dim=1).mean().item()
    acc_mce = (true_policy.argmax(1) == mce_policy.argmax(1)).float().mean().item() * 100

    results["mce_irl"] = {
        "parameters": dict(zip(env.parameter_names, mce_params.tolist())),
        "log_likelihood": mce_result.log_likelihood,
        "converged": mce_result.converged,
        "iterations": mce_result.num_iterations,
        "cosine_similarity": cos_sim_mce,
        "rmse": rmse_mce,
        "kl_divergence": kl_mce,
        "policy_accuracy": acc_mce,
        "feature_difference": mce_result.metadata.get("feature_difference") if mce_result.metadata else None,
        "time_seconds": mce_time,
    }

    if verbose:
        print(f"\n  Converged: {mce_result.converged}")
        print(f"  Log-likelihood: {mce_result.log_likelihood:.4f}")
        print(f"  Cosine similarity: {cos_sim_mce:.6f}")
        print(f"  Policy accuracy: {acc_mce:.1f}%")
        print(f"  Time: {mce_time:.2f}s")

    # =====================================================================
    # MaxEnt IRL (Ziebart 2008)
    # =====================================================================
    if verbose:
        print(f"\n{'=' * 50}")
        print("  MaxEnt IRL (Ziebart 2008)")
        print(f"{'=' * 50}")

    t0 = time.time()
    maxent_estimator = MaxEntIRLEstimator(
        inner_solver="policy", inner_tol=1e-10,
        outer_tol=1e-6, outer_max_iter=200,
        compute_hessian=False, verbose=False,
    )
    maxent_result = maxent_estimator.estimate(
        panel=panel, utility=reward_fn, problem=problem, transitions=transitions,
    )
    maxent_time = time.time() - t0

    maxent_params = maxent_result.parameters
    maxent_policy = maxent_result.policy

    cos_sim_maxent = torch.nn.functional.cosine_similarity(
        maxent_params.unsqueeze(0), true_params.unsqueeze(0)
    ).item()
    rmse_maxent = torch.sqrt(torch.mean((maxent_params - true_params) ** 2)).item()
    kl_maxent = (true_policy * torch.log((true_policy + eps) / (maxent_policy + eps))).sum(dim=1).mean().item()
    acc_maxent = (true_policy.argmax(1) == maxent_policy.argmax(1)).float().mean().item() * 100

    results["maxent_irl"] = {
        "parameters": dict(zip(env.parameter_names, maxent_params.tolist())),
        "log_likelihood": maxent_result.log_likelihood,
        "converged": maxent_result.converged,
        "iterations": maxent_result.num_iterations,
        "cosine_similarity": cos_sim_maxent,
        "rmse": rmse_maxent,
        "kl_divergence": kl_maxent,
        "policy_accuracy": acc_maxent,
        "time_seconds": maxent_time,
    }

    if verbose:
        print(f"  Converged: {maxent_result.converged}")
        print(f"  Log-likelihood: {maxent_result.log_likelihood:.4f}")
        print(f"  Cosine similarity: {cos_sim_maxent:.6f}")
        print(f"  Policy accuracy: {acc_maxent:.1f}%")
        print(f"  Time: {maxent_time:.2f}s")

    # =====================================================================
    # NFXP (Rust 1987) — Forward MLE benchmark
    # =====================================================================
    if verbose:
        print(f"\n{'=' * 50}")
        print("  NFXP (Rust 1987) — Forward MLE")
        print(f"{'=' * 50}")

    t0 = time.time()
    nfxp_estimator = NFXPEstimator(
        se_method="asymptotic",
        inner_solver="hybrid",
        inner_tol=1e-10,
        outer_tol=1e-6,
        outer_max_iter=200,
        compute_hessian=False,
        verbose=False,
    )
    nfxp_result = nfxp_estimator.estimate(
        panel=panel, utility=reward_fn, problem=problem, transitions=transitions,
    )
    nfxp_time = time.time() - t0

    nfxp_params = nfxp_result.parameters
    nfxp_policy = nfxp_result.policy

    cos_sim_nfxp = torch.nn.functional.cosine_similarity(
        nfxp_params.unsqueeze(0), true_params.unsqueeze(0)
    ).item()
    rmse_nfxp = torch.sqrt(torch.mean((nfxp_params - true_params) ** 2)).item()
    kl_nfxp = (true_policy * torch.log((true_policy + eps) / (nfxp_policy + eps))).sum(dim=1).mean().item()
    acc_nfxp = (true_policy.argmax(1) == nfxp_policy.argmax(1)).float().mean().item() * 100

    results["nfxp"] = {
        "parameters": dict(zip(env.parameter_names, nfxp_params.tolist())),
        "log_likelihood": nfxp_result.log_likelihood,
        "converged": nfxp_result.converged,
        "iterations": nfxp_result.num_iterations,
        "cosine_similarity": cos_sim_nfxp,
        "rmse": rmse_nfxp,
        "kl_divergence": kl_nfxp,
        "policy_accuracy": acc_nfxp,
        "time_seconds": nfxp_time,
    }

    if verbose:
        print(f"  Converged: {nfxp_result.converged}")
        print(f"  Log-likelihood: {nfxp_result.log_likelihood:.4f}")
        print(f"  Cosine similarity: {cos_sim_nfxp:.6f}")
        print(f"  Policy accuracy: {acc_nfxp:.1f}%")
        print(f"  RMSE: {rmse_nfxp:.6f}")
        print(f"  Time: {nfxp_time:.2f}s")

    # =====================================================================
    # Comparison table
    # =====================================================================
    if verbose:
        print(f"\n{'=' * 70}")
        print("  Comparison: MCE IRL vs MaxEnt IRL vs NFXP")
        print(f"{'=' * 70}")

        print(f"\n  Parameters:")
        print(f"  {'Name':<20} {'True':>10} {'MCE':>10} {'MaxEnt':>10} {'NFXP':>10}")
        print(f"  {'-' * 60}")
        for name in env.parameter_names:
            t = results["true_parameters"][name]
            m = results["mce_irl"]["parameters"][name]
            x = results["maxent_irl"]["parameters"][name]
            n = results["nfxp"]["parameters"][name]
            print(f"  {name:<20} {t:>10.4f} {m:>10.4f} {x:>10.4f} {n:>10.4f}")

        print(f"\n  Metrics:")
        print(f"  {'Metric':<30} {'MCE IRL':>12} {'MaxEnt IRL':>12} {'NFXP':>12}")
        print(f"  {'-' * 66}")
        print(f"  {'Log-likelihood':<30} {results['mce_irl']['log_likelihood']:>12.2f} {results['maxent_irl']['log_likelihood']:>12.2f} {results['nfxp']['log_likelihood']:>12.2f}")
        print(f"  {'Cosine similarity':<30} {cos_sim_mce:>12.6f} {cos_sim_maxent:>12.6f} {cos_sim_nfxp:>12.6f}")
        print(f"  {'RMSE':<30} {rmse_mce:>12.6f} {rmse_maxent:>12.6f} {rmse_nfxp:>12.6f}")
        print(f"  {'KL(true || model)':<30} {kl_mce:>12.6f} {kl_maxent:>12.6f} {kl_nfxp:>12.6f}")
        print(f"  {'Policy accuracy (%)':<30} {acc_mce:>12.1f} {acc_maxent:>12.1f} {acc_nfxp:>12.1f}")
        print(f"  {'Time (s)':<30} {mce_time:>12.2f} {maxent_time:>12.2f} {nfxp_time:>12.2f}")

        print(f"\n  Note: NFXP directly maximizes the data likelihood (forward problem)")
        print(f"  while IRL methods recover reward from behavior (inverse problem).")
        print(f"  IRL rewards are identified up to constants, so cosine similarity")
        print(f"  and policy accuracy matter more than raw parameter RMSE for IRL.")

        # Policy at key states
        print(f"\n  Policy at key states:")
        key_states = [0, grid_size * grid_size // 2, env.terminal_state - 1, env.terminal_state]
        for s in key_states:
            row, col = env.state_to_grid_position(s)
            print(f"\n  State {s} (row={row}, col={col}):")
            print(f"    {'Action':<8} {'True':>8} {'MCE':>8} {'MaxEnt':>8} {'NFXP':>8}")
            for a in range(env.num_actions):
                print(f"    {action_names[a]:<8} {true_policy[s,a].item():>8.4f} {mce_policy[s,a].item():>8.4f} {maxent_policy[s,a].item():>8.4f} {nfxp_policy[s,a].item():>8.4f}")

    # =====================================================================
    # Save results
    # =====================================================================
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"gridworld_{grid_size}x{grid_size}_n{n_traj}.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to {path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ziebart MCE IRL Gridworld Replication")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid side length")
    parser.add_argument("--n-traj", type=int, default=100, help="Number of trajectories")
    parser.add_argument("--n-periods", type=int, default=30, help="Periods per trajectory")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    run_experiment(
        grid_size=args.grid_size,
        n_traj=args.n_traj,
        n_periods=args.n_periods,
        discount=args.discount,
        seed=args.seed,
        save_results=args.save_results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
