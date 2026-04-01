"""
Beijing Taxi: MCE IRL + NFXP on T-Drive Data
=============================================

Full pipeline: load real GPS data → EDA → train/test split →
estimate with MCE IRL and NFXP → in-sample & out-of-sample benchmark.

Usage:
    python run_estimation.py
    python run_estimation.py --n-taxis 200 --grid-size 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.datasets.tdrive_panel import load_tdrive_panel
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.action_reward import ActionDependentReward


def compute_ll(params, reward_fn, problem, transitions, panel):
    """Compute log-likelihood of panel data given parameters."""
    operator = SoftBellmanOperator(problem, transitions)
    reward_matrix = reward_fn.compute(params)
    result = hybrid_iteration(operator, reward_matrix, tol=1e-10)
    log_probs = operator.compute_log_choice_probabilities(reward_matrix, result.V)
    ll = log_probs[panel.get_all_states(), panel.get_all_actions()].sum().item()
    return ll, result.policy


def compute_accuracy(policy, panel):
    """Fraction of observations where predicted best action matches observed."""
    best_actions = policy.argmax(dim=1)
    observed = panel.get_all_actions()
    states = panel.get_all_states()
    predicted = best_actions[states]
    return (predicted == observed).float().mean().item() * 100


def estimate_transitions(panel, n_actions, n_states):
    """Estimate transition matrices from panel data."""
    counts = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
    all_s = panel.get_all_states().numpy()
    all_a = panel.get_all_actions().numpy()
    all_ns = panel.get_all_next_states().numpy()
    for s, a, ns in zip(all_s, all_a, all_ns):
        counts[a, s, ns] += 1

    transitions = torch.zeros((n_actions, n_states, n_states), dtype=torch.float32)
    for a in range(n_actions):
        for s in range(n_states):
            row_sum = counts[a, s, :].sum()
            if row_sum > 0:
                transitions[a, s, :] = torch.tensor(counts[a, s, :] / row_sum, dtype=torch.float32)
            else:
                transitions[a, s, s] = 1.0
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Beijing Taxi MCE IRL + NFXP Benchmark")
    parser.add_argument("--n-taxis", type=int, default=100, help="Number of taxis")
    parser.add_argument("--grid-size", type=int, default=15, help="Grid side length N")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  Beijing Taxi: MCE IRL + NFXP Benchmark")
    print("=" * 70)

    # =================================================================
    # 1. LOAD DATA
    # =================================================================
    print("\n[1/6] Loading T-Drive data...")
    data = load_tdrive_panel(
        n_taxis=args.n_taxis,
        grid_size=args.grid_size,
        discount_factor=args.discount,
        seed=args.seed,
    )

    panel = data["panel"]
    feature_matrix = data["feature_matrix"]
    problem = data["problem"]
    meta = data["metadata"]
    feature_names = data["feature_names"]
    action_names = meta["action_names"]
    grid_size = meta["grid_size"]
    n_states = meta["n_states"]
    n_actions = meta["n_actions"]

    # =================================================================
    # 2. EDA
    # =================================================================
    print(f"\n[2/6] Exploratory Data Analysis")
    print(f"{'=' * 50}")
    print(f"  Grid: {grid_size}x{grid_size} ({n_states} states, {n_actions} actions)")
    print(f"  Taxis: {meta['n_taxis_loaded']}")
    print(f"  Trajectories: {meta['n_trajectories']}")
    print(f"  Observations: {meta['n_observations']:,}")
    print(f"  Features: {feature_names}")

    # Trajectory length stats
    traj_lens = [len(t) for t in panel.trajectories]
    print(f"\n  Trajectory lengths:")
    print(f"    Min: {min(traj_lens)}, Max: {max(traj_lens)}, Mean: {np.mean(traj_lens):.1f}, Median: {np.median(traj_lens):.0f}, Std: {np.std(traj_lens):.1f}")

    # Action distribution
    all_actions = panel.get_all_actions()
    print(f"\n  Action distribution:")
    for a, name in enumerate(action_names):
        count = (all_actions == a).sum().item()
        print(f"    {name:>5}: {count:7d} ({count / len(all_actions) * 100:5.1f}%)")

    # State visit frequency
    all_states = panel.get_all_states()
    state_counts = torch.zeros(n_states)
    for s in all_states:
        state_counts[s] += 1
    visited = (state_counts > 0).sum().item()
    print(f"\n  State coverage: {visited}/{n_states} ({visited/n_states*100:.1f}%) states visited")
    top5 = state_counts.topk(5)
    print(f"  Top 5 states: {', '.join(f's{i.item()}({int(c.item())})' for i, c in zip(top5.indices, top5.values))}")

    # Transition sparsity
    transitions_full = data["transitions"]
    possible = n_actions * n_states * n_states
    nonzero = (transitions_full > 0).sum().item()
    print(f"  Transition sparsity: {nonzero}/{possible} ({nonzero/possible*100:.2f}%) non-zero entries")

    # Text heatmap of state visits
    print(f"\n  State visit heatmap ({grid_size}x{grid_size} grid, N=top, E=right):")
    max_count = state_counts.max().item()
    chars = " .:-=+*#@"
    for row in range(grid_size - 1, -1, -1):  # Top to bottom (N at top)
        line = "    "
        for col in range(grid_size):
            s = row * grid_size + col
            if max_count > 0:
                idx = int(state_counts[s].item() / max_count * (len(chars) - 1))
            else:
                idx = 0
            line += chars[min(idx, len(chars) - 1)]
        print(line)

    # =================================================================
    # 3. TRAIN / TEST SPLIT
    # =================================================================
    print(f"\n[3/6] Train/test split ({args.train_frac:.0%} / {1-args.train_frac:.0%})")
    print(f"{'=' * 50}")

    rng = np.random.RandomState(args.seed)
    n_traj = len(panel.trajectories)
    indices = rng.permutation(n_traj)
    n_train = int(n_traj * args.train_frac)

    train_trajs = [panel.trajectories[i] for i in indices[:n_train]]
    test_trajs = [panel.trajectories[i] for i in indices[n_train:]]
    train_panel = Panel(trajectories=train_trajs)
    test_panel = Panel(trajectories=test_trajs)

    print(f"  Train: {len(train_trajs)} trajectories, {train_panel.num_observations:,} obs")
    print(f"  Test:  {len(test_trajs)} trajectories, {test_panel.num_observations:,} obs")

    # Estimate transitions from TRAIN only
    print(f"  Estimating transitions from train data...")
    train_transitions = estimate_transitions(train_panel, n_actions, n_states)

    # Setup reward function
    reward_fn = ActionDependentReward(
        feature_matrix=feature_matrix,
        parameter_names=feature_names,
    )

    # =================================================================
    # 4. MCE IRL
    # =================================================================
    print(f"\n[4/6] MCE IRL (Ziebart 2010)")
    print(f"{'=' * 50}")

    config = MCEIRLConfig(
        learning_rate=0.05,
        outer_max_iter=500,
        outer_tol=1e-7,
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=5000,
        use_adam=True,
        compute_se=False,
        verbose=True,
    )

    mce = MCEIRLEstimator(config=config)
    t0 = time.time()
    mce_result = mce.estimate(
        panel=train_panel, utility=reward_fn, problem=problem, transitions=train_transitions,
    )
    mce_time = time.time() - t0

    print(f"\n  Converged: {mce_result.converged}, Iterations: {mce_result.num_iterations}, Time: {mce_time:.1f}s")
    print(f"  Parameters:")
    for name, val in zip(feature_names, mce_result.parameters.tolist()):
        print(f"    {name:<20} {val:>10.4f}")

    # Out-of-sample LL
    mce_ll_train, mce_policy = compute_ll(mce_result.parameters, reward_fn, problem, train_transitions, train_panel)
    mce_ll_test, _ = compute_ll(mce_result.parameters, reward_fn, problem, train_transitions, test_panel)

    # =================================================================
    # 5. NFXP
    # =================================================================
    print(f"\n[5/6] NFXP (Rust 1987)")
    print(f"{'=' * 50}")

    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-10,
        outer_tol=1e-6,
        outer_max_iter=200,
        compute_hessian=False,
        verbose=False,
    )
    t0 = time.time()
    nfxp_result = nfxp.estimate(
        panel=train_panel, utility=reward_fn, problem=problem, transitions=train_transitions,
    )
    nfxp_time = time.time() - t0

    print(f"  Converged: {nfxp_result.converged}, Iterations: {nfxp_result.num_iterations}, Time: {nfxp_time:.1f}s")
    print(f"  Parameters:")
    for name, val in zip(feature_names, nfxp_result.parameters.tolist()):
        print(f"    {name:<20} {val:>10.4f}")

    nfxp_ll_train, nfxp_policy = compute_ll(nfxp_result.parameters, reward_fn, problem, train_transitions, train_panel)
    nfxp_ll_test, _ = compute_ll(nfxp_result.parameters, reward_fn, problem, train_transitions, test_panel)

    # =================================================================
    # 6. BENCHMARK TABLE
    # =================================================================
    print(f"\n[6/6] Benchmark Results")
    print(f"{'=' * 70}")

    n_train_obs = train_panel.num_observations
    n_test_obs = test_panel.num_observations

    cos_sim = torch.nn.functional.cosine_similarity(
        mce_result.parameters.unsqueeze(0), nfxp_result.parameters.unsqueeze(0)
    ).item()

    mce_train_acc = compute_accuracy(mce_policy, train_panel)
    mce_test_acc = compute_accuracy(mce_policy, test_panel)
    nfxp_train_acc = compute_accuracy(nfxp_policy, train_panel)
    nfxp_test_acc = compute_accuracy(nfxp_policy, test_panel)

    policy_agreement = (mce_policy.argmax(1) == nfxp_policy.argmax(1)).float().mean().item() * 100

    print(f"\n  Parameters:")
    print(f"  {'Feature':<20} {'MCE IRL':>10} {'NFXP':>10}")
    print(f"  {'-' * 40}")
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {mce_result.parameters[i].item():>10.4f} {nfxp_result.parameters[i].item():>10.4f}")

    print(f"\n  {'Metric':<30} {'MCE IRL':>12} {'NFXP':>12}")
    print(f"  {'-' * 54}")
    print(f"  {'Train LL':<30} {mce_ll_train:>12.2f} {nfxp_ll_train:>12.2f}")
    print(f"  {'Test LL':<30} {mce_ll_test:>12.2f} {nfxp_ll_test:>12.2f}")
    print(f"  {'Train LL/obs':<30} {mce_ll_train/n_train_obs:>12.4f} {nfxp_ll_train/n_train_obs:>12.4f}")
    print(f"  {'Test LL/obs':<30} {mce_ll_test/n_test_obs:>12.4f} {nfxp_ll_test/n_test_obs:>12.4f}")
    print(f"  {'Train accuracy (%)':<30} {mce_train_acc:>12.1f} {nfxp_train_acc:>12.1f}")
    print(f"  {'Test accuracy (%)':<30} {mce_test_acc:>12.1f} {nfxp_test_acc:>12.1f}")
    print(f"  {'Policy agreement (%)':<30} {policy_agreement:>12.1f} {'---':>12}")
    print(f"  {'Cosine sim (params)':<30} {cos_sim:>12.4f} {'---':>12}")
    print(f"  {'Time (s)':<30} {mce_time:>12.1f} {nfxp_time:>12.1f}")

    # Overfitting check
    mce_gap = mce_ll_train / n_train_obs - mce_ll_test / n_test_obs
    nfxp_gap = nfxp_ll_train / n_train_obs - nfxp_ll_test / n_test_obs
    print(f"\n  Generalization gap (train - test LL/obs):")
    print(f"    MCE IRL: {mce_gap:>8.4f}")
    print(f"    NFXP:    {nfxp_gap:>8.4f}")
    print(f"    (smaller = better generalization)")

    if args.save_results:
        results = {
            "config": vars(args),
            "eda": {"n_traj": n_traj, "n_train": n_train, "n_test": n_traj - n_train,
                    "n_train_obs": n_train_obs, "n_test_obs": n_test_obs},
            "mce_irl": {
                "parameters": dict(zip(feature_names, mce_result.parameters.tolist())),
                "train_ll": mce_ll_train, "test_ll": mce_ll_test,
                "train_acc": mce_train_acc, "test_acc": mce_test_acc, "time": mce_time,
            },
            "nfxp": {
                "parameters": dict(zip(feature_names, nfxp_result.parameters.tolist())),
                "train_ll": nfxp_ll_train, "test_ll": nfxp_ll_test,
                "train_acc": nfxp_train_acc, "test_acc": nfxp_test_acc, "time": nfxp_time,
            },
            "comparison": {"cosine_similarity": cos_sim, "policy_agreement": policy_agreement},
        }
        os.makedirs("results", exist_ok=True)
        path = f"results/benchmark_{args.grid_size}x{args.grid_size}_n{args.n_taxis}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {path}")


if __name__ == "__main__":
    main()
