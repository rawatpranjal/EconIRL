#!/usr/bin/env python3
"""Autolab: NGSIM lane-change IRL benchmark.

Runs NFXP-NK, MaxEnt IRL, MCE IRL, and AIRL on NGSIM US-101 data,
one estimator at a time. Appends results to CSV. Skips completed.

Usage:
    python scripts/run_ngsim_autolab.py
"""

import csv
import os
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "ngsim_benchmark.csv")
TIMEOUT = 300  # 5 min per estimator
MAX_VEHICLES = 500

CSV_COLUMNS = [
    "estimator", "time_seconds", "converged", "log_likelihood",
    "speed", "accel_cost", "headway_risk", "collision_risk", "lane_change_cost",
    "status",
]


def load_ngsim_problem():
    """Load NGSIM data and build DDC problem with structural features."""
    from econirl.datasets.ngsim import N_LANES, N_SPEED_BINS, N_ACTIONS
    from econirl.core.types import DDCProblem, Panel, Trajectory
    from econirl.preferences.linear import LinearUtility

    # Import the feature computation from the example
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "ngsim-lane-change"))
    from run_ngsim_mce_irl import load_and_compute_features, build_feature_matrix

    print("Loading NGSIM data...")
    df = load_and_compute_features(max_vehicles=MAX_VEHICLES)
    print(f"  {len(df)} observations, {df.vehicle_id.nunique()} vehicles")

    n_states = N_LANES * N_SPEED_BINS
    n_actions = N_ACTIONS

    problem = DDCProblem(num_states=n_states, num_actions=n_actions,
                         discount_factor=0.99, scale_parameter=1.0)

    # Transitions
    transitions = torch.zeros(n_actions, n_states, n_states)
    for _, row in df.iterrows():
        s, a, ns = int(row.state), int(row.action), int(row.next_state)
        if 0 <= s < n_states and 0 <= ns < n_states and 0 <= a < n_actions:
            transitions[a, s, ns] += 1
    for a in range(n_actions):
        row_sums = transitions[a].sum(dim=1, keepdim=True).clamp(min=1)
        transitions[a] = transitions[a] / row_sums
    for a in range(n_actions):
        zero_rows = transitions[a].sum(dim=1) < 0.5
        if zero_rows.any():
            transitions[a, zero_rows] = 1.0 / n_states

    # Features
    features, feature_names = build_feature_matrix(df)
    utility = LinearUtility(feature_matrix=features, parameter_names=feature_names)

    # Panel
    trajectories = []
    for vid in df["vehicle_id"].unique():
        vdata = df[df["vehicle_id"] == vid].sort_values("period")
        traj = Trajectory(
            states=torch.tensor(vdata["state"].values, dtype=torch.long),
            actions=torch.tensor(vdata["action"].values, dtype=torch.long),
            next_states=torch.tensor(vdata["next_state"].values, dtype=torch.long),
            individual_id=int(vid),
        )
        trajectories.append(traj)
    panel = Panel(trajectories=trajectories)

    return problem, transitions, utility, panel, feature_names


def get_completed():
    """Return set of completed estimator names."""
    if not os.path.exists(CSV_PATH):
        return set()
    done = set()
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row.get("status") == "OK":
                done.add(row["estimator"])
    return done


def append_result(row):
    """Append result to CSV."""
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_estimator(name, estimator, panel, utility, problem, transitions, feature_names):
    """Run one estimator with timeout."""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    row = {c: "" for c in CSV_COLUMNS}
    row["estimator"] = name

    t0 = time.time()
    try:
        result = estimator.estimate(
            panel=panel, utility=utility, problem=problem, transitions=transitions
        )
        elapsed = time.time() - t0
        row["time_seconds"] = f"{elapsed:.1f}"
        row["converged"] = str(result.converged)
        row["log_likelihood"] = f"{result.log_likelihood:.2f}" if result.log_likelihood else ""
        row["status"] = "OK"

        if result.parameters is not None:
            params = result.parameters
            for i, fname in enumerate(feature_names):
                if i < len(params):
                    row[fname] = f"{params[i].item():.6f}"

        print(f"\n  [OK] {name}: {elapsed:.1f}s, converged={result.converged}")
        if result.log_likelihood:
            print(f"  LL: {result.log_likelihood:.2f}")
        if result.parameters is not None:
            for i, fname in enumerate(feature_names):
                if i < len(params):
                    print(f"  {fname}: {params[i].item():.4f}")

    except Exception as e:
        elapsed = time.time() - t0
        row["time_seconds"] = f"{elapsed:.1f}"
        row["status"] = f"FAIL: {e}"
        print(f"\n  [FAIL] {name}: {elapsed:.1f}s — {e}")
        traceback.print_exc()

    append_result(row)
    return row


def main():
    problem, transitions, utility, panel, feature_names = load_ngsim_problem()

    completed = get_completed()
    print(f"\nCompleted: {completed or 'none'}")

    # ── 1. NFXP-NK ──
    if "NFXP-NK" not in completed:
        from econirl.estimation.nfxp import NFXPEstimator
        est = NFXPEstimator()
        run_estimator("NFXP-NK", est, panel, utility, problem, transitions, feature_names)
    else:
        print("\n  SKIP: NFXP-NK (already done)")

    # ── 2. MaxEnt IRL (Ziebart 2008) ──
    if "MaxEnt IRL" not in completed:
        from econirl.estimation.maxent_irl import MaxEntIRLEstimator
        from econirl.preferences.action_reward import ActionDependentReward
        reward_fn = ActionDependentReward(
            feature_matrix=utility.feature_matrix,
            parameter_names=utility.parameter_names,
        )
        est = MaxEntIRLEstimator(
            optimizer="L-BFGS-B", inner_solver="value",
            inner_tol=1e-8, inner_max_iter=1000,
            outer_tol=1e-6, outer_max_iter=500, verbose=True,
        )
        run_estimator("MaxEnt IRL", est, panel, reward_fn, problem, transitions, feature_names)
    else:
        print("\n  SKIP: MaxEnt IRL (already done)")

    # ── 3. MCE IRL (Ziebart 2010) ──
    if "MCE IRL" not in completed:
        from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
        config = MCEIRLConfig(
            compute_se=False,
            optimizer="L-BFGS-B",       # Use scipy L-BFGS-B for reliable convergence
            outer_max_iter=500,
            inner_max_iter=1000,
            outer_tol=1e-6,
            verbose=True,
        )
        est = MCEIRLEstimator(config=config)
        run_estimator("MCE IRL", est, panel, utility, problem, transitions, feature_names)
    else:
        print("\n  SKIP: MCE IRL (already done)")

    # ── 4. AIRL (Fu et al. 2018) ──
    if "AIRL" not in completed:
        try:
            from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
            from econirl.preferences.action_reward import ActionDependentReward
            # AIRL needs ActionDependentReward, wrap LinearUtility
            airl_reward = ActionDependentReward(
                feature_matrix=utility.feature_matrix,
                parameter_names=utility.parameter_names,
            )
            config = AIRLConfig(
                reward_type="linear", reward_lr=0.01,
                max_rounds=200, generator_max_iter=500, verbose=True,
            )
            est = AIRLEstimator(config=config)
            run_estimator("AIRL", est, panel, airl_reward, problem, transitions, feature_names)
        except Exception as e:
            print(f"\n  [FAIL] AIRL setup: {e}")
            append_result({"estimator": "AIRL", "status": f"FAIL: {e}",
                          **{c: "" for c in CSV_COLUMNS if c not in ["estimator", "status"]}})
    else:
        print("\n  SKIP: AIRL (already done)")

    print(f"\n{'='*60}")
    print(f"Results saved to: {CSV_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
