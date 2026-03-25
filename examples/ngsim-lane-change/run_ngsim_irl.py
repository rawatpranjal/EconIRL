"""
NGSIM US-101 Lane-Change IRL Benchmark
=======================================

This example applies IRL estimators to real highway driving data from the
NGSIM US-101 dataset to recover drivers' lane-change preferences.

The setup follows the classic IRL driving benchmark:
- State: (lane, speed_bin) — 5 lanes × 10 speed bins = 50 states
- Actions: lane_left / stay / lane_right — 3 actions
- Features: speed, headway, lane position → learn what drivers value

Algorithms tested (in order of relevance to driving IRL):
1. MCE IRL (Ziebart 2008) — THE canonical driving IRL method
2. CCP (Hotz-Miller 1993) — fast econometric two-step
3. NFXP (Rust 1987) — full MLE, gold standard
4. MaxMargin (Ratliff 2006) — margin-based, first IRL driving method
5. Deep MaxEnt — neural extension for richer reward
6. BC (behavioral cloning) — imitation baseline

Reference:
    Ziebart et al. (2008) applied MaxEnt IRL to Pittsburgh taxi drivers.
    This example applies the same family of methods to highway lane-change
    decisions from the NGSIM US-101 freeway in Los Angeles.

Requirements:
    Download NGSIM data first: data/raw/ngsim/us101_trajectories.csv
    (~814MB from https://datahub.transportation.gov/resource/8ect-6jqj)
"""

import time
import numpy as np
import torch
import pandas as pd

from econirl.datasets.ngsim import load_ngsim, get_ngsim_info, N_LANES, N_SPEED_BINS, N_ACTIONS
from econirl.core.types import DDCProblem, Panel
from econirl.preferences.linear import LinearUtility


# ============================================================================
# Step 1: Load and inspect NGSIM data
# ============================================================================

def load_data(max_vehicles=500):
    """Load NGSIM data and display summary statistics."""
    print("=" * 60)
    print("NGSIM US-101 Lane-Change IRL Benchmark")
    print("=" * 60)

    info = get_ngsim_info()
    print(f"\nDataset: {info['name']}")
    print(f"States: {info['n_states']} ({info['state_description']})")
    print(f"Actions: {info['n_actions']} ({info['action_description']})")

    print(f"\nLoading data (max {max_vehicles} vehicles)...")
    t0 = time.time()
    df = load_ngsim(max_vehicles=max_vehicles)
    print(f"Loaded in {time.time() - t0:.1f}s: {len(df)} observations, {df.vehicle_id.nunique()} vehicles")

    # Summary stats
    print(f"\nAction distribution:")
    action_names = ["Lane Left", "Stay", "Lane Right"]
    for a in range(N_ACTIONS):
        count = (df.action == a).sum()
        pct = count / len(df) * 100
        print(f"  {action_names[a]:12s}: {count:>8,d} ({pct:.1f}%)")

    print(f"\nLane distribution:")
    lane_names = ["Lane 1 (left)", "Lane 2", "Lane 3 (center)", "Lane 4", "Lane 5 (right)"]
    for lane in range(N_LANES):
        count = (df.lane == lane).sum()
        pct = count / len(df) * 100
        print(f"  {lane_names[lane]:18s}: {count:>8,d} ({pct:.1f}%)")

    print(f"\nSpeed distribution (bins):")
    for sb in range(N_SPEED_BINS):
        count = (df.speed_bin == sb).sum()
        if count > 0:
            speed_lo = sb * 5
            speed_hi = (sb + 1) * 5
            pct = count / len(df) * 100
            print(f"  {speed_lo:2d}-{speed_hi:2d} ft/s ({speed_lo*0.68:.0f}-{speed_hi*0.68:.0f} mph): {count:>8,d} ({pct:.1f}%)")

    return df


# ============================================================================
# Step 2: Build DDC problem specification
# ============================================================================

def build_problem(df):
    """Build DDCProblem, transitions, and features from NGSIM data."""
    n_states = N_LANES * N_SPEED_BINS  # 50
    n_actions = N_ACTIONS  # 3

    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.99,
        scale_parameter=1.0,
    )

    # Estimate transition probabilities from data
    print("\nEstimating transition probabilities...")
    transitions = torch.zeros(n_actions, n_states, n_states)
    for _, row in df.iterrows():
        s, a, ns = int(row.state), int(row.action), int(row.next_state)
        if 0 <= s < n_states and 0 <= ns < n_states and 0 <= a < n_actions:
            transitions[a, s, ns] += 1

    # Normalize rows (add small smoothing to avoid zeros)
    for a in range(n_actions):
        row_sums = transitions[a].sum(dim=1, keepdim=True)
        row_sums = row_sums.clamp(min=1)
        transitions[a] = transitions[a] / row_sums

    # Handle states with no observations: uniform transition
    for a in range(n_actions):
        zero_rows = transitions[a].sum(dim=1) < 0.5
        if zero_rows.any():
            transitions[a, zero_rows] = 1.0 / n_states

    print(f"  Transitions shape: {transitions.shape}")
    print(f"  Non-zero entries: {(transitions > 0).sum().item()}")

    # Build feature matrix: (n_states, n_actions, n_features)
    # Features: [lane_center_distance, speed_normalized, headway_proxy]
    n_features = 3
    features = torch.zeros(n_states, n_actions, n_features)

    for s in range(n_states):
        lane = s // N_SPEED_BINS
        speed_bin = s % N_SPEED_BINS

        # Feature 1: Distance from center lane (prefer middle lanes)
        center_lane = (N_LANES - 1) / 2  # 2.0
        lane_dist = abs(lane - center_lane) / center_lane  # 0 at center, 1 at edge
        features[s, :, 0] = -lane_dist  # negative = penalty for being away from center

        # Feature 2: Speed (prefer moderate speed)
        speed_norm = speed_bin / (N_SPEED_BINS - 1)  # 0 to 1
        features[s, :, 1] = speed_norm

        # Feature 3: Lane change cost (per action)
        for a in range(n_actions):
            if a == 0:  # left
                features[s, a, 2] = -0.5  # cost of changing lanes
            elif a == 2:  # right
                features[s, a, 2] = -0.5
            else:  # stay
                features[s, a, 2] = 0.0

    utility = LinearUtility(
        feature_matrix=features,
        parameter_names=["lane_center_pref", "speed_pref", "lane_change_cost"],
    )

    print(f"  Features shape: {features.shape}")
    print(f"  Feature names: {utility.parameter_names}")

    # Convert to Panel
    panel = load_ngsim(as_panel=True, max_vehicles=df.vehicle_id.nunique())

    return problem, transitions, utility, panel


# ============================================================================
# Step 3: Run estimators
# ============================================================================

def run_estimator(name, estimator, panel, utility, problem, transitions):
    """Run a single estimator and report results."""
    print(f"\n{'─' * 50}")
    print(f"Running {name}...")
    t0 = time.time()
    try:
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Converged: {result.converged}")

        if hasattr(result, 'parameters') and result.parameters is not None:
            params = result.parameters
            print(f"  Parameters:")
            for i, name_p in enumerate(utility.parameter_names):
                if i < len(params):
                    print(f"    {name_p}: {params[i].item():.4f}")

        if hasattr(result, 'log_likelihood') and result.log_likelihood is not None:
            print(f"  Log-likelihood: {result.log_likelihood:.2f}")

        return result, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        return None, elapsed


def run_all_estimators(panel, utility, problem, transitions):
    """Run all applicable estimators on NGSIM data."""
    results = {}

    # 1. MCE IRL — the canonical driving IRL method
    try:
        from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=200,
            outer_max_iter=50,
            learning_rate=0.1,
            verbose=False,
        )
        est = MCEIRLEstimator(config=config)
        results["MCE IRL"] = run_estimator("MCE IRL (Ziebart 2008)", est, panel, utility, problem, transitions)
    except Exception as e:
        print(f"MCE IRL setup failed: {e}")

    # 2. CCP — fast two-step econometric estimator
    try:
        from econirl.estimation.ccp import CCPEstimator
        est = CCPEstimator()
        results["CCP"] = run_estimator("CCP (Hotz-Miller 1993)", est, panel, utility, problem, transitions)
    except Exception as e:
        print(f"CCP setup failed: {e}")

    # 3. NFXP — full MLE
    try:
        from econirl.estimation.nfxp import NFXPEstimator
        est = NFXPEstimator()
        results["NFXP"] = run_estimator("NFXP (Rust 1987)", est, panel, utility, problem, transitions)
    except Exception as e:
        print(f"NFXP setup failed: {e}")

    # 4. Max Margin Planning
    try:
        from econirl.estimation.max_margin_planning import MaxMarginPlanningEstimator
        est = MaxMarginPlanningEstimator()
        results["MaxMargin"] = run_estimator("Max Margin (Ratliff 2006)", est, panel, utility, problem, transitions)
    except Exception as e:
        print(f"MaxMargin setup failed: {e}")

    # 5. Deep MaxEnt IRL
    try:
        from econirl.estimation.deep_maxent_irl import DeepMaxEntIRLEstimator
        est = DeepMaxEntIRLEstimator()
        results["DeepMaxEnt"] = run_estimator("Deep MaxEnt IRL", est, panel, utility, problem, transitions)
    except Exception as e:
        print(f"DeepMaxEnt setup failed: {e}")

    return results


# ============================================================================
# Step 4: Interpret results
# ============================================================================

def interpret_results(results):
    """Print interpretation of recovered lane-change preferences."""
    print("\n" + "=" * 60)
    print("RESULTS INTERPRETATION")
    print("=" * 60)

    print("""
Expected preferences for highway drivers:
  - lane_center_pref > 0: Drivers prefer center lanes (safer, more options)
  - speed_pref > 0: Higher speed is preferred (reach destination faster)
  - lane_change_cost < 0: Lane changes are costly (effort, risk)

These are the "structural parameters" that rationalize observed behavior
under the assumption of entropy-regularized optimality (logit DDC / MaxEnt IRL).
""")

    # Summary table
    print(f"{'Estimator':<20s} {'Time':>6s} {'Conv':>5s} {'lane_center':>12s} {'speed':>8s} {'lc_cost':>8s}")
    print("─" * 65)
    for name, (result, elapsed) in results.items():
        if result is not None and hasattr(result, 'parameters') and result.parameters is not None:
            p = result.parameters
            conv = "Yes" if result.converged else "No"
            params_str = [f"{p[i].item():>8.4f}" if i < len(p) else f"{'N/A':>8s}" for i in range(3)]
            print(f"{name:<20s} {elapsed:>5.1f}s {conv:>5s} {params_str[0]:>12s} {params_str[1]:>8s} {params_str[2]:>8s}")
        else:
            print(f"{name:<20s} {elapsed:>5.1f}s {'FAIL':>5s}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Load data (use 500 vehicles for reasonable runtime)
    df = load_data(max_vehicles=500)

    # Build problem specification
    problem, transitions, utility, panel = build_problem(df)

    # Run all estimators
    results = run_all_estimators(panel, utility, problem, transitions)

    # Interpret
    interpret_results(results)
