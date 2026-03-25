"""
NGSIM US-101 MCE IRL — Replicating Huang, Wu & Lv (2021)
=========================================================

Applies Maximum Causal Entropy IRL to the NGSIM US-101 highway dataset
to recover drivers' lane-change reward functions, following the feature
specification of:

    Huang, Wu & Lv (2021). "Driving Behavior Modeling using Naturalistic
    Human Driving Data with Inverse Reinforcement Learning."
    IEEE Trans. Intelligent Transportation Systems.

Their 8 reward features:
    1. Speed (travel efficiency)
    2. Longitudinal acceleration magnitude (comfort)
    3. Lateral acceleration proxy (comfort)
    4. Longitudinal jerk magnitude (comfort)
    5. Front headway risk: exp(-time_headway_front)
    6. Rear headway risk: exp(-time_headway_rear)
    7. Collision indicator
    8. Lane change indicator (proxy for interaction/social cost)

We adapt these to a discrete DDC framework:
    - State: (lane, speed_bin) → 50 states
    - Action: lane_left / stay / lane_right → 3 actions
    - Features computed as empirical averages per (state, action) cell

Expected results (from literature):
    - Risk aversion (front + rear headway) dominates for most drivers
    - Speed preference is positive but heterogeneous
    - Comfort features (accel, jerk) have negative weights
    - Lane changes are costly
"""

import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from econirl.datasets.ngsim import N_LANES, N_SPEED_BINS, N_ACTIONS
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.linear import LinearUtility


# Feature names matching Huang et al. (2021)
FEATURE_NAMES = [
    "speed",              # f1: travel efficiency (v)
    "accel_magnitude",    # f2: |longitudinal acceleration| (comfort)
    "lateral_movement",   # f3: lateral acceleration proxy (comfort)
    "jerk_magnitude",     # f4: |longitudinal jerk| (comfort)
    "headway_risk_front", # f5: exp(-THW_front) (risk aversion)
    "headway_risk_rear",  # f6: exp(-THW_rear) (risk aversion)
    "collision_risk",     # f7: near-collision indicator
    "lane_change_cost",   # f8: lane change indicator (interaction proxy)
]
N_FEATURES = len(FEATURE_NAMES)


def load_and_compute_features(max_vehicles=500, subsample=10):
    """Load NGSIM data and compute per-observation Huang et al. features."""
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "ngsim" / "us101_trajectories.csv"

    usecols = ["vehicle_id", "frame_id", "v_vel", "v_acc", "lane_id",
               "space_headway", "time_headway", "local_x"]
    df = pd.read_csv(data_path, usecols=usecols)

    # Clean quoted values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.strip('"'), errors="coerce")

    # Filter mainline lanes
    df = df[df["lane_id"].between(1, N_LANES)].copy()
    df = df.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)

    # Subsample (10Hz → 1Hz)
    if subsample > 1:
        df = df.groupby("vehicle_id", group_keys=False).apply(
            lambda x: x.iloc[::subsample]
        ).reset_index(drop=True)

    # Discretize
    df["lane"] = (df["lane_id"] - 1).astype(int)
    df["speed_bin"] = np.clip((df["v_vel"] / 5.0).astype(int), 0, N_SPEED_BINS - 1)
    df["state"] = df["lane"] * N_SPEED_BINS + df["speed_bin"]

    # Compute lane changes and next state
    df["next_lane"] = df.groupby("vehicle_id")["lane"].shift(-1)
    df["next_speed_bin"] = df.groupby("vehicle_id")["speed_bin"].shift(-1)
    df["lane_change"] = df["next_lane"] - df["lane"]
    df["action"] = 1  # stay
    df.loc[df["lane_change"] == -1, "action"] = 0  # left
    df.loc[df["lane_change"] == 1, "action"] = 2   # right
    df = df[df["lane_change"].abs() <= 1].copy()
    df["next_state"] = df["next_lane"] * N_SPEED_BINS + df["next_speed_bin"]
    df = df.dropna(subset=["next_lane", "next_speed_bin"]).copy()
    df["next_state"] = df["next_state"].astype(int)

    # Compute jerk (derivative of acceleration between consecutive frames)
    df["v_acc_next"] = df.groupby("vehicle_id")["v_acc"].shift(-1)
    df["jerk"] = (df["v_acc_next"] - df["v_acc"]).abs()  # Approximate jerk

    # ── Compute 8 Huang et al. features per observation ──

    # f1: Speed (travel efficiency) — higher is better
    df["f_speed"] = df["v_vel"].clip(lower=0)

    # f2: |Longitudinal acceleration| — lower is more comfortable
    df["f_accel"] = df["v_acc"].abs()

    # f3: Lateral movement proxy — |lane_change| as binary indicator
    #     (we don't have continuous lateral accel, so use lane change as proxy)
    df["f_lateral"] = df["lane_change"].abs().astype(float)

    # f4: |Longitudinal jerk| — lower is smoother
    df["f_jerk"] = df["jerk"].fillna(0)

    # f5: Front headway risk — exp(-time_headway)
    #     Higher value = closer to front car = more dangerous
    df["time_headway_clean"] = df["time_headway"].clip(lower=0.1, upper=20.0)
    df["f_headway_front"] = np.exp(-df["time_headway_clean"])

    # f6: Rear headway risk proxy — based on space headway
    #     (NGSIM doesn't directly give rear headway, use space_headway as proxy)
    df["space_headway_clean"] = df["space_headway"].clip(lower=1.0, upper=500.0)
    df["f_headway_rear"] = np.exp(-df["space_headway_clean"] / 50.0)

    # f7: Collision risk — very close headway indicator
    df["f_collision"] = ((df["time_headway_clean"] < 0.5) |
                         (df["space_headway_clean"] < 10.0)).astype(float)

    # f8: Lane change cost — binary indicator of lane change
    df["f_lane_change"] = (df["lane_change"] != 0).astype(float)

    # Filter vehicles with enough data
    vehicle_counts = df["vehicle_id"].value_counts()
    valid = vehicle_counts[vehicle_counts >= 50].index
    df = df[df["vehicle_id"].isin(valid)].copy()
    if max_vehicles:
        selected = df["vehicle_id"].unique()[:max_vehicles]
        df = df[df["vehicle_id"].isin(selected)].copy()

    df["period"] = df.groupby("vehicle_id").cumcount()
    return df


def build_feature_matrix(df):
    """Build (n_states, n_actions, n_features) STRUCTURAL feature matrix.

    Features are defined from the state/action structure, NOT empirical averages.
    State-level features (speed, headway) use empirical means per state.
    Action-level features (lane change) are structural binary indicators.

    This matches the DDC/IRL convention: φ(s,a) is a known function of (s,a),
    and θ are the unknown parameters to be estimated.
    """
    n_states = N_LANES * N_SPEED_BINS
    n_actions = N_ACTIONS

    # Compute empirical state-level statistics from data
    state_speed = torch.zeros(n_states)
    state_headway_front = torch.zeros(n_states)
    state_headway_rear = torch.zeros(n_states)
    state_accel = torch.zeros(n_states)
    state_collision = torch.zeros(n_states)
    state_counts = torch.zeros(n_states)

    for _, row in df.iterrows():
        s = int(row.state)
        if 0 <= s < n_states:
            state_speed[s] += row.f_speed
            state_headway_front[s] += row.f_headway_front
            state_headway_rear[s] += row.f_headway_rear
            state_accel[s] += row.f_accel
            state_collision[s] += row.f_collision
            state_counts[s] += 1

    state_counts_safe = state_counts.clamp(min=1)
    state_speed /= state_counts_safe
    state_headway_front /= state_counts_safe
    state_headway_rear /= state_counts_safe
    state_accel /= state_counts_safe
    state_collision /= state_counts_safe

    # Build structural features: (n_states, n_actions, 5 features)
    # Reduced to 5 well-identified features matching Huang et al. categories
    n_features_used = 5
    features = torch.zeros(n_states, n_actions, n_features_used)

    for s in range(n_states):
        lane = s // N_SPEED_BINS
        speed_bin = s % N_SPEED_BINS

        for a in range(n_actions):
            # f1: Speed (travel efficiency) — structural from state definition
            features[s, a, 0] = speed_bin / (N_SPEED_BINS - 1)

            # f2: Acceleration magnitude (comfort) — empirical per state
            features[s, a, 1] = -state_accel[s]  # negative = high accel is bad

            # f3: Front headway risk — empirical per state
            features[s, a, 2] = -state_headway_front[s]  # negative = close following is bad

            # f4: Collision risk — empirical per state
            features[s, a, 3] = -state_collision[s]  # negative = collision is bad

            # f5: Lane change cost — STRUCTURAL, action-dependent
            if a == 0 or a == 2:  # left or right
                features[s, a, 4] = -1.0  # lane change penalty
            else:
                features[s, a, 4] = 0.0   # stay = no penalty

    # Normalize each feature to [-1, 1]
    for k in range(n_features_used):
        fk = features[:, :, k]
        fmax = fk.abs().max()
        if fmax > 0:
            features[:, :, k] = fk / fmax

    return features, ["speed", "accel_cost", "headway_risk", "collision_risk", "lane_change_cost"]


def build_problem_from_data(df):
    """Build DDCProblem, transitions, features, panel from processed data."""
    n_states = N_LANES * N_SPEED_BINS
    n_actions = N_ACTIONS

    print("\n── Building DDC Problem ──")

    # Problem spec
    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.99,
        scale_parameter=1.0,
    )

    # Estimate transitions
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

    # Build structural feature matrix
    print("Computing structural features (Huang et al. categories)...")
    features, feature_names = build_feature_matrix(df)

    utility = LinearUtility(
        feature_matrix=features,
        parameter_names=feature_names,
    )

    # Build panel
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

    print(f"  States: {n_states}, Actions: {n_actions}, Features: {N_FEATURES}")
    print(f"  Transitions: {transitions.shape}, non-zero: {(transitions > 0).sum().item()}")
    print(f"  Panel: {panel.num_individuals} vehicles, {panel.num_observations} obs")

    return problem, transitions, utility, panel


def run_mce_irl(panel, utility, problem, transitions):
    """Run MCE IRL with settings tuned for NGSIM."""
    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig

    print("\n── Running MCE IRL (Ziebart 2008 / Huang et al. 2021 features) ──")

    config = MCEIRLConfig(
        compute_se=False,
        optimizer="L-BFGS-B",       # More stable than Adam for this problem
        inner_max_iter=1000,         # Ensure inner convergence
        outer_max_iter=500,          # More outer iterations
        learning_rate=0.01,          # Lower LR for stability
        outer_tol=1e-6,
        verbose=True,
    )
    estimator = MCEIRLEstimator(config=config)

    t0 = time.time()
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=problem,
        transitions=transitions,
    )
    elapsed = time.time() - t0

    print(f"\n── MCE IRL Results ({elapsed:.1f}s) ──")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.num_iterations}")
    if result.log_likelihood is not None:
        print(f"Log-likelihood: {result.log_likelihood:.2f}")

    # Report parameters
    print(f"\nLearned Reward Weights (θ):")
    print(f"{'Feature':<25s} {'Weight':>10s} {'SE':>10s} {'Interpretation'}")
    print("─" * 75)

    interpretations = {
        "speed": "Higher speed preferred → positive",
        "accel_magnitude": "Acceleration costly → negative",
        "lateral_movement": "Lane changes penalized → negative",
        "jerk_magnitude": "Jerk costly → negative",
        "headway_risk_front": "Close following risky → negative",
        "headway_risk_rear": "Being tailgated risky → negative",
        "collision_risk": "Collisions penalized → negative",
        "lane_change_cost": "Lane changes costly → negative",
    }

    if result.parameters is not None:
        params = result.parameters
        ses = result.standard_errors if hasattr(result, 'standard_errors') and result.standard_errors is not None else [None] * len(params)
        for i, name in enumerate(utility.parameter_names):
            if i < len(params):
                w = params[i].item()
                se_str = f"{ses[i].item():.4f}" if ses[i] is not None else "N/A"
                interp = interpretations.get(name, "")
                print(f"  {name:<23s} {w:>10.4f} {se_str:>10s}  {interp}")

    # Compare with Huang et al. expectations
    print(f"\n── Comparison with Huang et al. (2021) ──")
    print("""
Huang et al. found (qualitative, across 100 drivers):
  • Risk aversion (headway front + rear) is the most critical factor
    shared by most drivers
  • Speed preference varies: some drivers prioritize efficiency,
    others prioritize comfort
  • Collision penalty fixed at θ = -10 (they hardcoded this)
  • Interaction (social cost) varies significantly across drivers

Our discrete MCE IRL results should show:
  ✓ Negative headway_risk weights (risk aversion)
  ✓ Positive speed weight (travel efficiency)
  ✓ Negative comfort weights (accel, jerk penalized)
  ✓ Negative lane_change_cost (changes are costly)
""")

    return result


def run_all_three(panel, utility, problem, transitions):
    """Run MaxEnt IRL (2008), MCE IRL (2010), and NFXP-NK."""
    results = {}

    # 1. MaxEnt IRL (Ziebart 2008) — uses L-BFGS-B, should converge well
    try:
        from econirl.estimation.maxent_irl import MaxEntIRLEstimator
        from econirl.preferences.action_reward import ActionDependentReward
        print("\n── MaxEnt IRL (Ziebart 2008) — L-BFGS-B ──")

        # MaxEntIRL needs ActionDependentReward, wrap our LinearUtility features
        reward_fn = ActionDependentReward(
            feature_matrix=utility.feature_matrix,
            parameter_names=utility.parameter_names,
        )

        t0 = time.time()
        est = MaxEntIRLEstimator(
            optimizer="L-BFGS-B",
            inner_solver="value",
            inner_tol=1e-8,
            inner_max_iter=1000,
            outer_tol=1e-6,
            outer_max_iter=500,
            verbose=True,
        )
        r = est.estimate(panel=panel, utility=reward_fn, problem=problem, transitions=transitions)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s, Converged: {r.converged}")
        if r.parameters is not None:
            for i, name in enumerate(utility.parameter_names):
                if i < len(r.parameters):
                    print(f"  {name}: {r.parameters[i].item():.4f}")
        if r.log_likelihood is not None:
            print(f"  LL: {r.log_likelihood:.2f}")
        results["MaxEnt IRL (2008)"] = (r, elapsed)
    except Exception as e:
        print(f"  MaxEnt IRL failed: {e}")
        import traceback; traceback.print_exc()

    # 2. MCE IRL (Ziebart 2010) — Adam optimizer
    try:
        from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
        print("\n── MCE IRL (Ziebart 2010) — Adam ──")
        config = MCEIRLConfig(
            compute_se=False,
            use_adam=True,
            learning_rate=0.001,        # Very small for stability
            outer_max_iter=2000,         # Many iterations
            inner_max_iter=1000,
            outer_tol=1e-6,
            verbose=True,
        )
        t0 = time.time()
        est = MCEIRLEstimator(config=config)
        r = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s, Converged: {r.converged}")
        if r.parameters is not None:
            for i, name in enumerate(utility.parameter_names):
                if i < len(r.parameters):
                    print(f"  {name}: {r.parameters[i].item():.4f}")
        if r.log_likelihood is not None:
            print(f"  LL: {r.log_likelihood:.2f}")
        results["MCE IRL (2010)"] = (r, elapsed)
    except Exception as e:
        print(f"  MCE IRL failed: {e}")
        import traceback; traceback.print_exc()

    # 3. NFXP-NK (Rust 1987 / Iskhakov et al. 2016) — L-BFGS-B, gold standard
    try:
        from econirl.estimation.nfxp import NFXPEstimator
        print("\n── NFXP-NK (Rust 1987) — L-BFGS-B ──")
        t0 = time.time()
        est = NFXPEstimator()
        r = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s, Converged: {r.converged}")
        if r.parameters is not None:
            for i, name in enumerate(utility.parameter_names):
                if i < len(r.parameters):
                    print(f"  {name}: {r.parameters[i].item():.4f}")
        if r.log_likelihood is not None:
            print(f"  LL: {r.log_likelihood:.2f}")
        results["NFXP-NK"] = (r, elapsed)
    except Exception as e:
        print(f"  NFXP failed: {e}")
        import traceback; traceback.print_exc()

    # 4. AIRL (Fu et al. 2018) — adversarial reward recovery
    try:
        from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
        print("\n── AIRL (Fu et al. 2018) — Adversarial ──")
        airl_config = AIRLConfig(
            reward_type="linear",
            reward_lr=0.01,
            max_rounds=200,
            generator_max_iter=500,
            verbose=True,
        )
        t0 = time.time()
        est = AIRLEstimator(config=airl_config)
        r = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s, Converged: {r.converged}")
        if r.parameters is not None:
            for i, name in enumerate(utility.parameter_names):
                if i < len(r.parameters):
                    print(f"  {name}: {r.parameters[i].item():.4f}")
        if r.log_likelihood is not None:
            print(f"  LL: {r.log_likelihood:.2f}")
        results["AIRL (2018)"] = (r, elapsed)
    except Exception as e:
        print(f"  AIRL failed: {e}")
        import traceback; traceback.print_exc()

    return results


if __name__ == "__main__":
    # Load and compute features
    print("=" * 60)
    print("NGSIM MCE IRL — Matching Huang, Wu & Lv (2021)")
    print("=" * 60)

    df = load_and_compute_features(max_vehicles=500)
    print(f"\nLoaded: {len(df)} observations, {df.vehicle_id.nunique()} vehicles")
    print(f"\nFeature summary (raw, before normalization):")
    for col in ["f_speed", "f_accel", "f_jerk", "f_headway_front",
                "f_headway_rear", "f_collision", "f_lane_change"]:
        print(f"  {col:25s}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    # Build problem
    problem, transitions, utility, panel = build_problem_from_data(df)

    # Run all three: MaxEnt IRL (2008), MCE IRL (2010), NFXP-NK
    results = run_all_three(panel, utility, problem, transitions)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
