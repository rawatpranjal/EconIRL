#!/usr/bin/env python3
"""
AHS Housing Mobility -- Semi-Synthetic Parameter Recovery
=========================================================

Estimates structural parameters of a household mobility model using
transition dynamics calibrated from the 2023 American Housing Survey.
Known ground truth reward parameters are used to generate data from
the soft VI optimal policy. Four estimators (NFXP, CCP, NNES, TD-CCP)
recover the parameters, demonstrating both structural and neural
estimation on a realistic housing domain.

State: (tenure_type, age_bin, income_bin, duration_bin) -> 2 x 3 x 3 x 3 = 54
Action: stay=0, move=1

Features (5):
    0. housing_burden  -- median rent/income for renters (cost-to-income)
    1. duration        -- years in current unit (attachment/lock-in)
    2. renter          -- 1 if renting (renters are more mobile)
    3. age             -- householder age normalized (older = less mobile)
    4. move_cost       -- 1 for move action (transaction cost of moving)

Data: AHS 2023 National, household.csv (~55K households)
    Cross-section provides: transition calibration, initial state
    distribution, and burden values per state.

Reference:
    Ferreira, F., Gyourko, J., Tracy, J. (2010). "Housing Busts and
    Household Mobility." Journal of Urban Economics 68(1): 34-45.

Usage:
    python examples/ahs-housing/run_ccp.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.nnes import NNESEstimator
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel_from_policy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/raw/ahs")
SURVEY_YEAR = 2023
N_INDIVIDUALS = 5_000
N_PERIODS = 20
DISCOUNT_FACTOR = 0.95

# State dimensions
TENURE_TYPES = 2     # 0=own, 1=rent
AGE_BINS = 3         # young(<35), middle(35-64), senior(65+)
INCOME_BINS = 3      # low(<40K), middle(40-100K), high(100K+)
DURATION_BINS = 3    # new(0-2yr), established(3-9yr), long(10+yr)

NUM_STATES = TENURE_TYPES * AGE_BINS * INCOME_BINS * DURATION_BINS  # 54
NUM_ACTIONS = 2   # stay=0, move=1
NUM_FEATURES = 5

# Ground truth reward parameters
TRUE_PARAMS = {
    "theta_burden": -0.50,    # high burden reduces stay utility
    "theta_duration": 0.30,   # longer tenure increases attachment
    "theta_renter": -0.40,    # renters less attached to current unit
    "theta_age": 0.60,        # older households prefer staying
    "move_cost": -1.50,       # transaction cost of moving
}
TRUE_PARAM_ARRAY = jnp.array(list(TRUE_PARAMS.values()))
PARAM_NAMES = list(TRUE_PARAMS.keys())


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(tenure: int, age_bin: int, inc_bin: int, dur_bin: int) -> int:
    return (tenure * AGE_BINS * INCOME_BINS * DURATION_BINS
            + age_bin * INCOME_BINS * DURATION_BINS
            + inc_bin * DURATION_BINS
            + dur_bin)


def decode_state(s: int) -> tuple[int, int, int, int]:
    dur_bin = s % DURATION_BINS
    s //= DURATION_BINS
    inc_bin = s % INCOME_BINS
    s //= INCOME_BINS
    age_bin = s % AGE_BINS
    tenure = s // AGE_BINS
    return tenure, age_bin, inc_bin, dur_bin


# ---------------------------------------------------------------------------
# Load cross-section for calibration
# ---------------------------------------------------------------------------

def load_ahs_moments() -> dict:
    """Load AHS cross-section and compute calibration moments.

    Returns burden values per state and initial state distribution.
    """
    print("Loading AHS household.csv...")
    df = (
        pl.scan_csv(str(DATA_DIR / "household.csv"))
        .select(["TENURE", "HHMOVE", "RENT", "HINCP", "HHAGE", "NUMPEOPLE"])
        .collect()
    )
    print(f"  Loaded {df.shape[0]:,} households")

    # Filter to owner/renter, drop vacants
    df = df.filter(pl.col("TENURE").is_in(["'1'", "'2'"]))
    df = df.with_columns([
        (pl.col("TENURE") == "'2'").cast(pl.Int32).alias("renter"),
        pl.col("HINCP").cast(pl.Float64),
        pl.col("RENT").cast(pl.Float64),
        pl.col("HHAGE").cast(pl.Float64),
        pl.col("HHMOVE").cast(pl.Int64),
    ])

    df = df.filter(
        pl.col("HINCP") > 0,
        pl.col("HHAGE") > 0,
        pl.col("HHMOVE") > 0,
    )

    # Duration
    df = df.with_columns(
        (SURVEY_YEAR - pl.col("HHMOVE")).clip(0, 50).alias("duration")
    )

    # Housing burden: rent / annual income for renters
    df = df.with_columns(
        pl.when(
            (pl.col("renter") == 1) & (pl.col("RENT") > 0) & (pl.col("HINCP") > 0)
        )
        .then(pl.col("RENT") * 12 / pl.col("HINCP"))
        .otherwise(0.0)
        .clip(0.0, 2.0)
        .alias("burden")
    )

    # Bin state variables
    df = df.with_columns([
        pl.when(pl.col("HHAGE") < 35).then(0)
          .when(pl.col("HHAGE") < 65).then(1)
          .otherwise(2).cast(pl.Int32).alias("age_bin"),
        pl.when(pl.col("HINCP") < 40_000).then(0)
          .when(pl.col("HINCP") < 100_000).then(1)
          .otherwise(2).cast(pl.Int32).alias("inc_bin"),
        pl.when(pl.col("duration") <= 2).then(0)
          .when(pl.col("duration") <= 9).then(1)
          .otherwise(2).cast(pl.Int32).alias("dur_bin"),
    ])

    # Flat state index
    df = df.with_columns(
        (pl.col("renter") * AGE_BINS * INCOME_BINS * DURATION_BINS
         + pl.col("age_bin") * INCOME_BINS * DURATION_BINS
         + pl.col("inc_bin") * DURATION_BINS
         + pl.col("dur_bin")).cast(pl.Int32).alias("state")
    )

    n = len(df)
    print(f"  Valid households: {n:,}")
    print(f"  Renter share: {df['renter'].mean():.3f}")

    # Per-state burden values
    burden_by_state = np.zeros(NUM_STATES)
    for s in range(NUM_STATES):
        tenure, ab, ib, db = decode_state(s)
        sub = df.filter(
            (pl.col("renter") == tenure) &
            (pl.col("age_bin") == ab) &
            (pl.col("inc_bin") == ib)
        )
        if len(sub) > 0 and tenure == 1:
            burden_by_state[s] = float(sub["burden"].median())

    # Initial state distribution from cross-section
    state_counts = np.bincount(
        df["state"].to_numpy().astype(np.int32), minlength=NUM_STATES
    ).astype(np.float64)
    initial_dist = state_counts / state_counts.sum()

    # Income distribution for transition calibration
    inc_counts = np.array(
        df["inc_bin"].value_counts().sort("inc_bin")["count"].to_list(),
        dtype=np.float64
    )
    inc_probs = inc_counts / inc_counts.sum()

    return {
        "burden_by_state": burden_by_state,
        "initial_dist": initial_dist,
        "inc_probs": inc_probs,
        "renter_share": float(df["renter"].mean()),
        "n_valid": n,
    }


# ---------------------------------------------------------------------------
# Build structural transition matrices
# ---------------------------------------------------------------------------

def build_structural_transitions(inc_probs: np.ndarray) -> np.ndarray:
    """Build transition matrices from structural assumptions.

    Stay (a=0):
        - Duration increments: bin 0->1 with prob 0.7, stays 0 with 0.3
          bin 1->2 with prob 0.3, stays 1 with 0.7
          bin 2 stays at 2
        - Age increments every ~5 years: 10% per period for bin 0->1, 1->2
        - Income changes with 10% prob (random walk across bins)

    Move (a=1):
        - Duration resets to 0
        - Tenure may flip (15% prob)
        - Income redrawn from cross-sectional distribution
        - Age unchanged
    """
    transitions = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))

    for s in range(NUM_STATES):
        tenure, age_bin, inc_bin, dur_bin = decode_state(s)

        # --- Stay action (a=0) ---
        # Duration transition
        dur_probs = np.zeros(DURATION_BINS)
        if dur_bin == 0:
            dur_probs[0] = 0.3
            dur_probs[1] = 0.7
        elif dur_bin == 1:
            dur_probs[1] = 0.7
            dur_probs[2] = 0.3
        else:
            dur_probs[2] = 1.0

        # Age transition (slow aging)
        age_probs = np.zeros(AGE_BINS)
        if age_bin < AGE_BINS - 1:
            age_probs[age_bin] = 0.90
            age_probs[age_bin + 1] = 0.10
        else:
            age_probs[age_bin] = 1.0

        # Income transition (small random walk)
        inc_trans = np.zeros(INCOME_BINS)
        if inc_bin == 0:
            inc_trans[0] = 0.90
            inc_trans[1] = 0.10
        elif inc_bin == 1:
            inc_trans[0] = 0.05
            inc_trans[1] = 0.90
            inc_trans[2] = 0.05
        else:
            inc_trans[1] = 0.10
            inc_trans[2] = 0.90

        # Combine via outer product (independence assumption)
        for ab_next in range(AGE_BINS):
            for ib_next in range(INCOME_BINS):
                for db_next in range(DURATION_BINS):
                    sp = encode_state(tenure, ab_next, ib_next, db_next)
                    transitions[0, s, sp] += (
                        age_probs[ab_next]
                        * inc_trans[ib_next]
                        * dur_probs[db_next]
                    )

        # --- Move action (a=1) ---
        # Duration resets to 0, tenure may flip, income redrawn
        for tenure_next in range(TENURE_TYPES):
            if tenure_next == tenure:
                ten_prob = 0.85
            else:
                ten_prob = 0.15

            for ab_next in range(AGE_BINS):
                for ib_next in range(INCOME_BINS):
                    sp = encode_state(tenure_next, ab_next, ib_next, 0)
                    transitions[1, s, sp] += (
                        ten_prob
                        * age_probs[ab_next]
                        * inc_probs[ib_next]
                    )

    # Verify row normalization
    for a in range(NUM_ACTIONS):
        row_sums = transitions[a].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), \
            f"Action {a}: row sums not 1, max deviation {np.max(np.abs(row_sums - 1))}"

    return transitions


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(burden_by_state: np.ndarray) -> np.ndarray:
    """Build feature matrix (NUM_STATES, NUM_ACTIONS, NUM_FEATURES).

    Features on stay action (a=0):
        0. housing_burden  -- cost-to-income ratio (renters only)
        1. duration        -- dur_bin normalized (lock-in / attachment)
        2. renter          -- 1 if renter (renters more mobile = lower stay utility)
        3. age             -- age_bin normalized (older = higher stay utility)

    Feature on move action (a=1):
        4. move_cost       -- 1 (transaction cost / hassle of moving)
    """
    features = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_FEATURES))

    dur_midpoints = np.array([1.0, 6.0, 20.0]) / 20.0
    age_midpoints = np.array([25.0, 49.5, 72.5]) / 80.0

    for s in range(NUM_STATES):
        tenure, age_bin, inc_bin, dur_bin = decode_state(s)

        features[s, 0, 0] = burden_by_state[s]
        features[s, 0, 1] = dur_midpoints[dur_bin]
        features[s, 0, 2] = float(tenure)
        features[s, 0, 3] = age_midpoints[age_bin]
        features[s, 1, 4] = 1.0

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("AHS Housing Mobility -- Semi-Synthetic Parameter Recovery")
    print("=" * 72)

    # Step 1: Load cross-section
    print("\n--- Step 1: Load AHS Cross-Section ---")
    moments = load_ahs_moments()

    # Step 2: Build structural transitions
    print("\n--- Step 2: Build Structural Transitions ---")
    transitions = build_structural_transitions(moments["inc_probs"])
    print(f"  Transition shape: {transitions.shape}")

    # Step 3: Build features and problem
    print("\n--- Step 3: Build Features ---")
    features = build_feature_matrix(moments["burden_by_state"])
    utility = LinearUtility(feature_matrix=features, parameter_names=PARAM_NAMES)

    problem = DDCProblem(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        discount_factor=DISCOUNT_FACTOR,
    )

    # Pre-estimation diagnostics
    print("\n--- Pre-Estimation Diagnostics ---")
    F = features.reshape(-1, NUM_FEATURES)
    rank = np.linalg.matrix_rank(F)
    print(f"  Feature matrix rank: {rank} / {NUM_FEATURES}  "
          f"({'full rank' if rank == NUM_FEATURES else 'RANK DEFICIENT'})")
    nonzero_rows = F[F.any(axis=1)]
    if len(nonzero_rows) > 0:
        cond = np.linalg.cond(nonzero_rows)
        print(f"  Condition number: {cond:.1f}  "
              f"({'OK' if cond < 1e6 else 'HIGH'})")

    # Step 4: Compute true policy via soft VI
    print("\n--- Step 4: Compute True Policy ---")
    print(f"  True params: {TRUE_PARAMS}")
    transitions_jnp = jnp.array(transitions, dtype=jnp.float64)
    true_utility = utility.compute(TRUE_PARAM_ARRAY)
    operator = SoftBellmanOperator(problem, transitions_jnp)
    solver_result = value_iteration(operator, true_utility, tol=1e-10)
    print(f"  Soft VI converged: {solver_result.converged} "
          f"({solver_result.num_iterations} iterations)")
    print(f"  Move rate under true policy: "
          f"{float(jnp.dot(jnp.array(moments['initial_dist']), solver_result.policy[:, 1])):.4f}")

    # Step 5: Generate panel from policy
    print(f"\n--- Step 5: Generate {N_INDIVIDUALS:,} Trajectories ---")
    initial_dist = jnp.array(moments["initial_dist"])
    panel = simulate_panel_from_policy(
        problem=problem,
        transitions=transitions_jnp,
        policy=solver_result.policy,
        initial_distribution=initial_dist,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=42,
    )
    n_obs = sum(len(t.states) for t in panel.trajectories)
    all_actions = np.concatenate([np.array(t.actions) for t in panel.trajectories])
    all_states = np.concatenate([np.array(t.states) for t in panel.trajectories])
    state_counts = np.bincount(all_states.astype(np.int32), minlength=NUM_STATES)
    print(f"  Panel: {len(panel.trajectories):,} trajectories, {n_obs:,} observations")
    print(f"  Empirical move rate: {(all_actions == 1).mean():.4f}")
    print(f"  States with observations: {(state_counts > 0).sum()}/{NUM_STATES}")

    # Step 6: Run estimators
    results = {}

    print("\n--- NFXP ---")
    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust")
    results["NFXP"] = nfxp.estimate(panel, utility, problem, transitions_jnp)
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n--- CCP (NPL K=10) ---")
    t0 = time.time()
    ccp = CCPEstimator(num_policy_iterations=10, compute_hessian=True, verbose=True)
    results["CCP"] = ccp.estimate(panel, utility, problem, transitions_jnp)
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n--- NNES (NPL variant, Nguyen 2025) ---")
    t0 = time.time()
    nnes = NNESEstimator(
        hidden_dim=32,
        num_layers=2,
        v_epochs=200,
        v_lr=1e-3,
        n_outer_iterations=3,
        verbose=False,
    )
    results["NNES"] = nnes.estimate(panel, utility, problem, transitions_jnp)
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n--- TD-CCP (semigradient, Adusumilli & Eckardt 2025) ---")
    t0 = time.time()
    tdccp = TDCCPEstimator(config=TDCCPConfig(
        method="semigradient",
        basis_dim=8,
        cross_fitting=True,
        robust_se=False,
    ))
    results["TD-CCP"] = tdccp.estimate(panel, utility, problem, transitions_jnp)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Step 7: Parameter recovery table
    print("\n" + "=" * 72)
    print("Parameter Recovery")
    print("=" * 72)

    header = f"{'':>18} {'True':>10} {'NFXP':>10} {'CCP':>10} {'NNES':>10} {'TD-CCP':>10}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(PARAM_NAMES):
        true_val = float(TRUE_PARAM_ARRAY[i])
        nfxp_val = float(results["NFXP"].parameters[i])
        ccp_val = float(results["CCP"].parameters[i])
        nnes_val = float(results["NNES"].parameters[i])
        tdccp_val = float(results["TD-CCP"].parameters[i])
        print(f"{name:>18} {true_val:>10.4f} {nfxp_val:>10.4f} "
              f"{ccp_val:>10.4f} {nnes_val:>10.4f} {tdccp_val:>10.4f}")

    print(f"\n{'Standard Errors':>18} {'':>10} {'NFXP':>10} {'CCP':>10} {'NNES':>10} {'TD-CCP':>10}")
    print("-" * len(header))
    for i, name in enumerate(PARAM_NAMES):
        ses = {}
        for est_name in ["NFXP", "CCP", "NNES", "TD-CCP"]:
            r = results[est_name]
            if r.standard_errors is not None:
                ses[est_name] = float(r.standard_errors[i])
            else:
                ses[est_name] = float("nan")
        print(f"{name:>18} {'':>10} {ses['NFXP']:>10.4f} "
              f"{ses['CCP']:>10.4f} {ses['NNES']:>10.4f} {ses['TD-CCP']:>10.4f}")

    # Save results
    out = {
        "dataset": "ahs-housing",
        "methodology": "semi-synthetic parameter recovery",
        "true_parameters": TRUE_PARAMS,
        "n_observations": n_obs,
        "n_individuals": N_INDIVIDUALS,
        "discount_factor": DISCOUNT_FACTOR,
        "diagnostics": {
            "feature_rank": int(rank),
            "state_coverage": int((state_counts > 0).sum()),
            "n_states": NUM_STATES,
        },
        "cross_section_moments": {
            "renter_share": moments["renter_share"],
            "n_valid_households": moments["n_valid"],
        },
    }
    for est_name, r in results.items():
        out[est_name] = {
            "parameters": {
                name: float(r.parameters[i])
                for i, name in enumerate(PARAM_NAMES)
            },
            "standard_errors": {
                name: float(r.standard_errors[i])
                for i, name in enumerate(PARAM_NAMES)
            } if r.standard_errors is not None else None,
            "log_likelihood": float(r.log_likelihood),
            "converged": bool(r.converged),
        }

    out_path = Path("examples/ahs-housing/results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
