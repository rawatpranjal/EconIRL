#!/usr/bin/env python3
"""
Freddie Mac Mortgage -- Semi-Synthetic Parameter Recovery
=========================================================

Estimates structural parameters of a mortgage repayment model using
transition dynamics calibrated from Freddie Mac origination data.
Known ground truth reward parameters are used to generate data from
the soft VI optimal policy. Four estimators (NFXP, CCP, NNES, TD-CCP)
recover the parameters, demonstrating structural and neural estimation
on a three-action mortgage termination problem.

State: (delinquency, credit_bin, ltv_bin) -> 4 x 2 x 3 = 24 states
    delinquency: current=0, 30day=1, 60day=2, 90day+=3
    credit_bin:  low(<660)=0, high(660+)=1
    ltv_bin:     low(<70%)=0, medium(70-90%)=1, high(90%+)=2

Action: pay=0, prepay=1, default=2

Features (6):
    0. delinquency  -- delinq level normalized (pay action)
    1. credit       -- credit_bin (pay action)
    2. ltv          -- ltv_bin normalized (pay action)
    3. equity       -- 1 - ltv normalized (prepay action)
    4. prepay_cost  -- 1 for prepay (refinancing cost)
    5. default_cost -- 1 for default (credit damage)

Data: Freddie Mac Single-Family Loan-Level Dataset
    Prepared_Dataset.csv: 148K loans with CreditScore, LTV,
    EverDelinquent, MonthsDelinquent, YearsInRepayment

Reference:
    Campbell, J.Y., Cocco, J.F. (2015). "A Model of Mortgage Default."
    Journal of Finance 70(4): 1495-1554.

Usage:
    python examples/freddie-mac/run_ccp.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration
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

DATA_DIR = Path("data/raw/freddie_mac_small")
N_INDIVIDUALS = 10_000
N_PERIODS = 60          # 5 years of monthly decisions
DISCOUNT_FACTOR = 0.99  # monthly (0.99^12 ~ 0.886 annual)

# State dimensions
DELINQ_LEVELS = 4   # current=0, 30day=1, 60day=2, 90day+=3
CREDIT_BINS = 2     # low(<660)=0, high(660+)=1
LTV_BINS = 3        # low(<70%)=0, medium(70-90%)=1, high(90%+)=2

NUM_STATES = DELINQ_LEVELS * CREDIT_BINS * LTV_BINS   # 24
NUM_ACTIONS = 3    # pay=0, prepay=1, default=2
NUM_FEATURES = 6

# Ground truth reward parameters
TRUE_PARAMS = {
    "theta_delinquency": -0.80,   # distress reduces pay utility
    "theta_credit": 0.50,         # good credit supports paying
    "theta_ltv": -0.40,           # underwater stress reduces pay utility
    "theta_equity": 0.60,         # equity makes prepay attractive
    "prepay_cost": -1.00,         # refinancing transaction cost
    "default_cost": -2.00,        # credit damage and legal cost
}
TRUE_PARAM_ARRAY = jnp.array(list(TRUE_PARAMS.values()))
PARAM_NAMES = list(TRUE_PARAMS.keys())


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(delinq: int, credit_bin: int, ltv_bin: int) -> int:
    return delinq * CREDIT_BINS * LTV_BINS + credit_bin * LTV_BINS + ltv_bin


def decode_state(s: int) -> tuple[int, int, int]:
    ltv_bin = s % LTV_BINS
    remainder = s // LTV_BINS
    credit_bin = remainder % CREDIT_BINS
    delinq = remainder // CREDIT_BINS
    return delinq, credit_bin, ltv_bin


# ---------------------------------------------------------------------------
# Load cross-section for calibration
# ---------------------------------------------------------------------------

def load_freddie_moments() -> dict:
    """Load Freddie Mac cross-section and extract calibration moments."""
    print("Loading Prepared_Dataset.csv...")
    df = pl.read_csv(
        str(DATA_DIR / "Prepared_Dataset.csv"),
        null_values=["", "NA"],
    )
    print(f"  Loaded {df.shape[0]:,} loans")

    df = df.select(["CreditScore", "LTV", "EverDelinquent",
                    "MonthsDelinquent", "YearsInRepayment", "OrigLoanTerm"])

    ever_delinq_rate = float(df["EverDelinquent"].mean())
    credit_high_share = float((df["CreditScore"] >= 660).mean())
    ltv_low = float((df["LTV"] < 70).mean())
    ltv_mid = float(((df["LTV"] >= 70) & (df["LTV"] < 90)).mean())
    ltv_high = float((df["LTV"] >= 90).mean())

    # Initial state distribution: most loans start current, distribute by credit/LTV
    credit_probs = np.array([1 - credit_high_share, credit_high_share])
    ltv_probs = np.array([ltv_low, ltv_mid, ltv_high])

    # Build initial distribution (all start current, delinq=0)
    initial_dist = np.zeros(NUM_STATES)
    for cb in range(CREDIT_BINS):
        for lb in range(LTV_BINS):
            s = encode_state(0, cb, lb)
            initial_dist[s] = credit_probs[cb] * ltv_probs[lb]
    initial_dist /= initial_dist.sum()

    print(f"  Ever delinquent: {ever_delinq_rate:.3f}")
    print(f"  High credit (660+): {credit_high_share:.3f}")
    print(f"  LTV: low={ltv_low:.3f}, mid={ltv_mid:.3f}, high={ltv_high:.3f}")

    return {
        "initial_dist": initial_dist,
        "ever_delinq_rate": ever_delinq_rate,
        "credit_high_share": credit_high_share,
        "ltv_dist": [ltv_low, ltv_mid, ltv_high],
    }


# ---------------------------------------------------------------------------
# Build structural transition matrices
# ---------------------------------------------------------------------------

def build_structural_transitions() -> np.ndarray:
    """Structural transition matrices based on mortgage dynamics.

    Pay (a=0): delinquency can worsen or cure, LTV slowly amortizes.
    Prepay (a=1): absorbing (loan exits, state 0).
    Default (a=2): absorbing (loan exits, state 0).

    Delinquency dynamics for pay action:
        Current (0): 95% stay current, 5% -> 30day
        30day (1):   30% cure -> current, 30% stay, 40% -> 60day
        60day (2):   20% cure -> 30day, 30% stay, 50% -> 90day+
        90day+ (3):  10% cure -> 60day, 90% stay

    LTV amortization: 2% chance per period of dropping one bin.
    Credit bin is fixed (time-invariant characteristic).
    """
    transitions = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))

    for s in range(NUM_STATES):
        delinq, credit_bin, ltv_bin = decode_state(s)

        # --- Pay action (a=0): delinquency + LTV transitions ---
        # Delinquency transition probabilities
        delinq_trans = np.zeros(DELINQ_LEVELS)
        if delinq == 0:
            delinq_trans[0] = 0.95
            delinq_trans[1] = 0.05
        elif delinq == 1:
            delinq_trans[0] = 0.30
            delinq_trans[1] = 0.30
            delinq_trans[2] = 0.40
        elif delinq == 2:
            delinq_trans[1] = 0.20
            delinq_trans[2] = 0.30
            delinq_trans[3] = 0.50
        else:
            delinq_trans[2] = 0.10
            delinq_trans[3] = 0.90

        # LTV amortization (slow)
        ltv_trans = np.zeros(LTV_BINS)
        if ltv_bin > 0:
            ltv_trans[ltv_bin] = 0.98
            ltv_trans[ltv_bin - 1] = 0.02
        else:
            ltv_trans[0] = 1.0

        # Combine (credit stays fixed)
        for dq_next in range(DELINQ_LEVELS):
            for lv_next in range(LTV_BINS):
                sp = encode_state(dq_next, credit_bin, lv_next)
                transitions[0, s, sp] += delinq_trans[dq_next] * ltv_trans[lv_next]

        # --- Prepay (a=1): absorbing, reset to state 0 ---
        transitions[1, s, 0] = 1.0

        # --- Default (a=2): absorbing, reset to state 0 ---
        transitions[2, s, 0] = 1.0

    # Verify row normalization
    for a in range(NUM_ACTIONS):
        row_sums = transitions[a].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), \
            f"Action {a}: row sums not 1"

    return transitions


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix() -> np.ndarray:
    """Build feature matrix (NUM_STATES, NUM_ACTIONS, NUM_FEATURES).

    Pay (a=0): delinquency, credit, ltv
    Prepay (a=1): equity, prepay_cost
    Default (a=2): default_cost
    """
    features = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_FEATURES))

    for s in range(NUM_STATES):
        delinq, credit_bin, ltv_bin = decode_state(s)

        # Pay action (a=0)
        features[s, 0, 0] = delinq / 3.0           # delinquency level
        features[s, 0, 1] = float(credit_bin)       # credit quality
        features[s, 0, 2] = ltv_bin / 2.0           # LTV stress

        # Prepay action (a=1)
        features[s, 1, 3] = 1.0 - ltv_bin / 2.0    # equity (1 - normalized LTV)
        features[s, 1, 4] = 1.0                     # prepay cost intercept

        # Default action (a=2)
        features[s, 2, 5] = 1.0                     # default cost intercept

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("Freddie Mac Mortgage -- Semi-Synthetic Parameter Recovery")
    print("=" * 72)

    # Step 1: Load cross-section
    print("\n--- Step 1: Load Freddie Mac Cross-Section ---")
    moments = load_freddie_moments()

    # Step 2: Build structural transitions
    print("\n--- Step 2: Build Structural Transitions ---")
    transitions = build_structural_transitions()
    print(f"  Transition shape: {transitions.shape}")

    # Step 3: Build features and problem
    print("\n--- Step 3: Build Features ---")
    features = build_feature_matrix()
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
    # Use policy_iteration for high beta (0.99)
    solver_result = policy_iteration(
        operator, true_utility, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    print(f"  Solver converged: {solver_result.converged} "
          f"({solver_result.num_iterations} iterations)")

    # Report action shares under true policy
    init_dist = jnp.array(moments["initial_dist"])
    pay_rate = float(jnp.dot(init_dist, solver_result.policy[:, 0]))
    prepay_rate = float(jnp.dot(init_dist, solver_result.policy[:, 1]))
    default_rate = float(jnp.dot(init_dist, solver_result.policy[:, 2]))
    print(f"  Action shares (initial states): pay={pay_rate:.3f}, "
          f"prepay={prepay_rate:.3f}, default={default_rate:.3f}")

    # Step 5: Generate panel from policy
    print(f"\n--- Step 5: Generate {N_INDIVIDUALS:,} Loan Trajectories ---")
    panel = simulate_panel_from_policy(
        problem=problem,
        transitions=transitions_jnp,
        policy=solver_result.policy,
        initial_distribution=init_dist,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=42,
    )
    n_obs = sum(len(t.states) for t in panel.trajectories)
    all_actions = np.concatenate([np.array(t.actions) for t in panel.trajectories])
    all_states = np.concatenate([np.array(t.states) for t in panel.trajectories])
    state_counts = np.bincount(all_states.astype(np.int32), minlength=NUM_STATES)
    print(f"  Panel: {len(panel.trajectories):,} trajectories, {n_obs:,} observations")
    print(f"  Action shares: pay={(all_actions == 0).mean():.3f}, "
          f"prepay={(all_actions == 1).mean():.3f}, "
          f"default={(all_actions == 2).mean():.3f}")
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

    header = f"{'':>20} {'True':>10} {'NFXP':>10} {'CCP':>10} {'NNES':>10} {'TD-CCP':>10}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(PARAM_NAMES):
        true_val = float(TRUE_PARAM_ARRAY[i])
        nfxp_val = float(results["NFXP"].parameters[i])
        ccp_val = float(results["CCP"].parameters[i])
        nnes_val = float(results["NNES"].parameters[i])
        tdccp_val = float(results["TD-CCP"].parameters[i])
        print(f"{name:>20} {true_val:>10.4f} {nfxp_val:>10.4f} "
              f"{ccp_val:>10.4f} {nnes_val:>10.4f} {tdccp_val:>10.4f}")

    print(f"\n{'Standard Errors':>20} {'':>10} {'NFXP':>10} {'CCP':>10} {'NNES':>10} {'TD-CCP':>10}")
    print("-" * len(header))
    for i, name in enumerate(PARAM_NAMES):
        ses = {}
        for est_name in ["NFXP", "CCP", "NNES", "TD-CCP"]:
            r = results[est_name]
            if r.standard_errors is not None:
                ses[est_name] = float(r.standard_errors[i])
            else:
                ses[est_name] = float("nan")
        print(f"{name:>20} {'':>10} {ses['NFXP']:>10.4f} "
              f"{ses['CCP']:>10.4f} {ses['NNES']:>10.4f} {ses['TD-CCP']:>10.4f}")

    # Save results
    out = {
        "dataset": "freddie-mac",
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
        "calibration_moments": {
            "ever_delinq_rate": moments["ever_delinq_rate"],
            "credit_high_share": moments["credit_high_share"],
            "ltv_dist": moments["ltv_dist"],
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

    out_path = Path("examples/freddie-mac/results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
