#!/usr/bin/env python3
"""
Rust (1987) Bus Engine Replacement — Full Replication
=====================================================

Replicates the structural estimation from:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.

Using the NFXP algorithm with the SA→NK polyalgorithm from:
    Iskhakov, Jorgensen, Rust & Schjerning (2016). "Comment on Constrained
    Optimization Approaches to Estimation of Structural Models." Econometrica.

Implementation follows the ruspy reference (OpenSourceEconomics/ruspy).

Pipeline:
    1. Load original Rust (1987) data (162 buses, 4 groups, Madison Metro)
    2. Estimate transition probabilities from data (first stage)
    3. Estimate structural parameters via NFXP with BHHH + analytical gradient
    4. Cross-validate with CCP/NPL estimator
    5. Per-group estimation
    6. Compare with ruspy (if installed)

Usage:
    python examples/rust_bus_replication.py
"""

import time

import numpy as np
import torch

from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.nfxp import NFXPEstimator, estimate_transitions_from_panel
from econirl.estimation.transitions import estimate_transition_probs
from econirl.preferences.linear import LinearUtility


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def run_nfxp(panel, utility, problem, transitions, verbose=False):
    """Run NFXP with BHHH optimizer and analytical gradient."""
    estimator = NFXPEstimator(
        optimizer="BHHH",
        inner_solver="policy",
        inner_tol=1e-12,
        inner_max_iter=200,
        analytical_gradient=True,
        compute_hessian=True,
        outer_tol=1e-3,
        verbose=verbose,
    )
    return estimator.estimate(panel, utility, problem, transitions)


def run_npl(panel, utility, problem, transitions, K=20):
    """Run CCP-NPL (Aguirregabiria-Mira) for cross-validation."""
    estimator = CCPEstimator(
        num_policy_iterations=K,
        compute_hessian=True,
        verbose=False,
    )
    return estimator.estimate(panel, utility, problem, transitions)


def main():
    print_header("Rust (1987) Bus Engine Replacement — Full Replication")
    print("econirl NFXP with BHHH + analytical gradient (Iskhakov et al. 2016)")

    # =========================================================================
    # Step 1: Load Original Data
    # =========================================================================
    print_section("Step 1: Load Original Data")

    df = load_rust_bus(original=True)
    panel = load_rust_bus(original=True, as_panel=True)

    n_obs = len(df)
    n_buses = df["bus_id"].nunique()
    n_groups = df["group"].nunique()
    n_replace = df["replaced"].sum()
    replace_rate = df["replaced"].mean()

    print(f"  Observations:     {n_obs:,}")
    print(f"  Buses:            {n_buses}")
    print(f"  Groups:           {n_groups} (Grumman 870, RT-50, T8H203, A5308)")
    print(f"  Replacements:     {n_replace} ({replace_rate:.2%})")
    print(f"  Mileage range:    bins {df['mileage_bin'].min()}-{df['mileage_bin'].max()}")
    print(f"  Mean mileage:     bin {df['mileage_bin'].mean():.1f}")

    # =========================================================================
    # Step 2: Estimate Transition Probabilities (First Stage)
    # =========================================================================
    print_section("Step 2: Estimate Transition Probabilities (First Stage)")

    trans_probs = estimate_transition_probs(df)
    print(f"  P(dx=0) = {trans_probs[0]:.4f}")
    print(f"  P(dx=1) = {trans_probs[1]:.4f}")
    print(f"  P(dx=2) = {trans_probs[2]:.4f}")

    # Build transition matrices from estimated probabilities
    transitions = estimate_transitions_from_panel(panel, num_states=90, max_increment=2)

    # Also estimate per group
    from econirl.estimation.transitions import estimate_transition_probs_by_group

    trans_by_group = estimate_transition_probs_by_group(df)
    print("\n  Per-group transition probabilities:")
    group_names = {1: "Grumman 870", 2: "Chance RT-50", 3: "GMC T8H203", 4: "GMC A5308"}
    for g, probs in trans_by_group.items():
        n = len(df[df["group"] == g])
        print(f"    Group {g} ({group_names.get(g, '?'):>14}): ({probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f})  n={n}")

    # =========================================================================
    # Step 3: NFXP Estimation (Pooled, All Groups)
    # =========================================================================
    print_section("Step 3: NFXP Estimation (Pooled, All Groups)")

    env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec

    t0 = time.time()
    result_nfxp = run_nfxp(panel, utility, problem, transitions, verbose=True)
    t_nfxp = time.time() - t0

    print(f"\n  Time: {t_nfxp:.1f}s")
    print(result_nfxp.summary())

    # Convert to Rust parameterization
    op_cost = result_nfxp.parameters[0].item()
    rc = result_nfxp.parameters[1].item()
    rust_c = op_cost / 0.001
    print(f"  Rust parameterization: c = {rust_c:.4f}, RC = {rc:.4f}")
    print(f"  (cost function: c * 0.001 * mileage_bin)")

    # =========================================================================
    # Step 4: Cross-Validate with NPL
    # =========================================================================
    print_section("Step 4: Cross-Validate with CCP-NPL (Aguirregabiria-Mira)")

    t0 = time.time()
    result_npl = run_npl(panel, utility, problem, transitions)
    t_npl = time.time() - t0

    npl_op = result_npl.parameters[0].item()
    npl_rc = result_npl.parameters[1].item()

    print(f"  NPL:  operating_cost = {npl_op:.6f}, RC = {npl_rc:.4f}, LL = {result_npl.log_likelihood:.4f}  ({t_npl:.1f}s)")
    print(f"  NFXP: operating_cost = {op_cost:.6f}, RC = {rc:.4f}, LL = {result_nfxp.log_likelihood:.4f}  ({t_nfxp:.1f}s)")
    print(f"  Agreement: |dLL| = {abs(result_nfxp.log_likelihood - result_npl.log_likelihood):.4f}")

    # =========================================================================
    # Step 5: Per-Group Estimation
    # =========================================================================
    print_section("Step 5: Per-Group NFXP Estimation")

    print(f"\n  {'Group':<8} {'Name':<16} {'Buses':>5} {'Obs':>6} {'Repl':>5} {'op_cost':>10} {'RC':>10} {'c':>8} {'LL':>10}")
    print("  " + "-" * 90)

    for group_id in sorted(df["group"].unique()):
        group_df = df[df["group"] == group_id]
        group_panel = load_rust_bus(original=True, group=group_id, as_panel=True)
        group_trans = estimate_transitions_from_panel(group_panel, num_states=90, max_increment=2)

        try:
            result_g = run_nfxp(group_panel, utility, problem, group_trans)
            g_op = result_g.parameters[0].item()
            g_rc = result_g.parameters[1].item()
            g_c = g_op / 0.001
            g_ll = result_g.log_likelihood
            n_repl = group_df["replaced"].sum()
            print(
                f"  {group_id:<8} {group_names.get(group_id, '?'):<16} "
                f"{group_df['bus_id'].nunique():>5} {len(group_df):>6} {n_repl:>5} "
                f"{g_op:>10.6f} {g_rc:>10.4f} {g_c:>8.4f} {g_ll:>10.2f}"
            )
        except Exception as e:
            print(f"  {group_id:<8} {group_names.get(group_id, '?'):<16}  FAILED: {e}")

    # =========================================================================
    # Step 6: Cross-Validate with ruspy (if installed)
    # =========================================================================
    print_section("Step 6: Cross-Validate with ruspy")

    try:
        from ruspy.model_code.fix_point_alg import calc_fixp
        from ruspy.model_code.choice_probabilities import choice_prob_gumbel
        from ruspy.model_code.cost_functions import calc_obs_costs, lin_cost

        # Build ruspy transition matrix
        n = 90
        tm = np.zeros((n, n))
        for i in range(n):
            for j in range(len(trans_probs)):
                tm[i, min(i + j, n - 1)] += trans_probs[j]

        # Evaluate at our NFXP estimates
        params_rust_fmt = np.array([rc, rust_c])
        costs = calc_obs_costs(n, lin_cost, params_rust_fmt, 0.001)

        ev_result = calc_fixp(tm, costs, 0.9999)
        ev = ev_result[0] if isinstance(ev_result, tuple) else ev_result
        probs = choice_prob_gumbel(ev, costs, 0.9999)

        # Compute LL
        ruspy_ll = 0.0
        for bus_id in sorted(df["bus_id"].unique()):
            bus = df[df["bus_id"] == bus_id].sort_values("period")
            for _, row in bus.iterrows():
                s = int(row["mileage_bin"])
                d = int(row["replaced"])  # 0=keep, 1=replace
                ruspy_ll += np.log(max(probs[s, d], 1e-300))

        print(f"  ruspy LL at our params:  {ruspy_ll:.4f}")
        print(f"  econirl LL:              {result_nfxp.log_likelihood:.4f}")
        print(f"  |difference|:            {abs(ruspy_ll - result_nfxp.log_likelihood):.6f}")
        print("  Status: MATCH" if abs(ruspy_ll - result_nfxp.log_likelihood) < 0.01 else "  Status: MISMATCH")

    except ImportError:
        print("  ruspy not installed. Install with: pip install git+https://github.com/OpenSourceEconomics/ruspy.git")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Replication Summary")

    print(f"""
  Model: Rust (1987) Table IX, Model 11 (linear cost, beta=0.9999)
  Data:  {n_obs:,} observations, {n_buses} buses, {n_groups} groups

  Pooled Estimates (NFXP + BHHH + analytical gradient):
    operating_cost = {op_cost:.6f}  (SE {result_nfxp.standard_errors[0]:.6f})
    replacement_cost = {rc:.4f}  (SE {result_nfxp.standard_errors[1]:.4f})
    Log-likelihood = {result_nfxp.log_likelihood:.4f}

  Rust parameterization: c = {rust_c:.4f}, RC = {rc:.4f}
  (cost = c * 0.001 * mileage_bin, where each bin = 5,000 miles)

  Cross-validation:
    NFXP vs NPL: |dLL| = {abs(result_nfxp.log_likelihood - result_npl.log_likelihood):.4f}

  Algorithm: Iskhakov et al. (2016) SA→NK polyalgorithm
    Inner solver: Policy iteration (9 iters at beta=0.9999)
    Outer solver: BHHH with analytical gradient
    Precision: float64 for inner solver (cond(I-beta*P) ~ 10^5)
""")


if __name__ == "__main__":
    main()
