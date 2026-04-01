"""Main identification experiment: AIRL vs IQ-Learn vs Reduced-Form Q.

Runs two phases:
  Phase 1: Estimate with all methods, compare reward recovery and in-sample fit.
  Phase 2: Evaluate Type I and Type II counterfactuals, show divergence.

Usage:
    python experiments/identification/run_main.py [--n_individuals 500]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from identification.config import ExperimentConfig, DGPConfig, EstimationConfig, CounterfactualConfig
from identification.environment import SerializedContentEnvironment, EXIT
from identification.estimators import (
    estimate_oracle,
    estimate_reduced_form_q,
    estimate_airl_with_anchors,
    estimate_airl_no_anchors,
    estimate_iq_learn,
    oracle_counterfactual_ccps,
)
from identification.counterfactuals import (
    evaluate_type_i,
    evaluate_type_ii,
    evaluate_type_i_from_q,
    evaluate_type_ii_from_q,
    solve_policy,
)
from identification.metrics import reward_mse, reward_correlation, ccp_error


def run_analytical_verification(cfg: ExperimentConfig) -> dict:
    """Analytical (population-level) verification of the taxonomy.

    Bypasses estimation entirely: constructs population-level objects
    (true reward, shaped reward, advantage) and tests counterfactual
    accuracy directly. This proves the theory independent of any
    optimization convergence issues.
    """
    print("=" * 70)
    print("PART A: ANALYTICAL VERIFICATION (population level)")
    print("=" * 70)

    env = SerializedContentEnvironment(cfg.dgp)
    problem = env.problem_spec
    absorbing = cfg.dgp.absorbing_state
    transitions = env.transition_matrices
    true_reward = env.reward_matrix

    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration

    operator = SoftBellmanOperator(problem, transitions)
    vi = value_iteration(operator, true_reward, tol=1e-10, max_iter=10000)
    true_V = vi.V
    true_Q = vi.Q
    true_policy = vi.policy

    sigma = cfg.dgp.scale_parameter

    # Construct the 5 method-level reward matrices at population level:
    # 1. Oracle: true r*
    # 2. AIRL + anchors: r* exactly (anchors pin shaping to zero)
    # 3. IQ-Learn: r* exactly (inverse Bellman from true Q)
    # 4. AIRL no anchors: r* + shaping residual delta
    #    Use delta(s) = 0.5 * V*(s) as a representative shaping function
    # 5. Reduced-form Q: advantage A*(s,a) = Q*(s,a) - V*(s) (not reward)

    # Oracle = true reward
    r_oracle = true_reward

    # AIRL + anchors recovers r* exactly (Theorems 1-3)
    r_airl_anchored = jnp.array(true_reward)

    # IQ-Learn recovers r via inverse Bellman from Q*
    EV = jnp.einsum("ast,t->as", transitions, true_V).T  # (S, A)
    r_iq = true_Q - cfg.dgp.discount_factor * EV
    # r_iq should equal true_reward exactly

    # AIRL no anchors: r* + h(s) - beta*h(s') where h(s) = 0.5*V*(s)
    h = 0.5 * true_V
    # For deterministic transitions, h(s') is just h at the next state
    # r_shaped(s,a) = r*(s,a) + h(s) - beta * sum_s' P(s'|s,a) h(s')
    Eh = jnp.einsum("ast,t->as", transitions, h).T
    r_airl_no_anchors = true_reward + h[:, None] - cfg.dgp.discount_factor * Eh

    # Reduced-form Q: only has A*(s,a), not r*(s,a)
    A_star = true_Q - true_V[:, None]
    # Use A* as the "reward" for re-solving (will fail for Type II)

    methods = [
        ("Oracle", r_oracle),
        ("AIRL + Anchors", r_airl_anchored),
        ("IQ-Learn", r_iq),
        ("AIRL No Anchors", r_airl_no_anchors),
        ("Reduced-Form Q (A*)", A_star),
    ]

    # Reward recovery
    print("\nReward Recovery:")
    for name, r in methods:
        mse = reward_mse(r, true_reward, absorbing, EXIT)
        corr = reward_correlation(r, true_reward, absorbing, EXIT)
        print(f"  {name:25s}  MSE={mse:.6f}  Corr={corr:.4f}")

    # Type I counterfactual
    wait_shift = cfg.counterfactual.type_i_wait_shift
    shifted_reward = env.build_shifted_reward(wait_shift)
    oracle_type_i = oracle_counterfactual_ccps(env, shifted_reward=shifted_reward)
    delta_r = shifted_reward - true_reward

    print(f"\nType I counterfactual (wait shift = {wait_shift}):")
    for name, r in methods:
        r_shifted = r + delta_r
        cf_policy, _ = solve_policy(r_shifted, transitions, problem)
        err = ccp_error(cf_policy, oracle_type_i, absorbing)
        print(f"  {name:25s}  CCP Error = {err:.6f}")

    # Type II counterfactual
    print(f"\nType II counterfactual (buy skips k episodes):")
    print(f"  {'Method':25s}", end="")
    for k in cfg.counterfactual.type_ii_skip_values:
        print(f"  k={k:2d}", end="")
    print()

    analytical_type_ii = {}
    for name, r in methods:
        print(f"  {name:25s}", end="")
        row = {}
        for k in cfg.counterfactual.type_ii_skip_values:
            new_trans = env.build_skip_transitions(k)
            oracle_cf = oracle_counterfactual_ccps(env, new_transitions=new_trans)
            est_cf, _ = solve_policy(r, new_trans, problem)
            err = ccp_error(est_cf, oracle_cf, absorbing)
            row[k] = err
            print(f"  {err:.4f}", end="")
        print()
        analytical_type_ii[name] = row

    return analytical_type_ii


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run the full identification experiment."""

    print("=" * 70)
    print("IDENTIFICATION EXPERIMENT: AIRL vs IQ-Learn vs Reduced-Form Q")
    print("=" * 70)

    # --- Setup ---
    env = SerializedContentEnvironment(cfg.dgp)
    problem = env.problem_spec
    absorbing = cfg.dgp.absorbing_state
    transitions = env.transition_matrices

    print(f"\nDGP: {cfg.dgp.num_episodes} episodes + 1 absorbing state")
    print(f"Actions: buy=0, wait=1, exit=2")
    print(f"Discount factor: {cfg.dgp.discount_factor}")
    print(f"True parameters: {env.true_parameters}")

    # --- Phase 1: Estimation ---
    print(f"\n{'='*70}")
    print(f"PHASE 1: Estimation (N={cfg.estimation.n_individuals} individuals, "
          f"T={cfg.estimation.n_periods} periods)")
    print(f"{'='*70}")

    panel = env.generate_panel(
        n_individuals=cfg.estimation.n_individuals,
        n_periods=cfg.estimation.n_periods,
        seed=cfg.estimation.seed,
    )
    print(f"Generated {panel.num_observations} observations")

    # Oracle
    print("\n[1/5] Oracle (true reward)...")
    t0 = time.time()
    oracle = estimate_oracle(env)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Reduced-Form Q
    print("[2/5] Reduced-Form Q (logit MLE)...")
    t0 = time.time()
    rf_q = estimate_reduced_form_q(panel, env, cfg.estimation)
    print(f"  Done in {time.time()-t0:.1f}s, LL={rf_q['log_likelihood']:.1f}")

    # AIRL with anchors
    print("[3/5] AIRL + Anchors...")
    t0 = time.time()
    airl_anchored = estimate_airl_with_anchors(panel, env, cfg.estimation)
    print(f"  Done in {time.time()-t0:.1f}s, LL={airl_anchored['log_likelihood']:.1f}")

    # AIRL without anchors
    print("[4/5] AIRL No Anchors...")
    t0 = time.time()
    airl_no_anchors = estimate_airl_no_anchors(panel, env, cfg.estimation)
    print(f"  Done in {time.time()-t0:.1f}s, LL={airl_no_anchors['log_likelihood']:.1f}")

    # IQ-Learn
    print("[5/5] IQ-Learn...")
    t0 = time.time()
    iq = estimate_iq_learn(panel, env, cfg.estimation)
    print(f"  Done in {time.time()-t0:.1f}s, LL={iq['log_likelihood']:.1f}")

    methods = [oracle, rf_q, airl_anchored, airl_no_anchors, iq]

    # --- Phase 1 Metrics ---
    print(f"\n{'='*70}")
    print("PHASE 1 RESULTS: Reward Recovery")
    print(f"{'='*70}")

    true_reward = env.reward_matrix
    rows = []
    for m in methods:
        r = m["reward_matrix"]
        if r is not None:
            mse = reward_mse(r, true_reward, absorbing, EXIT)
            corr = reward_correlation(r, true_reward, absorbing, EXIT)
        else:
            mse = float("nan")
            corr = float("nan")

        in_sample_err = ccp_error(m["policy"], oracle["policy"], absorbing)

        rows.append({
            "Method": m["name"],
            "Reward MSE": mse,
            "Reward Corr": corr,
            "In-Sample CCP Err": in_sample_err,
        })

    df_phase1 = pd.DataFrame(rows)
    print(df_phase1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Phase 2: Counterfactuals ---
    print(f"\n{'='*70}")
    print("PHASE 2: Counterfactual Evaluation")
    print(f"{'='*70}")

    # Type I: shift wait-cost
    wait_shift = cfg.counterfactual.type_i_wait_shift
    shifted_reward = env.build_shifted_reward(wait_shift)
    oracle_type_i = oracle_counterfactual_ccps(env, shifted_reward=shifted_reward)

    # The reward shift (delta_r) needed for RF-Q Type I
    delta_r = shifted_reward - true_reward

    print(f"\nType I counterfactual: wait-cost shift = {wait_shift}")

    type_i_rows = []
    for m in methods:
        if m["name"] == "Reduced-Form Q":
            cf_ccps = evaluate_type_i_from_q(m["q_table"], delta_r, problem.scale_parameter)
        elif m["reward_matrix"] is not None:
            # Compute shifted reward from the estimated reward
            # For the oracle & structural methods: substitute shifted reward
            # and re-solve. But note: these methods estimated reward under
            # original features. For Type I, we need to add the same delta.
            est_shifted_reward = m["reward_matrix"] + delta_r
            cf_ccps = evaluate_type_i(
                m["reward_matrix"], transitions, problem, est_shifted_reward,
            )
        else:
            continue

        err = ccp_error(cf_ccps, oracle_type_i, absorbing)
        type_i_rows.append({"Method": m["name"], "Type I CCP Error": err})

    df_type_i = pd.DataFrame(type_i_rows)
    print(df_type_i.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Type II: buy skips k episodes
    print(f"\nType II counterfactual: buy skips k episodes")

    type_ii_results = []
    for k in cfg.counterfactual.type_ii_skip_values:
        new_trans = env.build_skip_transitions(k)
        oracle_type_ii = oracle_counterfactual_ccps(env, new_transitions=new_trans)

        row = {"skip_k": k}
        for m in methods:
            if m["name"] == "Reduced-Form Q":
                # RF-Q cannot re-solve: returns original CCPs
                cf_ccps = evaluate_type_ii_from_q(m["q_table"], problem.scale_parameter)
            elif m["reward_matrix"] is not None:
                cf_ccps = evaluate_type_ii(m["reward_matrix"], new_trans, problem)
            else:
                continue
            err = ccp_error(cf_ccps, oracle_type_ii, absorbing)
            row[m["name"]] = err

        type_ii_results.append(row)

    df_type_ii = pd.DataFrame(type_ii_results)
    print(df_type_ii.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # --- Summary Table ---
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")

    # Use k=3 for the headline Type II number
    type_ii_k3 = None
    for r in type_ii_results:
        if r["skip_k"] == 3:
            type_ii_k3 = r
            break

    summary_rows = []
    for m in methods:
        r_mse = [r for r in rows if r["Method"] == m["name"]][0]["Reward MSE"]
        t1_err = [r for r in type_i_rows if r["Method"] == m["name"]]
        t1 = t1_err[0]["Type I CCP Error"] if t1_err else float("nan")
        t2 = type_ii_k3.get(m["name"], float("nan")) if type_ii_k3 else float("nan")
        summary_rows.append({
            "Method": m["name"],
            "Reward MSE": r_mse,
            "Type I CCP Err": t1,
            "Type II CCP Err (k=3)": t2,
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # --- Save results ---
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Convert to serializable format
    output = {
        "config": {
            "n_individuals": cfg.estimation.n_individuals,
            "n_periods": cfg.estimation.n_periods,
            "n_states": cfg.dgp.num_states,
            "n_actions": cfg.dgp.num_actions,
            "discount_factor": cfg.dgp.discount_factor,
        },
        "phase1": df_phase1.to_dict(orient="records"),
        "type_i": df_type_i.to_dict(orient="records"),
        "type_ii": type_ii_results,
        "summary": df_summary.to_dict(orient="records"),
    }

    with open(results_dir / "main_results.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    df_summary.to_csv(results_dir / "summary_table.csv", index=False)
    print(f"\nResults saved to {results_dir}/")

    return output


def main():
    parser = argparse.ArgumentParser(description="Identification experiment")
    parser.add_argument("--n_individuals", type=int, default=500)
    parser.add_argument("--n_periods", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--airl_max_rounds", type=int, default=300)
    parser.add_argument("--iq_max_iter", type=int, default=5000)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        estimation=EstimationConfig(
            n_individuals=args.n_individuals,
            n_periods=args.n_periods,
            seed=args.seed,
            airl_max_rounds=args.airl_max_rounds,
            iq_max_iter=args.iq_max_iter,
        ),
    )

    run_analytical_verification(cfg)
    print("\n")
    run_experiment(cfg)


if __name__ == "__main__":
    main()
