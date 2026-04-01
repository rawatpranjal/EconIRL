"""RDW Vehicle Scrappage: Complete Post-Estimation Showcase

Demonstrates the full econirl pipeline on real Dutch vehicle scrappage data
from the RDW open data API. Three estimators span structural econometrics
(NFXP, CCP) and neural inverse reinforcement learning (GLADIUS). All three
produce full inference, validation, and counterfactual analysis through
the same unified pipeline.

Counterfactual analyses include:
    - Scrappage subsidy: what if the government subsidizes replacement cost?
    - Defect deterioration: what if climate change worsens road conditions?
    - Elasticity: sensitivity of scrappage rates to replacement cost
    - Welfare: decomposition into direct effect and distribution effect

Sections:
    1. Load real RDW data, estimate transitions, train/test split
    2. Estimation (NFXP, CCP, GLADIUS)
    3. Inference and diagnostics (SEs, CIs, Wald test, identification)
    4. Validation (in-sample, out-of-sample)
    5. Counterfactual simulation (subsidy, deterioration, elasticity, welfare)
    6. Grand summary table

Run:
    python examples/rdw-scrappage/rdw_showcase.py
    python examples/rdw-scrappage/rdw_showcase.py --data-dir /path/to/rdw/
    python examples/rdw-scrappage/rdw_showcase.py --max-vehicles 5000
"""

import argparse
import time

import econirl._jax_config  # enable float64 before any JAX ops
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.datasets import load_rdw_scrappage
from econirl.environments.rdw_scrappage import (
    RDWScrapageEnvironment,
    state_to_components,
    components_to_state,
)
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.inference.identification import diagnose_identification_issues
from econirl.preferences.linear import LinearUtility
from econirl.simulation.counterfactual import (
    counterfactual_policy,
    counterfactual_transitions,
    elasticity_analysis,
    simulate_counterfactual,
    compute_welfare_effect,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_panel(panel: Panel, train_frac: float = 0.8) -> tuple[Panel, Panel]:
    """Split a panel into train and test sets by individual."""
    cutoff = int(panel.num_individuals * train_frac)
    train = Panel(trajectories=panel.trajectories[:cutoff])
    test = Panel(trajectories=panel.trajectories[cutoff:])
    return train, test


def estimate_rdw_transitions(panel: Panel, num_states: int) -> jnp.ndarray:
    """Estimate transition matrices by counting (s, a=0, s') pairs.

    For the 2D (age x defect) state space, the standard 1D increment
    estimator does not apply. Instead we count state-to-state transitions
    from keep-action observations and normalize.
    """
    keep_counts = np.zeros((num_states, num_states), dtype=np.float64)

    for traj in panel.trajectories:
        states = np.array(traj.states)
        actions = np.array(traj.actions)
        for t in range(len(states) - 1):
            if int(actions[t]) == 0:
                s, s_next = int(states[t]), int(states[t + 1])
                if 0 <= s < num_states and 0 <= s_next < num_states:
                    keep_counts[s, s_next] += 1

    row_sums = keep_counts.sum(axis=1, keepdims=True)
    keep_trans = np.where(
        row_sums > 0,
        keep_counts / np.maximum(row_sums, 1),
        np.ones((num_states, num_states)) / num_states,
    )

    transitions = np.zeros((2, num_states, num_states), dtype=np.float64)
    transitions[0] = keep_trans
    transitions[1, :, 0] = 1.0  # scrap resets to state 0

    return jnp.array(transitions, dtype=jnp.float32)


def evaluate_policy(result, panel, utility, problem, transitions):
    """Compute log-likelihood and accuracy of estimated policy on a panel."""
    operator = SoftBellmanOperator(problem, transitions)
    flow_u = utility.compute(result.parameters)
    sol = value_iteration(operator, flow_u, tol=1e-12, max_iter=100_000)
    log_probs = operator.compute_log_choice_probabilities(flow_u, sol.V)
    states = panel.get_all_states()
    actions = panel.get_all_actions()
    ll = log_probs[states, actions].sum()
    predicted = sol.policy[states].argmax(axis=1)
    acc = (predicted == actions).astype(jnp.float32).mean()
    return float(ll), float(acc)


def print_header(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")


def print_subheader(title: str) -> None:
    print(f"\n--- {title} ---\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RDW Vehicle Scrappage: Full Post-Estimation Showcase"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-vehicles", type=int, default=5000)
    args = parser.parse_args()

    # ===================================================================
    # Section 1: Load Real Data
    # ===================================================================

    print_header("SECTION 1: LOAD RDW DATA")

    df = load_rdw_scrappage(data_dir=args.data_dir, max_vehicles=args.max_vehicles)
    panel = load_rdw_scrappage(
        data_dir=args.data_dir, as_panel=True, max_vehicles=args.max_vehicles,
    )

    n_obs = len(df)
    n_veh = df["vehicle_id"].nunique()
    data_source = "real" if args.data_dir else "synthetic/real"

    print(f"Loaded RDW data: {n_obs:,} observations, {n_veh:,} vehicles ({data_source})")
    print(f"Scrappage rate: {df['scrapped'].mean():.2%}")
    print(f"Age range: [{df['age_bin'].min()}, {df['age_bin'].max()}]")
    print(f"Defect levels: pass={( df['defect_level']==0).mean():.1%}, "
          f"minor={(df['defect_level']==1).mean():.1%}, "
          f"major={(df['defect_level']==2).mean():.1%}")

    train_panel, test_panel = split_panel(panel, train_frac=0.8)
    print(f"\nTrain: {train_panel.num_observations:,} obs "
          f"({train_panel.num_individuals} vehicles)")
    print(f"Test:  {test_panel.num_observations:,} obs "
          f"({test_panel.num_individuals} vehicles)")

    # Environment and utility
    env = RDWScrapageEnvironment(discount_factor=0.95)
    utility = LinearUtility.from_environment(env)
    problem = DDCProblem(
        num_states=env.num_states,
        num_actions=2,
        discount_factor=0.95,
        scale_parameter=1.0,
    )

    # Estimate transitions from training data
    transitions = estimate_rdw_transitions(train_panel, env.num_states)
    n_empty = int((transitions[0].sum(axis=1) < 0.99).sum())
    print(f"\nProblem: {problem.num_states} states (25 age x 3 defect), "
          f"2 actions, beta={problem.discount_factor}")
    print(f"Transition estimation: {n_empty} states with no keep-action data "
          f"(uniform fallback)")

    # ===================================================================
    # Section 2: Estimation (NFXP, CCP, GLADIUS)
    # ===================================================================

    print_header("SECTION 2: ESTIMATION (NFXP, CCP, GLADIUS)")

    estimators = {
        "NFXP": NFXPEstimator(
            optimizer="BHHH",
            inner_solver="policy",
            inner_tol=1e-12,
            inner_max_iter=200,
            compute_hessian=True,
            se_method="asymptotic",
            outer_tol=1e-3,
        ),
        "CCP": CCPEstimator(
            num_policy_iterations=20,
            compute_hessian=True,
            se_method="asymptotic",
        ),
        "GLADIUS": GLADIUSEstimator(
            config=GLADIUSConfig(
                q_hidden_dim=64,
                q_num_layers=2,
                v_hidden_dim=64,
                v_num_layers=2,
                max_epochs=300,
                batch_size=1024,
                bellman_penalty_weight=1.0,
                compute_se=True,
                n_bootstrap=50,
                verbose=True,
            ),
        ),
    }

    results = {}
    timings = {}
    for name, estimator in estimators.items():
        print(f"\nFitting {name}...", end=" ", flush=True)
        t0 = time.time()
        results[name] = estimator.estimate(
            panel=train_panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )
        timings[name] = time.time() - t0
        r = results[name]
        print(f"done in {timings[name]:.1f}s "
              f"(converged={r.converged}, iters={r.num_iterations})")

    # Parameter comparison table
    print_subheader("Parameter Estimates")
    pnames = results["NFXP"].parameter_names
    print(f"{'Parameter':<24} ", end="")
    for name in estimators:
        print(f"{name:>14}", end="")
    print()
    print("-" * 66)
    for i, pname in enumerate(pnames):
        print(f"{pname:<24} ", end="")
        for name in estimators:
            print(f"{results[name].parameters[i]:>14.6f}", end="")
        print()

    # ===================================================================
    # Section 3: Inference and Diagnostics
    # ===================================================================

    print_header("SECTION 3: INFERENCE AND DIAGNOSTICS")

    # 3A: Summary tables
    print_subheader("3A: Estimation Summaries")
    for name, r in results.items():
        print(f"\n{'~' * 40}")
        print(f"  {name} Summary")
        print(f"{'~' * 40}")
        print(r.summary())

    # 3B: Confidence intervals
    print_subheader("3B: 95% Confidence Intervals")
    print(f"{'Estimator':<10} {'Parameter':<24} {'Lower':>12} {'Estimate':>12} {'Upper':>12}")
    print("-" * 76)
    for name, r in results.items():
        lower, upper = r.confidence_interval(alpha=0.05)
        for i, pname in enumerate(r.parameter_names):
            print(f"{name:<10} {pname:<24} {lower[i]:>12.4f} "
                  f"{r.parameters[i]:>12.4f} {upper[i]:>12.4f}")

    # 3C: Wald test (H0: replacement_cost = 5)
    print_subheader("3C: Wald Test (H0: replacement_cost = 5)")
    R = jnp.zeros((1, len(pnames)), dtype=jnp.float32)
    R = R.at[0, pnames.index("replacement_cost")].set(1.0)
    r_val = jnp.array([5.0], dtype=jnp.float32)
    print(f"{'Estimator':<10} {'Wald Stat':>12} {'df':>6} {'p-value':>12} {'Reject 5%?':>12}")
    print("-" * 58)
    for name, r in results.items():
        try:
            wald = r.wald_test(R, r_val)
            reject = "Yes" if wald["p_value"] < 0.05 else "No"
            print(f"{name:<10} {wald['statistic']:>12.2f} {wald['df']:>6} "
                  f"{wald['p_value']:>12.4e} {reject:>12}")
        except (ValueError, RuntimeError) as e:
            print(f"{name:<10} {'N/A':>12}  (unavailable)")

    # 3D: Identification diagnostics
    print_subheader("3D: Identification Diagnostics")
    print(f"{'Estimator':<10} {'Cond. Number':>14} {'Min Eigenval':>14} "
          f"{'Rank':>6} {'Status':<24}")
    print("-" * 74)
    for name, r in results.items():
        if r.identification is not None:
            ident = r.identification
            print(f"{name:<10} {ident.hessian_condition_number:>14.1f} "
                  f"{ident.min_eigenvalue:>14.4e} {ident.rank:>6} "
                  f"{ident.status:<24}")
        else:
            print(f"{name:<10} {'N/A':>14} {'N/A':>14} {'N/A':>6} {'Not computed':<24}")

    if results["NFXP"].hessian is not None:
        messages = diagnose_identification_issues(
            results["NFXP"].hessian,
            results["NFXP"].parameter_names,
        )
        if messages:
            print(f"\nNFXP detailed diagnostics:")
            for msg in messages:
                print(f"  - {msg}")
        else:
            print(f"\nNFXP: No identification issues detected.")

    # ===================================================================
    # Section 4: Validation
    # ===================================================================

    print_header("SECTION 4: VALIDATION (IN-SAMPLE, OUT-OF-SAMPLE)")

    val_results = {}
    for name, r in results.items():
        in_ll, in_acc = evaluate_policy(r, train_panel, utility, problem, transitions)
        oos_ll, oos_acc = evaluate_policy(r, test_panel, utility, problem, transitions)
        val_results[name] = {
            "in_ll": in_ll, "in_acc": in_acc,
            "oos_ll": oos_ll, "oos_acc": oos_acc,
        }

    print(f"{'Metric':<24}", end="")
    for name in estimators:
        print(f"{name:>14}", end="")
    print()
    print("-" * 66)
    for metric, label in [
        ("in_ll", "In-sample LL"),
        ("oos_ll", "Out-of-sample LL"),
        ("in_acc", "In-sample Accuracy"),
        ("oos_acc", "OOS Accuracy"),
    ]:
        print(f"{label:<24}", end="")
        for name in estimators:
            v = val_results[name][metric]
            if "ll" in metric:
                print(f"{v:>14.1f}", end="")
            else:
                print(f"{v:>14.4f}", end="")
        print()

    # ===================================================================
    # Section 5: Counterfactual Simulation
    # ===================================================================

    print_header("SECTION 5: COUNTERFACTUAL SIMULATION")

    nfxp_result = results["NFXP"]
    n_d = env._num_defect_levels

    # 5A: Scrappage subsidy -- reduce replacement cost by 30%
    print_subheader("5A: Scrappage Subsidy (30% Reduction in Replacement Cost)")
    rc_idx = pnames.index("replacement_cost")
    new_params = nfxp_result.parameters.at[rc_idx].mul(0.7)
    cf_subsidy = counterfactual_policy(
        nfxp_result, new_params, utility, problem, transitions,
    )
    print(f"Replacement cost: {nfxp_result.parameters[rc_idx]:.4f} -> "
          f"{new_params[rc_idx]:.4f} (30% subsidy)")
    print(f"Welfare change (mean value): {cf_subsidy.welfare_change:+.4f}")
    print(f"Max absolute policy change:  "
          f"{float(jnp.abs(cf_subsidy.policy_change).max()):.4f}")

    print(f"\nScrappage probability at selected states:")
    print(f"{'State (age, defect)':<22} {'Baseline':>10} {'Subsidy':>10} {'Change':>10}")
    print("-" * 56)
    for age in [5, 10, 15, 20]:
        for defect in [0, 2]:
            s = components_to_state(age, defect, n_d)
            label = f"({age}, {'pass' if defect == 0 else 'major'})"
            bp = cf_subsidy.baseline_policy[s, 1]
            cp = cf_subsidy.counterfactual_policy[s, 1]
            ch = cf_subsidy.policy_change[s, 1]
            print(f"{label:<22} {bp:>10.4f} {cp:>10.4f} {ch:>+10.4f}")

    # 5B: Defect deterioration -- worse defect transitions
    print_subheader("5B: Defect Deterioration (Climate/Road Damage Scenario)")
    worse_env = RDWScrapageEnvironment(
        defect_age_sensitivity=0.04,  # double the sensitivity
        discount_factor=0.95,
    )
    cf_deterioration = counterfactual_transitions(
        nfxp_result,
        worse_env.transition_matrices,
        utility,
        problem,
        transitions,
    )
    print(f"Scenario: defect_age_sensitivity 0.02 -> 0.04 (doubled)")
    print(f"Welfare change (mean value): {cf_deterioration.welfare_change:+.4f}")
    print(f"Max absolute policy change:  "
          f"{float(jnp.abs(cf_deterioration.policy_change).max()):.4f}")

    print(f"\nScrappage probability shift under worse defects:")
    print(f"{'State (age, defect)':<22} {'Baseline':>10} {'Worse':>10} {'Change':>10}")
    print("-" * 56)
    for age in [5, 10, 15, 20]:
        for defect in [0, 2]:
            s = components_to_state(age, defect, n_d)
            label = f"({age}, {'pass' if defect == 0 else 'major'})"
            bp = cf_deterioration.baseline_policy[s, 1]
            cp = cf_deterioration.counterfactual_policy[s, 1]
            ch = cf_deterioration.policy_change[s, 1]
            print(f"{label:<22} {bp:>10.4f} {cp:>10.4f} {ch:>+10.4f}")

    # 5C: Elasticity of scrappage to replacement cost
    print_subheader("5C: Elasticity of Scrappage to Replacement Cost")
    ea = elasticity_analysis(
        nfxp_result, utility, problem, transitions,
        parameter_name="replacement_cost",
        pct_changes=[-0.50, -0.30, -0.10, 0.10, 0.30, 0.50],
    )
    print(f"{'% Change':>10} {'Avg Policy Change':>20} {'Welfare Change':>18}")
    print("-" * 52)
    for i, pct in enumerate(ea["pct_changes"]):
        print(f"{pct:>+10.0%} {ea['policy_changes'][i]:>20.6f} "
              f"{ea['welfare_changes'][i]:>+18.4f}")
    if "welfare_elasticity" in ea:
        print(f"\nEstimated welfare elasticity: {ea['welfare_elasticity']:.4f}")

    # 5D: Welfare decomposition
    print_subheader("5D: Welfare Decomposition (Subsidy Scenario)")
    welfare = compute_welfare_effect(cf_subsidy, transitions)
    print(f"{'Metric':<40} {'Value':>12}")
    print("-" * 56)
    print(f"{'Baseline expected value':<40} {welfare['baseline_expected_value']:>12.4f}")
    print(f"{'Counterfactual expected value':<40} {welfare['counterfactual_expected_value']:>12.4f}")
    print(f"{'Total welfare change':<40} {welfare['total_welfare_change']:>+12.4f}")
    print(f"{'Direct effect (value change)':<40} {welfare['welfare_change_fixed_distribution']:>+12.4f}")
    print(f"{'Distribution effect':<40} {welfare['distribution_effect']:>+12.4f}")

    # 5E: Simulate counterfactual outcomes
    print_subheader("5E: Simulated Outcomes (Baseline vs Subsidy)")
    sim = simulate_counterfactual(
        nfxp_result, cf_subsidy, problem, transitions,
        n_individuals=1000, n_periods=50, seed=42,
    )
    print(f"Simulated 1,000 vehicles x 50 years under each scenario.\n")
    print(f"{'Metric':<30} {'Baseline':>12} {'Subsidy':>12} {'Change':>12}")
    print("-" * 70)
    scrap_base = sim["baseline_action_frequencies"][1]
    scrap_cf = sim["counterfactual_action_frequencies"][1]
    print(f"{'Scrappage frequency':<30} {scrap_base:>12.4f} {scrap_cf:>12.4f} "
          f"{scrap_cf - scrap_base:>+12.4f}")
    print(f"{'Mean state index':<30} {sim['baseline_mean_state']:>12.1f} "
          f"{sim['counterfactual_mean_state']:>12.1f} "
          f"{sim['counterfactual_mean_state'] - sim['baseline_mean_state']:>+12.1f}")

    # ===================================================================
    # Section 6: Grand Summary Table
    # ===================================================================

    print_header("SECTION 6: GRAND SUMMARY")

    print(f"\n{'':34}", end="")
    for name in estimators:
        print(f"{name:>14}", end="")
    print()
    print("=" * 76)

    print("PARAMETER ESTIMATES")
    for i, pname in enumerate(pnames):
        print(f"  {pname:<32}", end="")
        for name in estimators:
            print(f"{results[name].parameters[i]:>14.4f}", end="")
        print()

    print()
    print("STANDARD ERRORS")
    for i, pname in enumerate(pnames):
        label = f"SE({pname})"
        print(f"  {label:<32}", end="")
        for name in estimators:
            se = results[name].standard_errors[i]
            if np.isnan(se):
                print(f"{'N/A':>14}", end="")
            else:
                print(f"{se:>14.4f}", end="")
        print()

    print()
    print("IDENTIFICATION")
    print(f"  {'Condition Number':<32}", end="")
    for name in estimators:
        if results[name].identification is not None:
            print(f"{results[name].identification.hessian_condition_number:>14.1f}", end="")
        else:
            print(f"{'N/A':>14}", end="")
    print()

    print()
    print("VALIDATION")
    for metric, label in [
        ("in_ll", "In-sample LL"),
        ("oos_ll", "Out-of-sample LL"),
        ("in_acc", "In-sample Accuracy"),
        ("oos_acc", "OOS Accuracy"),
    ]:
        print(f"  {label:<32}", end="")
        for name in estimators:
            v = val_results[name][metric]
            if "ll" in metric:
                print(f"{v:>14.1f}", end="")
            else:
                print(f"{v:>14.4f}", end="")
        print()

    print()
    print("PERFORMANCE")
    print(f"  {'Time (seconds)':<32}", end="")
    for name in estimators:
        print(f"{timings[name]:>14.1f}", end="")
    print()
    print(f"  {'Converged':<32}", end="")
    for name in estimators:
        print(f"{str(results[name].converged):>14}", end="")
    print()

    print("=" * 76)
    print(f"\nThree estimators on real Dutch vehicle scrappage data.")
    print(f"Structural (NFXP, CCP) and neural (GLADIUS) through the same pipeline:")
    print(f"inference, validation, and counterfactual simulation.")


if __name__ == "__main__":
    main()
