"""Rust Bus Engine: Complete Post-Estimation Showcase

Demonstrates every post-estimation capability in econirl on the real
Rust (1987) bus engine replacement data using three estimators spanning
structural econometrics and inverse reinforcement learning: NFXP
(nested fixed point), CCP (conditional choice probability), and MCE-IRL
(maximum causal entropy inverse reinforcement learning). All three produce
full inference, validation, and counterfactual analysis through the same
unified pipeline.

Sections:
    1. Load real data, estimate transitions, train/test split
    2. Estimation (3 estimators on training data)
    3. Inference and diagnostics (SEs, CIs, Wald test, identification)
    4. Validation (in-sample, out-of-sample, transfer)
    5. Counterfactual simulation (parameter change, transition change, elasticity)
    6. Grand summary table

Run:
    python examples/rust-bus-engine/showcase.py
"""

import time

import numpy as np
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator
from econirl.estimation.nfxp import NFXPEstimator, estimate_transitions_from_panel
from econirl.inference.identification import diagnose_identification_issues
from econirl.preferences.linear import LinearUtility
from econirl.simulation.counterfactual import (
    counterfactual_policy,
    counterfactual_transitions,
    elasticity_analysis,
    simulate_counterfactual,
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


def evaluate_policy(result, panel, utility, problem, transitions):
    """Compute log-likelihood and accuracy of estimated policy on any panel.

    This is the generic out-of-sample evaluator. It re-solves the Bellman
    equation with the estimated parameters and scores the observed choices
    against the implied choice probabilities.
    """
    operator = SoftBellmanOperator(problem, transitions)
    flow_u = utility.compute(result.parameters)
    sol = value_iteration(operator, flow_u, tol=1e-12, max_iter=100_000)
    log_probs = operator.compute_log_choice_probabilities(flow_u, sol.V)
    states = panel.get_all_states()
    actions = panel.get_all_actions()
    ll = log_probs[states, actions].sum().item()
    predicted = sol.policy[states].argmax(dim=1)
    acc = (predicted == actions).float().mean().item()
    return ll, acc


def print_header(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")


def print_subheader(title: str) -> None:
    print(f"\n--- {title} ---\n")


# ---------------------------------------------------------------------------
# Section 1: Load Real Data
# ---------------------------------------------------------------------------

def main():
    print_header("SECTION 1: LOAD REAL DATA")

    # Load the original Rust (1987) bus engine data
    df = load_rust_bus(original=True)
    print(f"Loaded Rust (1987) data: {len(df):,} observations, "
          f"{df['bus_id'].nunique()} buses, {df['group'].nunique()} groups")
    print(f"Replacement rate: {df['replaced'].mean():.2%}")
    print(f"Mileage range: [{df['mileage_bin'].min()}, {df['mileage_bin'].max()}]")

    # Convert to Panel
    panel = load_rust_bus(original=True, as_panel=True)
    train_panel, test_panel = split_panel(panel, train_frac=0.8)
    print(f"\nTrain: {train_panel.num_observations:,} obs "
          f"({train_panel.num_individuals} buses)")
    print(f"Test:  {test_panel.num_observations:,} obs "
          f"({test_panel.num_individuals} buses)")

    # Estimate transitions from training data (cast to float32 for consistency
    # with utility computation and counterfactual functions)
    transitions = estimate_transitions_from_panel(
        train_panel, num_states=90, max_increment=2
    ).float()

    # Count mileage increments for display
    counts = np.zeros(3)
    for traj in train_panel.trajectories:
        for t in range(len(traj.states) - 1):
            if traj.actions[t].item() == 0:
                inc = traj.states[t + 1].item() - traj.states[t].item()
                if 0 <= inc <= 2:
                    counts[int(inc)] += 1
    probs = counts / counts.sum()
    print(f"\nEstimated transition probs: P(+0)={probs[0]:.4f}, "
          f"P(+1)={probs[1]:.4f}, P(+2)={probs[2]:.4f}")

    # Build utility and problem from the RustBusEnvironment template
    # (just for the feature matrix and problem spec, not for data)
    env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
    utility = LinearUtility.from_environment(env)
    problem = DDCProblem(
        num_states=90,
        num_actions=2,
        discount_factor=0.9999,
        scale_parameter=1.0,
    )
    print(f"Problem: {problem.num_states} states, {problem.num_actions} actions, "
          f"beta={problem.discount_factor}")

    # Transfer environment: faster bus wear (counterfactual dynamics)
    transfer_env = RustBusEnvironment(
        mileage_transition_probs=(0.5, 0.4, 0.1),
        discount_factor=0.9999,
    )
    transfer_transitions = transfer_env.transition_matrices
    print(f"Transfer env: mileage_transition_probs=(0.5, 0.4, 0.1) [faster wear]")

    # -----------------------------------------------------------------------
    # Section 2: Estimation
    # -----------------------------------------------------------------------

    print_header("SECTION 2: ESTIMATION (NFXP, CCP, MCE-IRL)")

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
        "MCE-IRL": MCEIRLEstimator(
            compute_se=True,
            se_method="hessian",
            inner_solver="hybrid",
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
    print(f"{'Parameter':<18} ", end="")
    for name in estimators:
        print(f"{name:>12}", end="")
    print()
    print("-" * 54)

    for i, pname in enumerate(results["NFXP"].parameter_names):
        print(f"{pname:<18} ", end="")
        for name in estimators:
            print(f"{results[name].parameters[i].item():>12.6f}", end="")
        print()

    # -----------------------------------------------------------------------
    # Section 3: Inference and Diagnostics
    # -----------------------------------------------------------------------

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
    print(f"{'Estimator':<10} {'Parameter':<18} {'Lower':>12} {'Estimate':>12} {'Upper':>12}")
    print("-" * 70)
    for name, r in results.items():
        lower, upper = r.confidence_interval(alpha=0.05)
        for i, pname in enumerate(r.parameter_names):
            print(f"{name:<10} {pname:<18} {lower[i].item():>12.6f} "
                  f"{r.parameters[i].item():>12.6f} {upper[i].item():>12.6f}")

    # 3C: Wald test (H0: replacement_cost = 10)
    print_subheader("3C: Wald Test (H0: replacement_cost = 10)")
    R = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    r_val = torch.tensor([10.0], dtype=torch.float32)
    print(f"{'Estimator':<10} {'Wald Stat':>12} {'df':>6} {'p-value':>12} {'Reject?':>10}")
    print("-" * 56)
    for name, r in results.items():
        try:
            wald = r.wald_test(R, r_val)
            reject = "Yes" if wald["p_value"] < 0.05 else "No"
            print(f"{name:<10} {wald['statistic']:>12.2f} {wald['df']:>6} "
                  f"{wald['p_value']:>12.4e} {reject:>10}")
        except ValueError as e:
            print(f"{name:<10} {'N/A':>12}  (variance-covariance unavailable: {e})")

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

    # Detailed diagnostics for NFXP
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

    # 3E: SE method comparison (NFXP only)
    print_subheader("3E: SE Method Comparison (NFXP only)")
    print("Running NFXP with robust (sandwich) standard errors...")
    nfxp_robust = NFXPEstimator(
        optimizer="BHHH",
        inner_solver="policy",
        inner_tol=1e-12,
        inner_max_iter=200,
        compute_hessian=True,
        se_method="robust",
        outer_tol=1e-3,
    )
    result_robust = nfxp_robust.estimate(
        panel=train_panel,
        utility=utility,
        problem=problem,
        transitions=transitions,
    )
    print(f"\n{'Parameter':<18} {'Asymptotic SE':>16} {'Robust SE':>16} {'Ratio':>10}")
    print("-" * 66)
    for i, pname in enumerate(results["NFXP"].parameter_names):
        se_asym = results["NFXP"].standard_errors[i].item()
        se_robust = result_robust.standard_errors[i].item()
        ratio = se_robust / se_asym if se_asym > 0 else float("nan")
        print(f"{pname:<18} {se_asym:>16.6f} {se_robust:>16.6f} {ratio:>10.4f}")
    print("\nNote: CCP and MCE-IRL do not produce per-observation gradient")
    print("contributions, so robust (sandwich) SEs are only available for NFXP.")

    # -----------------------------------------------------------------------
    # Section 4: Validation
    # -----------------------------------------------------------------------

    print_header("SECTION 4: VALIDATION (IN-SAMPLE, OUT-OF-SAMPLE, TRANSFER)")

    # 4A/4B: In-sample and out-of-sample
    print_subheader("4A-B: Log-Likelihood and Accuracy")

    val_results = {}
    for name, r in results.items():
        in_ll, in_acc = evaluate_policy(r, train_panel, utility, problem, transitions)
        oos_ll, oos_acc = evaluate_policy(r, test_panel, utility, problem, transitions)
        val_results[name] = {
            "in_ll": in_ll,
            "in_acc": in_acc,
            "oos_ll": oos_ll,
            "oos_acc": oos_acc,
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
            if "ll" in metric.lower():
                print(f"{v:>14.1f}", end="")
            else:
                print(f"{v:>14.4f}", end="")
        print()

    # -----------------------------------------------------------------------
    # Section 5: Counterfactual Simulation
    # -----------------------------------------------------------------------

    print_header("SECTION 5: COUNTERFACTUAL SIMULATION")

    nfxp_result = results["NFXP"]

    # 5A: Parameter counterfactual -- double replacement cost
    print_subheader("5A: Double Replacement Cost")
    new_params = nfxp_result.parameters.clone()
    new_params[1] = new_params[1] * 2  # double replacement_cost
    cf = counterfactual_policy(nfxp_result, new_params, utility, problem, transitions)
    print(f"Replacement cost: {nfxp_result.parameters[1].item():.4f} -> "
          f"{new_params[1].item():.4f}")
    print(f"Welfare change (mean value): {cf.welfare_change:+.4f}")
    print(f"Max absolute policy change:  {cf.policy_change.abs().max().item():.4f}")
    print(f"\nReplacement probability at selected mileage states:")
    print(f"{'Mileage Bin':<14} {'Baseline':>12} {'Counterfactual':>16} {'Change':>12}")
    print("-" * 58)
    for s in [0, 20, 40, 60, 80, 89]:
        base_p = cf.baseline_policy[s, 1].item()
        cf_p = cf.counterfactual_policy[s, 1].item()
        change = cf.policy_change[s, 1].item()
        print(f"{s:<14} {base_p:>12.6f} {cf_p:>16.6f} {change:>+12.6f}")

    # 5B: Transition counterfactual -- faster bus wear
    print_subheader("5B: Transition Counterfactual (Faster Wear)")
    cf_trans = counterfactual_transitions(
        nfxp_result, transfer_transitions, utility, problem, transitions,
    )
    print(f"Scenario: estimated transition probs -> (0.50, 0.40, 0.10) [faster wear]")
    print(f"Welfare change (mean value): {cf_trans.welfare_change:+.4f}")
    print(f"Max absolute policy change:  {cf_trans.policy_change.abs().max().item():.4f}")
    print(f"\nReplacement probability shift under faster wear:")
    print(f"{'Mileage Bin':<14} {'Baseline':>12} {'Faster Wear':>14} {'Change':>12}")
    print("-" * 56)
    for s in [0, 20, 40, 60, 80, 89]:
        base_p = cf_trans.baseline_policy[s, 1].item()
        cf_p = cf_trans.counterfactual_policy[s, 1].item()
        change = cf_trans.policy_change[s, 1].item()
        print(f"{s:<14} {base_p:>12.6f} {cf_p:>14.6f} {change:>+12.6f}")

    # 5C: Elasticity analysis
    print_subheader("5C: Elasticity of Policy to Replacement Cost")
    ea = elasticity_analysis(
        nfxp_result, utility, problem, transitions,
        parameter_name=utility.parameter_names[1],
        pct_changes=[-0.50, -0.25, -0.10, 0.10, 0.25, 0.50, 1.00],
    )
    print(f"{'% Change':>10} {'Avg Policy Change':>20} {'Welfare Change':>18}")
    print("-" * 52)
    for i, pct in enumerate(ea["pct_changes"]):
        print(f"{pct:>+10.0%} {ea['policy_changes'][i]:>20.6f} "
              f"{ea['welfare_changes'][i]:>+18.4f}")
    if "welfare_elasticity" in ea:
        print(f"\nEstimated welfare elasticity: {ea['welfare_elasticity']:.4f}")

    # 5D: Simulate counterfactual outcomes
    print_subheader("5D: Simulated Outcomes (Baseline vs Double RC)")
    sim = simulate_counterfactual(
        nfxp_result, cf, problem, transitions,
        n_individuals=1000, n_periods=100, seed=123,
    )
    print(f"Simulated 1,000 individuals x 100 periods under each scenario.\n")
    print(f"{'Metric':<30} {'Baseline':>12} {'Double RC':>12} {'Change':>12}")
    print("-" * 70)
    repl_base = sim["baseline_action_frequencies"][1].item()
    repl_cf = sim["counterfactual_action_frequencies"][1].item()
    print(f"{'Replacement frequency':<30} {repl_base:>12.4f} {repl_cf:>12.4f} "
          f"{repl_cf - repl_base:>+12.4f}")
    print(f"{'Mean mileage bin':<30} {sim['baseline_mean_state']:>12.1f} "
          f"{sim['counterfactual_mean_state']:>12.1f} "
          f"{sim['counterfactual_mean_state'] - sim['baseline_mean_state']:>+12.1f}")

    # -----------------------------------------------------------------------
    # Section 6: Grand Summary Table
    # -----------------------------------------------------------------------

    print_header("SECTION 6: GRAND SUMMARY")

    print(f"\n{'':30}", end="")
    for name in estimators:
        print(f"{name:>14}", end="")
    print()
    print("=" * 72)

    # Parameter estimates
    print("PARAMETER ESTIMATES")
    for i, pname in enumerate(results["NFXP"].parameter_names):
        print(f"  {pname:<28}", end="")
        for name in estimators:
            print(f"{results[name].parameters[i].item():>14.6f}", end="")
        print()

    print()
    print("STANDARD ERRORS")
    for i, pname in enumerate(results["NFXP"].parameter_names):
        print(f"  SE({pname}){'':<{20 - len(pname)}}", end="")
        for name in estimators:
            se = results[name].standard_errors[i].item()
            if np.isnan(se):
                print(f"{'N/A':>14}", end="")
            else:
                print(f"{se:>14.6f}", end="")
        print()

    print()
    print("IDENTIFICATION")
    print(f"  {'Condition Number':<28}", end="")
    for name in estimators:
        if results[name].identification is not None:
            print(f"{results[name].identification.hessian_condition_number:>14.1f}", end="")
        else:
            print(f"{'N/A':>14}", end="")
    print()
    print(f"  {'Wald(RC=10) p-value':<28}", end="")
    for name in estimators:
        try:
            wald = results[name].wald_test(R, r_val)
            print(f"{wald['p_value']:>14.2e}", end="")
        except (ValueError, RuntimeError):
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
        print(f"  {label:<28}", end="")
        for name in estimators:
            v = val_results[name][metric]
            if "ll" in metric:
                print(f"{v:>14.1f}", end="")
            else:
                print(f"{v:>14.4f}", end="")
        print()

    print()
    print("PERFORMANCE")
    print(f"  {'Time (seconds)':<28}", end="")
    for name in estimators:
        print(f"{timings[name]:>14.1f}", end="")
    print()
    print(f"  {'Converged':<28}", end="")
    for name in estimators:
        print(f"{str(results[name].converged):>14}", end="")
    print()

    print("=" * 72)
    print("\nThree estimators, one dataset, identical diagnostics.")
    print("Real Rust (1987) bus engine data through the same unified pipeline:")
    print("inference, validation, and counterfactual simulation.")


if __name__ == "__main__":
    main()
