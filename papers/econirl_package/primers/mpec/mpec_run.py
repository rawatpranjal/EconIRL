#!/usr/bin/env python3
"""
MPEC Primer -- Companion simulation (Su & Judd 2012).

Core finding: at beta=0.9999, MPEC eliminates the inner Bellman loop entirely
by treating V as a decision variable and enforcing Bellman as a constraint.
All three estimators (NFXP-SA, MPEC-SQP, NFXP-NK) recover the same MLE, but
MPEC is 7-12x faster because it does zero inner iterations regardless of beta.

Usage:
    cd papers/econirl_package/primers/mpec
    python mpec_run.py
    pdflatex mpec.tex && bibtex mpec && pdflatex mpec.tex && pdflatex mpec.tex
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jax.numpy as jnp
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
from econirl.preferences.linear import LinearUtility
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration

TRUE_OC = 0.001
TRUE_RC = 3.0
N_BINS = 90
BETA = 0.9999
N_INDIVIDUALS = 500
N_PERIODS = 100
SEED = 42

# At beta=0.9999, SA needs log(1e-10)/log(0.9999) = 230,258 iterations.
# We cap at 500 to expose the failure mode.
SA_BUDGET = 500
SA_NEEDED = 230258  # theoretical iterations at tol=1e-10


def build_env():
    return RustBusEnvironment(
        num_mileage_bins=N_BINS,
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        discount_factor=BETA,
    )


def run_experiment(env):
    """Run NFXP-SA (budget), MPEC, and NFXP-NK on Rust bus at beta=0.9999."""
    panel = env.generate_panel(
        n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED
    )
    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    results = {
        "n_states": N_BINS,
        "beta": BETA,
        "n_obs": panel.num_observations,
        "sa_budget": SA_BUDGET,
        "sa_iterations_needed": SA_NEEDED,
    }

    # --- NFXP-SA (budget-limited, exposes failure at high beta) ---
    print(f"\n[1/3] NFXP-SA  (inner_max_iter={SA_BUDGET}, needs {SA_NEEDED:,})")
    t0 = time.time()
    nfxp_sa = NFXPEstimator(
        inner_solver="value",
        inner_max_iter=SA_BUDGET,
        inner_tol=1e-10,
    )
    sa_result = nfxp_sa.estimate(panel, utility, problem, transitions)
    sa_time = time.time() - t0
    results["nfxp_value"] = {
        "oc": float(sa_result.parameters[0]),
        "rc": float(sa_result.parameters[1]),
        "ll": float(sa_result.log_likelihood),
        "time_s": sa_time,
        "converged": bool(sa_result.converged),
    }
    print(
        f"  OC={sa_result.parameters[0]:.6f}  RC={sa_result.parameters[1]:.4f}  "
        f"LL={sa_result.log_likelihood:.2f}  t={sa_time:.1f}s  conv={sa_result.converged}"
    )

    # --- MPEC ---
    print("\n[2/3] MPEC  (SLSQP + JAX gradients, outer_max_iter=200)")
    t0 = time.time()
    mpec = MPECEstimator(
        MPECConfig(
            solver="sqp",
            outer_max_iter=200,
            tol=1e-8,
            constraint_tol=1e-6,
        )
    )
    mpec_result = mpec.estimate(panel, utility, problem, transitions)
    mpec_time = time.time() - t0
    cv = float(mpec_result.metadata.get("final_constraint_violation", float("nan")))
    results["mpec"] = {
        "oc": float(mpec_result.parameters[0]),
        "rc": float(mpec_result.parameters[1]),
        "ll": float(mpec_result.log_likelihood),
        "time_s": mpec_time,
        "converged": bool(mpec_result.converged),
        "constraint_violation": cv,
    }
    print(
        f"  OC={mpec_result.parameters[0]:.6f}  RC={mpec_result.parameters[1]:.4f}  "
        f"LL={mpec_result.log_likelihood:.2f}  t={mpec_time:.1f}s  "
        f"conv={mpec_result.converged}  CV={cv:.1e}"
    )

    # --- NFXP-NK (reference: Iskhakov et al. 2016) ---
    print("\n[3/3] NFXP-NK  (hybrid SA->NK, reference)")
    t0 = time.time()
    nfxp_nk = NFXPEstimator(inner_solver="hybrid")
    nk_result = nfxp_nk.estimate(panel, utility, problem, transitions)
    nk_time = time.time() - t0
    results["nfxp_nk"] = {
        "oc": float(nk_result.parameters[0]),
        "rc": float(nk_result.parameters[1]),
        "ll": float(nk_result.log_likelihood),
        "time_s": nk_time,
        "converged": bool(nk_result.converged),
    }
    print(
        f"  OC={nk_result.parameters[0]:.6f}  RC={nk_result.parameters[1]:.4f}  "
        f"LL={nk_result.log_likelihood:.2f}  t={nk_time:.1f}s  conv={nk_result.converged}"
    )

    return results


def run_bc_baseline(env):
    """Compute BC baseline at the same beta."""
    panel = env.generate_panel(n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )
    operator = SoftBellmanOperator(
        env.problem_spec,
        jnp.asarray(env.transition_matrices, dtype=jnp.float64),
    )
    true_reward = jnp.asarray(
        utility.compute(jnp.array([TRUE_OC, TRUE_RC], dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = policy_iteration(
        operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_actions = np.argmax(np.array(true_result.policy), axis=1)
    bc = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    bc_res = bc.estimate(
        panel, utility, env.problem_spec, jnp.asarray(env.transition_matrices)
    )
    bc_actions = np.argmax(np.array(bc_res.policy), axis=1)
    return {
        "ll": float(bc_res.log_likelihood),
        "policy_acc": float(np.mean(bc_actions == true_actions)),
    }


OUT_DIR = Path(__file__).resolve().parent


def write_results_tex(results, bc_result):
    """Write LaTeX macros for auto-inclusion in mpec.tex."""
    path = OUT_DIR / "mpec_results.tex"
    n = results["nfxp_value"]
    m = results["mpec"]
    k = results["nfxp_nk"]

    speedup_sa = n["time_s"] / m["time_s"] if m["time_s"] > 0 else 0
    speedup_nk = k["time_s"] / m["time_s"] if m["time_s"] > 0 else 0
    ll_gap = abs(m["ll"] - k["ll"])

    lines = [
        "% Auto-generated by mpec_run.py -- do not edit by hand\n",
        f"\\newcommand{{\\mpecNFXPTime}}{{{n['time_s']:.0f}s}}\n",
        f"\\newcommand{{\\mpecMPECTime}}{{{m['time_s']:.1f}s}}\n",
        f"\\newcommand{{\\mpecNKTime}}{{{k['time_s']:.1f}s}}\n",
        f"\\newcommand{{\\mpecSpeedup}}{{{speedup_sa:.0f}x}}\n",
        f"\\newcommand{{\\mpecNKSpeedup}}{{{speedup_nk:.0f}x}}\n",
        f"\\newcommand{{\\mpecOC}}{{{m['oc']:.6f}}}\n",
        f"\\newcommand{{\\mpecRC}}{{{m['rc']:.4f}}}\n",
        f"\\newcommand{{\\mpecNKOC}}{{{k['oc']:.6f}}}\n",
        f"\\newcommand{{\\mpecNKRC}}{{{k['rc']:.4f}}}\n",
        f"\\newcommand{{\\mpecSAOC}}{{{n['oc']:.6f}}}\n",
        f"\\newcommand{{\\mpecSARC}}{{{n['rc']:.4f}}}\n",
        f"\\newcommand{{\\mpecLLgap}}{{{ll_gap:.2f}}}\n",
        f"\\newcommand{{\\mpecCV}}{{{m['constraint_violation']:.1e}}}\n",
        f"\\newcommand{{\\mpecSAConv}}{{{str(n['converged']).lower()}}}\n",
        f"\\newcommand{{\\mpecMPECConv}}{{{str(m['converged']).lower()}}}\n",
        f"\\newcommand{{\\mpecBCacc}}{{{bc_result['policy_acc'] * 100:.1f}\\%}}\n",
        f"\\newcommand{{\\mpecBCLL}}{{{bc_result['ll']:.2f}}}\n",
        f"\\newcommand{{\\mpecNObs}}{{{results['n_obs']:,}}}\n",
        f"\\newcommand{{\\mpecNStates}}{{{results['n_states']}}}\n",
        f"\\newcommand{{\\mpecBetaVal}}{{{results['beta']}}}\n",
        f"\\newcommand{{\\mpecSABudget}}{{{results['sa_budget']:,}}}\n",
        f"\\newcommand{{\\mpecSANeeded}}{{{results['sa_iterations_needed']:,}}}\n",
    ]
    path.write_text("".join(lines))
    print(f"\nMacros written to {path}")


def main():
    print("=" * 70)
    print(f"MPEC Primer: Rust Bus Engine ({N_BINS} states, beta={BETA})")
    print(f"True: OC={TRUE_OC}, RC={TRUE_RC}")
    print(f"SA budget: {SA_BUDGET:,} / {SA_NEEDED:,} iterations needed at tol=1e-10")
    print("=" * 70)

    env = build_env()

    print("\nComputing BC baseline...")
    bc_result = run_bc_baseline(env)
    print(f"  BC: LL={bc_result['ll']:.2f}, policy_acc={bc_result['policy_acc']:.2%}")

    results = run_experiment(env)
    results["bc"] = bc_result

    n, m, k = results["nfxp_value"], results["mpec"], results["nfxp_nk"]
    speedup = n["time_s"] / m["time_s"] if m["time_s"] > 0 else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"True params:       OC={TRUE_OC:.6f}  RC={TRUE_RC:.4f}")
    print(
        f"NFXP-SA ({SA_BUDGET}it):  OC={n['oc']:.6f}  RC={n['rc']:.4f}  "
        f"LL={n['ll']:.2f}  t={n['time_s']:.1f}s  conv={n['converged']}"
    )
    print(
        f"MPEC:              OC={m['oc']:.6f}  RC={m['rc']:.4f}  "
        f"LL={m['ll']:.2f}  t={m['time_s']:.1f}s  conv={m['converged']}  "
        f"CV={m['constraint_violation']:.1e}"
    )
    print(
        f"NFXP-NK:           OC={k['oc']:.6f}  RC={k['rc']:.4f}  "
        f"LL={k['ll']:.2f}  t={k['time_s']:.1f}s  conv={k['converged']}"
    )
    print(f"MPEC speedup over NFXP-SA: {speedup:.1f}x")

    write_results_tex(results, bc_result)
    (OUT_DIR / "mpec_results.json").write_text(json.dumps(results, indent=2))
    print(f"JSON saved to {OUT_DIR / 'mpec_results.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
