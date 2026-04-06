"""SEES primer toy example: parameter recovery on the Rust bus engine.

Runs SEES (Fourier basis, K=8) and NFXP (oracle) on a simulated panel
from the Rust (1987) bus replacement model. Writes results to
sees_results.tex as \\newcommand definitions so the primer PDF
auto-updates when this script is rerun.

Usage:
    python sees_toy.py
"""

import json
import os
import time

import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.sees import SEESConfig, SEESEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

# ---------- DGP settings (Rust 1987) ----------
TRUE_OC = 0.001          # operating cost per mileage bin
TRUE_RC = 3.0            # replacement cost
N_STATES = 200           # mileage bins
DISCOUNT = 0.9999        # beta (high to amplify operating cost signal)
N_BUSES = 200            # number of individuals
N_PERIODS = 100          # periods per bus
SEED = 42

# ---------- SEES settings (Luo & Sang 2024) ----------
BASIS_TYPE = "fourier"
BASIS_DIM = 8
PENALTY_WEIGHT = 10.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # --- Simulate data ---
    env = RustBusEnvironment(
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        num_mileage_bins=N_STATES,
        discount_factor=DISCOUNT,
        seed=SEED,
    )
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)

    panel = simulate_panel(
        env,
        n_individuals=N_BUSES,
        n_periods=N_PERIODS,
        seed=SEED + 1000,
    )

    true_params = np.array([TRUE_OC, TRUE_RC])
    results = {}

    # --- NFXP oracle ---
    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-12,
        inner_max_iter=100000,
        switch_tol=1e-3,
        outer_max_iter=200,
        compute_hessian=True,
        verbose=False,
    )
    t0 = time.time()
    nfxp_res = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    nfxp_params = np.asarray(nfxp_res.parameters)
    nfxp_se = np.asarray(nfxp_res.standard_errors)

    results["nfxp"] = {
        "theta_c": round(float(nfxp_params[0]), 8),
        "RC": round(float(nfxp_params[1]), 6),
        "se_theta_c": round(float(nfxp_se[0]), 8),
        "se_RC": round(float(nfxp_se[1]), 6),
        "ll": round(float(nfxp_res.log_likelihood), 2),
        "time_s": round(nfxp_time, 2),
    }
    print(f"NFXP:  theta_c={nfxp_params[0]:.6f}  RC={nfxp_params[1]:.4f}  "
          f"LL={nfxp_res.log_likelihood:.1f}  time={nfxp_time:.1f}s")

    # --- SEES ---
    sees_cfg = SEESConfig(
        basis_type=BASIS_TYPE,
        basis_dim=BASIS_DIM,
        penalty_weight=PENALTY_WEIGHT,
        max_iter=500,
        compute_se=True,
        se_method="asymptotic",
        verbose=False,
    )
    sees_est = SEESEstimator(config=sees_cfg)

    t0 = time.time()
    sees_res = sees_est.estimate(panel, utility, problem, transitions)
    sees_time = time.time() - t0
    sees_params = np.asarray(sees_res.parameters)
    sees_se = np.asarray(sees_res.standard_errors)

    # Bellman violation at solution
    bellman_viol = 0.0
    if sees_res.metadata and "bellman_violation" in sees_res.metadata:
        bellman_viol = float(sees_res.metadata["bellman_violation"])

    results["sees"] = {
        "theta_c": round(float(sees_params[0]), 8),
        "RC": round(float(sees_params[1]), 6),
        "se_theta_c": round(float(sees_se[0]), 8),
        "se_RC": round(float(sees_se[1]), 6),
        "ll": round(float(sees_res.log_likelihood), 2),
        "time_s": round(sees_time, 2),
        "bellman_violation": round(float(bellman_viol), 6),
        "basis_type": BASIS_TYPE,
        "basis_dim": BASIS_DIM,
    }
    print(f"SEES:  theta_c={sees_params[0]:.6f}  RC={sees_params[1]:.4f}  "
          f"LL={sees_res.log_likelihood:.1f}  time={sees_time:.1f}s")

    # --- Relative errors ---
    oc_pct = float(abs(sees_params[0] - TRUE_OC) / TRUE_OC * 100)
    rc_pct = float(abs(sees_params[1] - TRUE_RC) / TRUE_RC * 100)
    results["relative_error_pct"] = {
        "theta_c": round(oc_pct, 2),
        "RC": round(rc_pct, 2),
    }
    print(f"Relative error:  theta_c={oc_pct:.1f}%  RC={rc_pct:.1f}%")

    # --- DGP metadata ---
    results["dgp"] = {
        "true_theta_c": TRUE_OC,
        "true_RC": TRUE_RC,
        "n_states": N_STATES,
        "discount": DISCOUNT,
        "n_buses": N_BUSES,
        "n_periods": N_PERIODS,
        "n_obs": N_BUSES * N_PERIODS,
    }

    # --- Write JSON ---
    json_path = os.path.join(OUT_DIR, "sees_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")

    # --- Write LaTeX snippet ---
    tex_path = os.path.join(OUT_DIR, "sees_results.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by sees_toy.py -- do not edit\n")
        # DGP
        f.write(f"\\newcommand{{\\dgpOC}}{{{TRUE_OC}}}\n")
        f.write(f"\\newcommand{{\\dgpRC}}{{{TRUE_RC}}}\n")
        f.write(f"\\newcommand{{\\dgpNstates}}{{{N_STATES}}}\n")
        f.write(f"\\newcommand{{\\dgpBeta}}{{{DISCOUNT}}}\n")
        f.write(f"\\newcommand{{\\dgpNobs}}{{{N_BUSES * N_PERIODS}}}\n")
        # NFXP
        f.write(f"\\newcommand{{\\nfxpOC}}{{{nfxp_params[0]:.6f}}}\n")
        f.write(f"\\newcommand{{\\nfxpRC}}{{{nfxp_params[1]:.4f}}}\n")
        f.write(f"\\newcommand{{\\nfxpSEOC}}{{{nfxp_se[0]:.6f}}}\n")
        f.write(f"\\newcommand{{\\nfxpSERC}}{{{nfxp_se[1]:.4f}}}\n")
        f.write(f"\\newcommand{{\\nfxpLL}}{{{nfxp_res.log_likelihood:.1f}}}\n")
        f.write(f"\\newcommand{{\\nfxpTime}}{{{nfxp_time:.1f}}}\n")
        # SEES
        f.write(f"\\newcommand{{\\seesOC}}{{{sees_params[0]:.6f}}}\n")
        f.write(f"\\newcommand{{\\seesRC}}{{{sees_params[1]:.4f}}}\n")
        f.write(f"\\newcommand{{\\seesSEOC}}{{{sees_se[0]:.6f}}}\n")
        f.write(f"\\newcommand{{\\seesSERC}}{{{sees_se[1]:.4f}}}\n")
        f.write(f"\\newcommand{{\\seesLL}}{{{sees_res.log_likelihood:.1f}}}\n")
        f.write(f"\\newcommand{{\\seesTime}}{{{sees_time:.1f}}}\n")
        f.write(f"\\newcommand{{\\seesBellmanViol}}{{{bellman_viol:.2e}}}\n")
        # Relative errors
        f.write(f"\\newcommand{{\\seesOCpct}}{{{oc_pct:.1f}}}\n")
        f.write(f"\\newcommand{{\\seesRCpct}}{{{rc_pct:.1f}}}\n")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
