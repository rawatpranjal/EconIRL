"""MPEC vs NFXP-NK Monte Carlo comparison.

Runs a grid over discount factors and state space sizes to show the
crossover between MPEC and NFXP-NK. Inspired by Su and Judd (2012)
and the comment by Iskhakov, Rust and Schjerning (2016) showing that
NFXP-NK (hybrid SA-then-NK polyalgorithm) eliminates the speed
advantage that Su and Judd observed against the weaker NFXP-SA.

Grid: 3 discount factors x 2 state sizes = 6 cells, 1 rep each.
"""

import time
import numpy as np
import json

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

N_INDIVIDUALS = 200
N_PERIODS = 100
TRUE_OC = 0.001
TRUE_RC = 3.0
SEED = 42

GRID = [
    # (beta, n_bins)
    (0.95,   90),
    (0.95,  200),
    (0.99,   90),
    (0.99,  200),
    (0.9999, 90),
    (0.9999, 200),
]


def run_cell(beta: float, n_bins: int):
    """Run one grid cell: NFXP-NK and MPEC-SLSQP."""
    env = RustBusEnvironment(
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        num_mileage_bins=n_bins,
        discount_factor=beta,
        seed=SEED,
    )
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)

    panel = simulate_panel(
        env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED + 1000,
    )

    true_params = np.array([TRUE_OC, TRUE_RC])
    results = {}

    # NFXP-NK (hybrid polyalgorithm from Rust 1987/2000)
    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-12,
        inner_max_iter=300000,
        switch_tol=1e-3,
        outer_max_iter=200,
        compute_hessian=False,
        verbose=False,
    )
    t0 = time.time()
    try:
        nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
        nfxp_time = time.time() - t0
        results["NFXP-NK"] = {
            "params": np.asarray(nfxp_result.parameters).tolist(),
            "ll": float(nfxp_result.log_likelihood),
            "time": nfxp_time,
            "converged": bool(nfxp_result.converged),
        }
    except Exception as e:
        results["NFXP-NK"] = {
            "params": [float("nan")] * 2,
            "ll": float("nan"),
            "time": time.time() - t0,
            "converged": False,
            "error": str(e),
        }

    # MPEC-SLSQP
    mpec = MPECEstimator(
        config=MPECConfig(solver="slsqp", max_iter=500, constraint_tol=1e-8),
        compute_hessian=False,
        verbose=False,
    )
    t0 = time.time()
    try:
        mpec_result = mpec.estimate(panel, utility, problem, transitions)
        mpec_time = time.time() - t0
        results["MPEC"] = {
            "params": np.asarray(mpec_result.parameters).tolist(),
            "ll": float(mpec_result.log_likelihood),
            "time": mpec_time,
            "converged": bool(mpec_result.converged),
            "constraint_violation": mpec_result.metadata.get(
                "final_constraint_violation", float("nan")
            ),
        }
    except Exception as e:
        results["MPEC"] = {
            "params": [float("nan")] * 2,
            "ll": float("nan"),
            "time": time.time() - t0,
            "converged": False,
            "error": str(e),
        }

    return results


def main():
    print("MPEC vs NFXP-NK: grid over beta and state space size")
    print(f"Data: {N_INDIVIDUALS} buses x {N_PERIODS} periods per cell")
    print(f"True params: theta_c={TRUE_OC}, RC={TRUE_RC}")
    print()

    all_results = []

    for beta, n_bins in GRID:
        print(f"beta={beta}, N={n_bins}...", end=" ", flush=True)
        res = run_cell(beta, n_bins)
        res["beta"] = beta
        res["n_bins"] = n_bins
        all_results.append(res)

        nfxp = res["NFXP-NK"]
        mpec = res["MPEC"]
        print(f"NFXP: {nfxp['time']:.1f}s (conv={nfxp['converged']}) | "
              f"MPEC: {mpec['time']:.1f}s (conv={mpec['converged']})")

    # Summary table
    print("\n" + "=" * 90)
    print("Summary: MPEC vs NFXP-NK across discount factors and state space sizes")
    print("=" * 90)
    header = (f"{'beta':>8} {'N':>5} | "
              f"{'NFXP time':>10} {'NFXP conv':>10} {'NFXP LL':>12} | "
              f"{'MPEC time':>10} {'MPEC conv':>10} {'MPEC LL':>12} | "
              f"{'Ratio':>8}")
    print(header)
    print("-" * len(header))

    for res in all_results:
        nfxp = res["NFXP-NK"]
        mpec = res["MPEC"]
        ratio = nfxp["time"] / max(mpec["time"], 0.01) if mpec["converged"] else float("nan")
        print(f"{res['beta']:>8} {res['n_bins']:>5} | "
              f"{nfxp['time']:>9.1f}s {str(nfxp['converged']):>10} {nfxp['ll']:>12.2f} | "
              f"{mpec['time']:>9.1f}s {str(mpec['converged']):>10} {mpec['ll']:>12.2f} | "
              f"{ratio:>7.1f}x")

    with open("mpec_vs_nfxp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to mpec_vs_nfxp_results.json")


if __name__ == "__main__":
    main()
