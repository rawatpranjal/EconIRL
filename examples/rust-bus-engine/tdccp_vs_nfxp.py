"""NFXP vs CCP vs TD-CCP comparison on large Rust bus.

Single-replication comparison on 200-bin Rust bus at high discount
showing three approaches to the Bellman fixed-point problem:
- NFXP: exact Bellman solve (hybrid SA-then-NK)
- CCP: Hotz-Miller matrix inversion (avoids inner loop)
- TD-CCP: neural approximate value iteration (avoids inner loop + model-free)
"""

import time
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

N_BINS = 200
DISCOUNT = 0.9999
TRUE_OC = 0.001
TRUE_RC = 3.0
N_INDIVIDUALS = 200
N_PERIODS = 100
SEED = 42


def main():
    print(f"Environment: Rust bus, {N_BINS} bins, beta={DISCOUNT}")
    print(f"Data: {N_INDIVIDUALS} buses x {N_PERIODS} periods = "
          f"{N_INDIVIDUALS * N_PERIODS} obs\n")

    env = RustBusEnvironment(
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        num_mileage_bins=N_BINS,
        discount_factor=DISCOUNT,
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

    # --- NFXP ---
    print("Running NFXP (hybrid SA-then-NK)...")
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
    nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    nfxp_params = np.asarray(nfxp_result.parameters)
    print(f"  params={nfxp_params}, LL={nfxp_result.log_likelihood:.2f}, "
          f"time={nfxp_time:.1f}s\n")
    results["NFXP"] = {
        "params": nfxp_params, "ll": nfxp_result.log_likelihood,
        "time": nfxp_time, "converged": nfxp_result.converged,
    }

    # --- CCP ---
    print("Running CCP (Hotz-Miller with NPL iterations)...")
    ccp = CCPEstimator(
        num_policy_iterations=5,
        compute_hessian=False,
        verbose=False,
    )
    t0 = time.time()
    ccp_result = ccp.estimate(panel, utility, problem, transitions)
    ccp_time = time.time() - t0
    ccp_params = np.asarray(ccp_result.parameters)
    print(f"  params={ccp_params}, LL={ccp_result.log_likelihood:.2f}, "
          f"time={ccp_time:.1f}s\n")
    results["CCP"] = {
        "params": ccp_params, "ll": ccp_result.log_likelihood,
        "time": ccp_time, "converged": ccp_result.converged,
    }

    # --- TD-CCP ---
    print("Running TD-CCP (neural approximate VI)...")
    tdccp_config = TDCCPConfig(
        hidden_dim=32,
        num_hidden_layers=2,
        avi_iterations=15,
        epochs_per_avi=20,
        learning_rate=1e-3,
        batch_size=4096,
        n_policy_iterations=3,
        compute_se=False,
        verbose=False,
    )
    tdccp = TDCCPEstimator(config=tdccp_config)
    t0 = time.time()
    tdccp_result = tdccp.estimate(panel, utility, problem, transitions)
    tdccp_time = time.time() - t0
    tdccp_params = np.asarray(tdccp_result.parameters)
    print(f"  params={tdccp_params}, LL={tdccp_result.log_likelihood:.2f}, "
          f"time={tdccp_time:.1f}s\n")
    results["TD-CCP"] = {
        "params": tdccp_params, "ll": tdccp_result.log_likelihood,
        "time": tdccp_time, "converged": tdccp_result.converged,
    }

    # --- Summary ---
    print("=" * 75)
    print(f"NFXP vs CCP vs TD-CCP: {N_BINS} bins, beta={DISCOUNT}")
    print(f"True params: theta_c={TRUE_OC}, RC={TRUE_RC}")
    print("=" * 75)

    header = f"{'Metric':<25} {'NFXP':>15} {'CCP':>15} {'TD-CCP':>15}"
    print(header)
    print("-" * len(header))

    for name in ["NFXP", "CCP", "TD-CCP"]:
        r = results[name]
        bias = r["params"] - true_params
        results[name]["bias"] = bias

    names = ["NFXP", "CCP", "TD-CCP"]
    print(f"{'theta_c':<25} " + " ".join(
        f"{results[n]['params'][0]:>15.6f}" for n in names))
    print(f"{'RC':<25} " + " ".join(
        f"{results[n]['params'][1]:>15.4f}" for n in names))
    print(f"{'Bias (theta_c)':<25} " + " ".join(
        f"{results[n]['bias'][0]:>15.6f}" for n in names))
    print(f"{'Bias (RC)':<25} " + " ".join(
        f"{results[n]['bias'][1]:>15.4f}" for n in names))
    print(f"{'Log-likelihood':<25} " + " ".join(
        f"{results[n]['ll']:>15.2f}" for n in names))
    print(f"{'Wall time (s)':<25} " + " ".join(
        f"{results[n]['time']:>15.1f}" for n in names))
    print(f"{'Converged':<25} " + " ".join(
        f"{str(results[n]['converged']):>15}" for n in names))


if __name__ == "__main__":
    main()
