"""NNES Bellman approximation error experiment.

Demonstrates that NNES-NPL is robust to V-approximation error while
NNES-NFXP is not. The key theoretical result is Neyman orthogonality
(Nguyen 2025, Propositions 3-4): the NPL mapping has zero Jacobian at
the true value function, so first-order errors in V drop out of the
Phase 2 score. NNES-NFXP lacks this property, and V-errors contaminate
the structural parameter estimates.

The experiment varies v_epochs (the number of SGD epochs used to train
the V-network in Phase 1) from a very small number (severe early
stopping, large V-error) to a large number (well-converged V). For
each setting, both NNES-NPL and NNES-NFXP are fitted on the same data.
The prediction is that NNES-NPL theta estimates remain stable across
all v_epochs settings, while NNES-NFXP estimates are biased when
v_epochs is small and only converge to the truth at high v_epochs.

Usage:
    python experiments/nnes_bellman_error/run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from econirl import NNES
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRUE_THETA_C = 0.001
TRUE_RC = 3.0
DISCOUNT = 0.9999
N_STATES = 90

N_INDIVIDUALS = 5000
N_PERIODS = 100

V_EPOCHS_GRID = [10, 25, 50, 100, 200, 500]
N_REPLICATIONS = 3  # increase to 10 for smoother results


def fit_nnes(panel, bellman: str, v_epochs: int, seed: int = 0) -> dict:
    """Fit a single NNES model and return parameter estimates.

    Returns a dict with keys theta_c, RC, and converged. If the fit
    raises an exception, theta_c and RC are set to NaN.
    """
    model = NNES(
        n_states=N_STATES,
        n_actions=2,
        discount=DISCOUNT,
        bellman=bellman,
        hidden_dim=32,
        num_layers=2,
        v_lr=1e-3,
        v_epochs=v_epochs,
        n_outer_iterations=1,
        verbose=False,
    )
    try:
        model.fit(panel)
        return {
            "theta_c": model.params_["theta_c"],
            "RC": model.params_["RC"],
            "converged": model.converged_,
        }
    except Exception as e:
        print(f"    [WARN] {bellman} v_epochs={v_epochs} failed: {e}")
        return {"theta_c": float("nan"), "RC": float("nan"), "converged": False}


def main():
    print("=" * 72)
    print("NNES Bellman Approximation Error Experiment")
    print("=" * 72)
    print()
    print(f"True parameters: theta_c = {TRUE_THETA_C}, RC = {TRUE_RC}")
    print(f"Discount factor: {DISCOUNT}")
    print(f"Data: {N_INDIVIDUALS} individuals x {N_PERIODS} periods")
    print(f"V-epochs grid: {V_EPOCHS_GRID}")
    print(f"Monte Carlo replications: {N_REPLICATIONS}")
    print()
    print(
        "Theory predicts NNES-NPL estimates remain stable across all "
        "v_epochs values because the NPL Bellman has the zero Jacobian "
        "property (Neyman orthogonality). NNES-NFXP estimates should be "
        "biased when v_epochs is small because V-approximation errors "
        "contaminate the Phase 2 score."
    )
    print()

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    env = RustBusEnvironment(
        operating_cost=TRUE_THETA_C,
        replacement_cost=TRUE_RC,
        num_mileage_bins=N_STATES,
        discount_factor=DISCOUNT,
    )

    # ------------------------------------------------------------------
    # Storage for Monte Carlo results
    # ------------------------------------------------------------------
    # results[bellman][v_epochs] = list of dicts across replications
    results = {
        "npl": {ve: [] for ve in V_EPOCHS_GRID},
        "nfxp": {ve: [] for ve in V_EPOCHS_GRID},
    }

    t_start = time.time()

    for rep in range(N_REPLICATIONS):
        # Simulate a fresh dataset for each replication
        seed = 1000 + rep
        print(f"--- Replication {rep + 1}/{N_REPLICATIONS} (seed={seed}) ---")
        panel = simulate_panel(
            env,
            n_individuals=N_INDIVIDUALS,
            n_periods=N_PERIODS,
            seed=seed,
        )

        for v_epochs in V_EPOCHS_GRID:
            for bellman in ["npl", "nfxp"]:
                t0 = time.time()
                res = fit_nnes(panel, bellman=bellman, v_epochs=v_epochs, seed=seed)
                elapsed = time.time() - t0
                results[bellman][v_epochs].append(res)
                print(
                    f"  {bellman:>4s}  v_epochs={v_epochs:>4d}  "
                    f"theta_c={res['theta_c']:.6f}  RC={res['RC']:.4f}  "
                    f"({elapsed:.1f}s)"
                )

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s")

    # ------------------------------------------------------------------
    # Compute bias and RMSE
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("RESULTS: Point Estimates (averaged over replications)")
    print("=" * 72)
    print()
    header = (
        f"{'v_epochs':>8s}  "
        f"{'NPL theta_c':>12s}  {'NPL RC':>8s}  "
        f"{'NFXP theta_c':>13s}  {'NFXP RC':>8s}  "
        f"{'True theta_c':>13s}  {'True RC':>8s}"
    )
    print(header)
    print("-" * len(header))

    for v_epochs in V_EPOCHS_GRID:
        npl_tc = np.nanmean([r["theta_c"] for r in results["npl"][v_epochs]])
        npl_rc = np.nanmean([r["RC"] for r in results["npl"][v_epochs]])
        nfxp_tc = np.nanmean([r["theta_c"] for r in results["nfxp"][v_epochs]])
        nfxp_rc = np.nanmean([r["RC"] for r in results["nfxp"][v_epochs]])
        print(
            f"{v_epochs:>8d}  "
            f"{npl_tc:>12.6f}  {npl_rc:>8.4f}  "
            f"{nfxp_tc:>13.6f}  {nfxp_rc:>8.4f}  "
            f"{TRUE_THETA_C:>13.6f}  {TRUE_RC:>8.4f}"
        )

    print()
    print("=" * 72)
    print("RESULTS: Bias (estimate minus truth)")
    print("=" * 72)
    print()
    header_bias = (
        f"{'v_epochs':>8s}  "
        f"{'NPL bias_tc':>12s}  {'NPL bias_RC':>12s}  "
        f"{'NFXP bias_tc':>13s}  {'NFXP bias_RC':>13s}"
    )
    print(header_bias)
    print("-" * len(header_bias))

    for v_epochs in V_EPOCHS_GRID:
        npl_bias_tc = np.nanmean(
            [r["theta_c"] - TRUE_THETA_C for r in results["npl"][v_epochs]]
        )
        npl_bias_rc = np.nanmean(
            [r["RC"] - TRUE_RC for r in results["npl"][v_epochs]]
        )
        nfxp_bias_tc = np.nanmean(
            [r["theta_c"] - TRUE_THETA_C for r in results["nfxp"][v_epochs]]
        )
        nfxp_bias_rc = np.nanmean(
            [r["RC"] - TRUE_RC for r in results["nfxp"][v_epochs]]
        )
        print(
            f"{v_epochs:>8d}  "
            f"{npl_bias_tc:>12.6f}  {npl_bias_rc:>12.4f}  "
            f"{nfxp_bias_tc:>13.6f}  {nfxp_bias_rc:>13.4f}"
        )

    print()
    print("=" * 72)
    print("RESULTS: RMSE")
    print("=" * 72)
    print()
    header_rmse = (
        f"{'v_epochs':>8s}  "
        f"{'NPL rmse_tc':>12s}  {'NPL rmse_RC':>12s}  "
        f"{'NFXP rmse_tc':>13s}  {'NFXP rmse_RC':>13s}"
    )
    print(header_rmse)
    print("-" * len(header_rmse))

    for v_epochs in V_EPOCHS_GRID:
        npl_rmse_tc = np.sqrt(
            np.nanmean(
                [(r["theta_c"] - TRUE_THETA_C) ** 2 for r in results["npl"][v_epochs]]
            )
        )
        npl_rmse_rc = np.sqrt(
            np.nanmean(
                [(r["RC"] - TRUE_RC) ** 2 for r in results["npl"][v_epochs]]
            )
        )
        nfxp_rmse_tc = np.sqrt(
            np.nanmean(
                [(r["theta_c"] - TRUE_THETA_C) ** 2 for r in results["nfxp"][v_epochs]]
            )
        )
        nfxp_rmse_rc = np.sqrt(
            np.nanmean(
                [(r["RC"] - TRUE_RC) ** 2 for r in results["nfxp"][v_epochs]]
            )
        )
        print(
            f"{v_epochs:>8d}  "
            f"{npl_rmse_tc:>12.6f}  {npl_rmse_rc:>12.4f}  "
            f"{nfxp_rmse_tc:>13.6f}  {nfxp_rmse_rc:>13.4f}"
        )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print()

    # Check whether NPL bias is stable (low variance across v_epochs)
    npl_biases_tc = []
    nfxp_biases_tc = []
    for v_epochs in V_EPOCHS_GRID:
        npl_biases_tc.append(
            np.nanmean([r["theta_c"] - TRUE_THETA_C for r in results["npl"][v_epochs]])
        )
        nfxp_biases_tc.append(
            np.nanmean([r["theta_c"] - TRUE_THETA_C for r in results["nfxp"][v_epochs]])
        )

    npl_range_tc = max(npl_biases_tc) - min(npl_biases_tc)
    nfxp_range_tc = max(nfxp_biases_tc) - min(nfxp_biases_tc)

    print(
        f"Range of mean theta_c bias across v_epochs settings: "
        f"NPL = {npl_range_tc:.6f}, NFXP = {nfxp_range_tc:.6f}"
    )

    if nfxp_range_tc > 2 * npl_range_tc:
        print(
            "NFXP bias varies much more with v_epochs than NPL, consistent "
            "with the theory that NFXP lacks Neyman orthogonality."
        )
    elif npl_range_tc < 1e-4 and nfxp_range_tc < 1e-4:
        print(
            "Both methods show very small bias variation. With 5000 "
            "trajectories and 100 periods, even NFXP may have enough data "
            "to compensate for V-error at these epoch counts. Try smaller "
            "v_epochs values or fewer trajectories to see the divergence."
        )
    else:
        print(
            "Results are ambiguous at this sample size and replication "
            "count. Increase N_REPLICATIONS to 10 for cleaner separation."
        )


if __name__ == "__main__":
    main()
