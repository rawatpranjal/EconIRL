#!/usr/bin/env python3
"""Autolab runner: S/M/L benchmark, one estimator at a time.

Runs each estimator sequentially across all three tiers (Small/Medium/Large),
appending results to a CSV after each estimator completes.
"""

import csv
import math
import os
import signal
import sys
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from econirl.evaluation.benchmark import (
    BenchmarkDGP,
    EstimatorSpec,
    run_single,
    get_scaling_estimator_specs,
)

# --- Configuration ---
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "sml_benchmark.csv")
TIMEOUT_SECONDS = 900  # 15 min per estimator per tier
SEED = 42

# S/M/L tiers: (tier_name, K, M)
TIERS = [
    ("Small", 1, 20),   # 20 states
    ("Medium", 2, 15),  # 225 states
    ("Large", 3, 15),   # 3375 states
]

# Curated estimator list (names to include)
CURATED = {
    "BC", "NFXP", "CCP", "MCE IRL", "Deep MaxEnt",
    "Max Margin", "AIRL", "TD-CCP", "GLADIUS", "NNES", "SEES",
}

CSV_COLUMNS = [
    "tier", "K", "M", "estimator", "n_states", "n_agents", "n_periods",
    "time_seconds", "pct_optimal", "pct_optimal_transfer",
    "converged", "skipped",
]


def get_completed_estimators() -> set[str]:
    """Read CSV and return set of estimator names that have all 3 tiers done."""
    if not os.path.exists(CSV_PATH):
        return set()
    done: dict[str, set[str]] = {}
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["estimator"]
            tier = row["tier"]
            if name not in done:
                done[name] = set()
            done[name].add(tier)
    # Only count as complete if all 3 tiers present
    return {name for name, tiers in done.items() if len(tiers) == 3}


def append_result(row: dict) -> None:
    """Append a single result row to the CSV."""
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_curated_spec(name: str, n_states: int) -> EstimatorSpec | None:
    """Get the EstimatorSpec for a given name from the scaling specs."""
    all_specs = get_scaling_estimator_specs(n_states)
    for spec in all_specs:
        if spec.name == name:
            return spec
    return None


def run_one_estimator(est_name: str) -> None:
    """Run a single estimator across all three tiers."""
    print(f"\n{'='*60}")
    print(f"  ESTIMATOR: {est_name}")
    print(f"{'='*60}")

    timed_out = False

    for tier_name, K, M in TIERS:
        n_states = M ** K
        n_agents = max(200, 2 * n_states)
        n_periods = max(100, n_states)

        print(f"\n  --- {tier_name}: K={K}, M={M}, states={n_states}, "
              f"agents={n_agents}, periods={n_periods} ---")

        if timed_out:
            row = {
                "tier": tier_name, "K": K, "M": M,
                "estimator": est_name, "n_states": n_states,
                "n_agents": n_agents, "n_periods": n_periods,
                "time_seconds": "", "pct_optimal": "",
                "pct_optimal_transfer": "",
                "converged": False, "skipped": True,
            }
            append_result(row)
            print(f"    SKIPPED (timed out at smaller tier)")
            continue

        spec = get_curated_spec(est_name, n_states)
        if spec is None:
            print(f"    ERROR: No spec found for {est_name}")
            continue

        dgp = BenchmarkDGP(K=K, n_states=M, discount_factor=0.99)

        try:
            t0 = time.perf_counter()
            result = run_single(dgp, spec, n_agents=n_agents,
                                n_periods=n_periods, seed=SEED)
            elapsed = result.time_seconds

            transfer = result.pct_optimal_transfer
            transfer_str = f"{transfer:.1f}%" if transfer is not None else "N/A"
            status = "OK" if result.converged else "FAIL"

            print(f"    {elapsed:7.1f}s  pct_optimal={result.pct_optimal:6.1f}%  "
                  f"transfer={transfer_str}  [{status}]")

            row = {
                "tier": tier_name, "K": K, "M": M,
                "estimator": est_name, "n_states": n_states,
                "n_agents": n_agents, "n_periods": n_periods,
                "time_seconds": f"{elapsed:.2f}",
                "pct_optimal": f"{result.pct_optimal:.2f}" if not math.isnan(result.pct_optimal) else "",
                "pct_optimal_transfer": f"{transfer:.2f}" if transfer is not None else "",
                "converged": result.converged,
                "skipped": False,
            }
            append_result(row)

            if elapsed > TIMEOUT_SECONDS:
                timed_out = True
                print(f"    ^ TIMED OUT ({elapsed:.0f}s > {TIMEOUT_SECONDS}s)")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"    ERROR after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            row = {
                "tier": tier_name, "K": K, "M": M,
                "estimator": est_name, "n_states": n_states,
                "n_agents": n_agents, "n_periods": n_periods,
                "time_seconds": f"{elapsed:.2f}",
                "pct_optimal": "", "pct_optimal_transfer": "",
                "converged": False, "skipped": False,
            }
            append_result(row)
            if elapsed > TIMEOUT_SECONDS:
                timed_out = True


def main():
    # Determine which estimators still need running
    completed = get_completed_estimators()
    remaining = [name for name in CURATED if name not in completed]

    if not remaining:
        print("All curated estimators have been run!")
        print(f"Results at: {CSV_PATH}")
        return

    # Sort for deterministic order
    order = ["BC", "NFXP", "CCP", "MCE IRL", "Deep MaxEnt",
             "Max Margin", "AIRL", "TD-CCP", "GLADIUS", "NNES", "SEES"]
    remaining = [n for n in order if n in remaining]

    print(f"Remaining estimators: {remaining}")
    print(f"Already completed: {sorted(completed) if completed else 'none'}")
    print(f"Output: {CSV_PATH}")

    # Run one at a time
    for est_name in remaining:
        run_one_estimator(est_name)
        print(f"\n  >>> {est_name} DONE <<<\n")


if __name__ == "__main__":
    main()
