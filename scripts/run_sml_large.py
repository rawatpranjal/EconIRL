#!/usr/bin/env python3
"""Run a single estimator at the Large tier (K=3, M=15, 3375 states).

Usage: python scripts/run_sml_large.py "ESTIMATOR_NAME"

Uses capped data: 1000 agents × 500 periods = 500K transitions.
"""

import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from econirl.evaluation.benchmark import (
    BenchmarkDGP,
    run_single,
    get_scaling_estimator_specs,
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "sml_benchmark.csv")
COLS = [
    "tier", "K", "M", "estimator", "n_states", "n_agents", "n_periods",
    "time_seconds", "pct_optimal", "pct_optimal_transfer", "converged", "skipped",
]


def save(row):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    est_name = sys.argv[1] if len(sys.argv) > 1 else None
    if not est_name:
        print("Usage: python scripts/run_sml_large.py 'ESTIMATOR_NAME'")
        sys.exit(1)

    K, M = 3, 15
    n_states = M ** K  # 3375
    n_agents = 1000
    n_periods = 500
    dgp = BenchmarkDGP(K=K, n_states=M, discount_factor=0.99)

    specs = get_scaling_estimator_specs(n_states)
    spec = next((s for s in specs if s.name == est_name), None)
    if spec is None:
        print(f"Unknown estimator: {est_name}")
        print(f"Available: {[s.name for s in specs]}")
        sys.exit(1)

    print(f"=== {est_name} Large ===")
    print(f"K={K}, M={M}, states={n_states}, agents={n_agents}, periods={n_periods}")
    print("Running...", flush=True)

    t0 = time.perf_counter()
    r = run_single(dgp, spec, n_agents=n_agents, n_periods=n_periods, seed=42)
    wall = time.perf_counter() - t0

    tr = r.pct_optimal_transfer
    trstr = f"{tr:.1f}%" if tr is not None else "N/A"
    status = "OK" if r.converged else "FAIL"
    print(f"  est_time={r.time_seconds:.1f}s  wall={wall:.1f}s  "
          f"pct={r.pct_optimal:.1f}%  transfer={trstr}  [{status}]")

    save({
        "tier": "Large", "K": K, "M": M,
        "estimator": est_name, "n_states": n_states,
        "n_agents": n_agents, "n_periods": n_periods,
        "time_seconds": f"{r.time_seconds:.2f}",
        "pct_optimal": f"{r.pct_optimal:.2f}" if not math.isnan(r.pct_optimal) else "",
        "pct_optimal_transfer": f"{tr:.2f}" if tr is not None else "",
        "converged": r.converged, "skipped": False,
    })
    print(f"{est_name} Large DONE")


if __name__ == "__main__":
    main()
