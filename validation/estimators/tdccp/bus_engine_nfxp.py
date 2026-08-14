#!/usr/bin/env python3
"""NFXP reference replication of the Adusumilli-Eckardt (2025) bus-engine Monte
Carlo (Online Appendix Table B.1).

The paper's linear semi-gradient TD-CCP recovers the bus-engine parameters at the
efficiency of maximum likelihood. This driver runs the nested fixed point (the
efficient MLE) on the same DGP and Monte Carlo design, so the package's reference
estimator reproduces the published Table B.1 means, standard deviations, and MSE.
It shares the exact DGP with ``bus_engine_mc.py`` (type s in {1,2}, beta=0.9,
theta = (2, -0.15, 1), 1000 buses, T=30).

Usage:
    PYTHONPATH=src:. python validation/estimators/tdccp/bus_engine_nfxp.py --n-reps 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "tdccp_bus_engine_nfxp.json"
for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import bus_engine_mc as B  # noqa: E402  (shared DGP)
import jax.numpy as jnp  # noqa: E402

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.estimation.nfxp import NFXPEstimator  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402
from validation.known_truth import to_jsonable  # noqa: E402


def summarize(estimates: np.ndarray, ses: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(B.PARAM_NAMES):
        col = estimates[:, j]
        err = col - B.THETA_TRUE[j]
        out[name] = {
            "true": float(B.THETA_TRUE[j]),
            "mean": float(np.mean(col)),
            "sd": float(np.std(col, ddof=1)) if col.size > 1 else 0.0,
            "bias": float(np.mean(err)),
            "mse": float(np.mean(err**2)),
            "mean_asymptotic_se": float(np.nanmean(ses[:, j])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--n-buses", type=int, default=1000)
    parser.add_argument("--n-periods", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20250)
    args = parser.parse_args()

    print("NFXP bus-engine Monte Carlo (Adusumilli-Eckardt 2025, Table B.1 reference)")
    print(
        f"  DGP: X_MAX={B.X_MAX}, types={B.N_TYPES}, beta={B.BETA}, "
        f"theta_true={B.THETA_TRUE.tolist()}"
    )
    print(f"  panel: {args.n_buses} buses x {args.n_periods} periods   reps: {args.n_reps}")

    dgp = B.build_dgp()
    operator = SoftBellmanOperator(dgp["problem"], dgp["transitions"])
    true_reward = dgp["utility"].compute(jnp.asarray(B.THETA_TRUE))
    truth = value_iteration(operator, true_reward, tol=1e-12, max_iter=20_000)
    policy = truth.policy
    init_dist = B.stationary_initial_distribution(dgp["problem"], dgp["transitions"], policy)
    replacement_share = float(np.asarray(init_dist) @ np.asarray(policy)[:, 1])
    print(f"  stationary replacement share: {replacement_share:.4f}")

    # Checkpoint/resume: append each rep to a JSONL so a killed run resumes.
    ckpt = JSON_OUT.with_suffix(".checkpoint.jsonl")
    done: dict[int, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[int(rec["rep"])] = rec
        print(f"  resuming: {len(done)} reps already in {ckpt.name}", flush=True)

    t0 = time.time()
    with ckpt.open("a") as fh:
        for rep in range(args.n_reps):
            if rep in done:
                continue
            seed = args.base_seed + rep
            panel = simulate_panel_from_policy(
                dgp["problem"],
                dgp["transitions"],
                policy,
                init_dist,
                n_individuals=args.n_buses,
                n_periods=args.n_periods,
                seed=seed,
            )
            s = NFXPEstimator().estimate(
                panel=panel,
                utility=dgp["utility"],
                problem=dgp["problem"],
                transitions=dgp["transitions"],
            )
            params = np.asarray(s.parameters, dtype=np.float64)
            se = (
                np.asarray(s.standard_errors, dtype=np.float64)
                if s.standard_errors is not None
                else [float("nan")] * 3
            )
            rec = {"rep": rep, "params": [float(x) for x in params], "ses": [float(x) for x in se]}
            done[rep] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (len(done)) % 25 == 0 or rep == 0:
                print(
                    f"  rep {rep + 1}/{args.n_reps}  done={len(done)}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    order = sorted(done)
    estimates = np.array([done[r]["params"] for r in order], dtype=np.float64)
    ses = np.array([done[r]["ses"] for r in order], dtype=np.float64)
    results = summarize(estimates, ses)
    payload = {
        "estimator": "NFXP",
        "paper": "Adusumilli and Eckardt (2025), Table B.1 (bus-engine replacement), reference MLE",
        "arxiv": "1912.09509",
        "dgp": {
            "model": "bus-engine replacement, deterministic mileage, permanent type s in {1,2}",
            "x_max": B.X_MAX,
            "n_types": B.N_TYPES,
            "beta": B.BETA,
            "theta_true": B.THETA_TRUE.tolist(),
            "param_names": B.PARAM_NAMES,
            "n_buses": args.n_buses,
            "n_periods": args.n_periods,
            "n_reps": len(order),
        },
        "paper_table_b1": B.PAPER_TABLE_B1,
        "results": {"nfxp_mle": results},
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ckpt.unlink(missing_ok=True)

    print(
        "\n  parameter        true   pkg mean  paper mean   pkg SD   paper SD   pkg MSE   paper MSE"
    )
    for name in B.PARAM_NAMES:
        r = results[name]
        p = B.PAPER_TABLE_B1[name]
        print(
            f"  {name:<16}{r['true']:>6.2f}{r['mean']:>10.4f}{p['mean_nlr']:>11.4f}"
            f"{r['sd']:>10.4f}{p['sd_nlr']:>10.4f}{r['mse']:>10.5f}{p['mse_nlr']:>11.5f}"
        )
    print(f"\n  wrote: {JSON_OUT}")


if __name__ == "__main__":
    main()
