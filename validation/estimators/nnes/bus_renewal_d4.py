#!/usr/bin/env python3
"""NNES paper (Nguyen 2025) Table 2: the efficiency claim in d = 4.

Same renewal DGP as Table 1, four INDEPENDENT 1D modules. The paper's oracle
solves four 1D DPs exploiting that independence; NNES uses one 4D value net. The
joint MLE factorizes across independent modules, so the tabular package estimates
each module separately and the per-module NNES vs oracle comparison is the d = 4
efficiency claim (NNES sd = NFXP sd on all 8 parameters).

DGP: beta=0.9, n=1200 per module, k=12, 100 reps. Modules (crep, c):
1 (2.0, 0.05), 2 (2.5, 0.07), 3 (1.5, 0.09), 4 (1.8, 0.11).

As in the d=2 driver, the transition kernel is discretized to the Delta=0.5 grid,
so absolute dispersion differs from the paper's continuous-mileage design; the
reproduced result is the efficiency EQUALITY (NNES = NFXP).

Usage:
    PYTHONPATH=src:. python validation/estimators/nnes/bus_renewal_d4.py --n-reps 100
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
JSON_OUT = ROOT / "validation" / "results" / "nnes_bus_renewal_d4.json"
for path in (ROOT, ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import bus_renewal_efficiency as bre  # noqa: E402

from econirl.estimation.nfxp import NFXPEstimator  # noqa: E402
from econirl.estimation.nnes import NNESEstimator  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402

# Module (name, true_c, true_crep) for the four independent renewal modules.
MODULES = [
    ("module_1", 0.05, 2.0),
    ("module_2", 0.07, 2.5),
    ("module_3", 0.09, 1.5),
    ("module_4", 0.11, 1.8),
]
# Paper Table 2 (beta=0.9, n=1200, 100 reps): (nfxp_mean, nfxp_sd, nnes_mean, nnes_sd).
# NNES = numerical-derivative column (numerical and analytical agree to ~3 figs).
PAPER = {
    "module_1": {
        "true_crep": 2.0,
        "true_c": 0.05,
        "crep": (1.9733, 0.1366, 1.8947, 0.1401),
        "c": (0.0501, 0.0053, 0.0489, 0.0055),
    },
    "module_2": {
        "true_crep": 2.5,
        "true_c": 0.07,
        "crep": (2.4533, 0.1658, 2.4393, 0.1665),
        "c": (0.0715, 0.0067, 0.0794, 0.0070),
    },
    "module_3": {
        "true_crep": 1.5,
        "true_c": 0.09,
        "crep": (1.5102, 0.1295, 1.5111, 0.1301),
        "c": (0.0897, 0.0085, 0.0890, 0.0085),
    },
    "module_4": {
        "true_crep": 1.8,
        "true_c": 0.11,
        "crep": (1.8222, 0.1467, 1.9023, 0.1503),
        "c": (0.1104, 0.0098, 0.1171, 0.0106),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--n-buses", type=int, default=60)
    parser.add_argument("--n-periods", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=900)
    args = parser.parse_args()

    print("NNES bus-engine renewal efficiency, d=4 (Nguyen 2025, Table 2)")
    print(f"  grid: {bre.S} nodes (Delta={bre.DELTA}, Mmax={bre.MMAX}), beta={bre.BETA}")
    print(f"  panel: {args.n_buses} buses x {args.n_periods} periods   reps: {args.n_reps}")

    utility, problem, transitions = bre.build_module()
    ckpt = JSON_OUT.with_suffix(".checkpoint.jsonl")
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["module"], r["rep"])] = r
        print(f"  resuming: {len(done)} (module, rep) cells in {ckpt.name}", flush=True)

    t0 = time.time()
    with ckpt.open("a") as fh:
        for mod, true_c, true_crep in MODULES:
            policy, init = bre.solve(utility, problem, transitions, true_c, true_crep)
            for rep in range(args.n_reps):
                if (mod, rep) in done:
                    continue
                panel = simulate_panel_from_policy(
                    problem,
                    transitions,
                    policy,
                    init,
                    n_individuals=args.n_buses,
                    n_periods=args.n_periods,
                    seed=args.base_seed + rep,
                )
                nfxp = NFXPEstimator(verbose=False, outer_max_iter=200).estimate(
                    panel=panel, utility=utility, problem=problem, transitions=transitions
                )
                nnes = NNESEstimator().estimate(
                    panel=panel, utility=utility, problem=problem, transitions=transitions
                )
                rec = {
                    "module": mod,
                    "rep": rep,
                    "nfxp": [float(x) for x in np.asarray(nfxp.parameters)],
                    "nnes": [float(x) for x in np.asarray(nnes.parameters)],
                }
                done[(mod, rep)] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if (rep + 1) % 10 == 0 or rep == 0:
                    print(
                        f"  [{mod}] rep {rep + 1}/{args.n_reps}  ({time.time() - t0:.0f}s)",
                        flush=True,
                    )

    results = {}
    for mod, true_c, true_crep in MODULES:
        reps = [r for (m, r) in done if m == mod]
        nfxp_arr = [done[(mod, r)]["nfxp"] for r in sorted(reps)]
        nnes_arr = [done[(mod, r)]["nnes"] for r in sorted(reps)]
        results[mod] = {
            "n_reps": len(reps),
            "nfxp": bre.summarize(nfxp_arr, true_c, true_crep),
            "nnes": bre.summarize(nnes_arr, true_c, true_crep),
        }

    payload = {
        "estimator": "NNES",
        "paper": "Nguyen (2025), Table 2 (four-module bus-engine renewal, d=4), efficiency claim",
        "claim": "NNES attains the NFXP (oracle MLE) efficiency bound in d=4: NNES sd = NFXP sd.",
        "dgp": {
            "grid_nodes": bre.S,
            "delta": bre.DELTA,
            "mmax": bre.MMAX,
            "beta": bre.BETA,
            "exp_rate": bre.LAM,
            "n_buses": args.n_buses,
            "n_periods": args.n_periods,
            "n_reps": args.n_reps,
            "n_modules": 4,
            "note": "independent modules estimated separately; transition kernel "
            "discretized to the grid, the NNES=NFXP equality reproduces.",
        },
        "paper_table2": PAPER,
        "results": results,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ckpt.unlink(missing_ok=True)

    print(
        "\n  module    param   NFXP mean(sd)      NNES mean(sd)     paper NFXP(sd)   paper NNES(sd)"
    )
    for mod, _, _ in MODULES:
        for pk in ("crep", "c"):
            r = results[mod]
            p = PAPER[mod][pk]
            print(
                f"  {mod} {pk:>4}  {r['nfxp'][pk]['mean']:.4f}({r['nfxp'][pk]['sd']:.4f})  "
                f"{r['nnes'][pk]['mean']:.4f}({r['nnes'][pk]['sd']:.4f})   "
                f"{p[0]}({p[1]})   {p[2]}({p[3]})"
            )
    print(f"\n  wrote: {JSON_OUT}")


if __name__ == "__main__":
    main()
