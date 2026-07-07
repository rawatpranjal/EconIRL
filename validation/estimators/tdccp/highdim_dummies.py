#!/usr/bin/env python3
"""TD-CCP on the high-dimensional bus engine (Adusumilli-Eckardt 2025, the high-dim
setup). The relevant state is (mileage, type); we append K iid dummy state variables
drawn from {-10..10} that affect neither the reward nor the transitions. The method
should recover theta = [2, -0.15, 1] despite the dummies polluting the basis and the
first-stage CCP, and the error should stay controlled as K grows (paper Figure 1).

The method is designed for continuous / high-dimensional state spaces, so the state
set is not enumerable. Each observation is its own point in feature space (the
natural continuous-state treatment): a unique state index per observation, the
encoder returns that observation's [mileage, type, mileage*type, K dummies], and the
exact (A, S, S) policy/value solve is skipped (``compute_policy=False``). theta and
its standard errors are still recovered; the parameter stage never needs the kernel.

The dynamic continuation is load-bearing here (not static logit): with the real
next-state link theta1 recovers to ~-0.15, and breaking that link (shuffle s' or set
beta=0) sends theta1 to ~-0.35. ``--check-dynamics`` runs that guard.

Usage:
  PYTHONPATH=src python validation/estimators/tdccp/highdim_dummies.py --ks 0,5,10,20 --seeds 5
  PYTHONPATH=src python validation/estimators/tdccp/highdim_dummies.py --check-dynamics
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
for p in (HERE.parent, ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import dataclasses  # noqa: E402

import bus_engine_mc as be  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.core.types import DDCProblem, Panel, Trajectory  # noqa: E402
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator  # noqa: E402
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402

NT = be.N_TYPES
SCALE = be.MILEAGE_SCALE
THETA = be.THETA_TRUE


def _relevant_truth():
    dgp = be.build_dgp()
    op = SoftBellmanOperator(dgp["problem"], dgp["transitions"])
    truth = value_iteration(
        op, dgp["utility"].compute(jnp.asarray(THETA)), tol=1e-12, max_iter=20000
    )
    init = be.stationary_initial_distribution(dgp["problem"], dgp["transitions"], truth.policy)
    return dgp, truth, init


def build_highdim(dgp, truth, init, K, seed, n_buses=150, shuffle=False):
    """Build a high-dim panel: each obs a unique state with per-obs features
    [mileage, type, mileage*type, K dummies]. ``shuffle`` breaks the s->s' link."""
    rel = simulate_panel_from_policy(
        dgp["problem"],
        dgp["transitions"],
        truth.policy,
        init,
        n_individuals=n_buses,
        n_periods=30,
        seed=seed,
    )
    rng = np.random.default_rng(1000 + seed)
    feats, z_rows, trajs = [], [], []
    counter = 0
    for tr in rel.trajectories:
        st = np.asarray(tr.states)
        ac = np.asarray(tr.actions)
        T = len(st)
        gidx = np.arange(counter, counter + T)
        mil = (st // NT).astype(float)
        typ = (st % NT).astype(float)
        for t in range(T):
            d = rng.integers(-10, 11, K).astype(float) / 10.0 if K else np.zeros(0)
            feats.append(np.concatenate([[mil[t] / SCALE, typ[t], (mil[t] / SCALE) * typ[t]], d]))
            zk = np.zeros((2, 3))
            zk[0] = [1.0, mil[t], typ[t] + 1.0]  # keep; replace normalized to 0
            z_rows.append(zk)
        nxt = np.append(gidx[1:], gidx[-1])  # last period dropped by _extract_transitions
        trajs.append(
            Trajectory(
                states=jnp.asarray(gidx, dtype=jnp.int32),
                actions=jnp.asarray(ac, dtype=jnp.int32),
                next_states=jnp.asarray(nxt, dtype=jnp.int32),
                individual_id=tr.individual_id,
            )
        )
        counter += T
    feat = jnp.asarray(np.array(feats))
    zmat = jnp.asarray(np.array(z_rows))
    n_obs = counter
    if shuffle:
        rng2 = np.random.default_rng(7 + seed)
        trajs = [
            Trajectory(
                states=tr.states,
                actions=tr.actions,
                next_states=jnp.asarray(rng2.integers(0, n_obs, len(tr.next_states)), jnp.int32),
                individual_id=tr.individual_id,
            )
            for tr in trajs
        ]
    problem = DDCProblem(
        num_states=n_obs,
        num_actions=2,
        discount_factor=be.BETA,
        scale_parameter=1.0,
        state_dim=feat.shape[1],
        state_encoder=lambda s: feat[jnp.asarray(s, dtype=jnp.int32)],
    )
    return (
        Panel(trajectories=trajs, metadata={}),
        ActionDependentReward(zmat, be.PARAM_NAMES),
        problem,
        n_obs,
    )


def fit(panel, utility, problem, basis_dim=2, beta_override=None):
    if beta_override is not None:
        problem = dataclasses.replace(problem, discount_factor=beta_override)
    cfg = TDCCPConfig(
        method="semigradient",
        basis_type="encoded",
        basis_dim=basis_dim,
        basis_ridge=1e-5,
        ccp_method="logit",
        ccp_poly_degree=basis_dim,
        ccp_use_encoder=True,
        cross_fitting=False,
        robust_se=False,
        compute_se=False,
        compute_policy=False,
        verbose=False,
    )
    s = TDCCPEstimator(config=cfg, seed=0).estimate(
        panel=panel, utility=utility, problem=problem, transitions=jnp.zeros((2, 1, 1))
    )
    return np.asarray(s.parameters, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="0,5,10,20")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-buses", type=int, default=150)
    ap.add_argument("--basis-dim", type=int, default=2)
    ap.add_argument(
        "--check-dynamics",
        action="store_true",
        help="Prove the dynamic continuation is load-bearing (shuffle s' / beta=0).",
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "validation" / "results" / "tdccp_highdim.json"
    )
    args = ap.parse_args()
    dgp, truth, init = _relevant_truth()

    if args.check_dynamics:
        print("Dynamics load-bearing check (K=0, 3 seeds): theta1 should recover only with real s'")
        for label, kw in [
            ("correct", {}),
            ("shuffled s'", {"shuffle": True}),
            ("myopic beta=0", {"beta": True}),
        ]:
            ths = []
            for sd in range(3):
                p, u, pr, _ = build_highdim(
                    dgp, truth, init, 0, sd, args.n_buses, kw.get("shuffle", False)
                )
                ths.append(
                    fit(p, u, pr, args.basis_dim, beta_override=0.0 if kw.get("beta") else None)
                )
            m = np.array(ths).mean(0)
            print(f"  {label:14s}: theta {np.round(m, 4).tolist()} (true 2,-0.15,1)")
        return

    ks = [int(k) for k in args.ks.split(",")]
    records = []
    print(
        f"high-dim bus: {args.n_buses} buses x 30, basis_dim={args.basis_dim}, "
        f"{args.seeds} seeds. true theta=[2,-0.15,1]"
    )
    for K in ks:
        ths, t0 = [], time.time()
        for sd in range(args.seeds):
            p, u, pr, n_obs = build_highdim(dgp, truth, init, K, sd, args.n_buses)
            ths.append(fit(p, u, pr, args.basis_dim))
        a = np.array(ths)
        m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(args.seeds)
        rec = {
            "K": K,
            "n_obs": int(n_obs),
            "mean": m.tolist(),
            "sem": se.tolist(),
            "theta1_abs_err": float(abs(m[1] - THETA[1])),
        }
        records.append(rec)
        print(
            f"  K={K:3d} (n_obs={n_obs}): theta0 {m[0]:+.3f}({se[0]:.3f})  "
            f"theta1 {m[1]:+.4f}({se[1]:.4f})  theta2 {m[2]:+.3f}({se[2]:.3f})  "
            f"({(time.time() - t0) / args.seeds:.0f}s/seed)",
            flush=True,
        )
    args.out.write_text(
        json.dumps(
            {
                "estimator": "TD-CCP",
                "regime": "high-dimensional (bus + K irrelevant dummy state vars)",
                "paper": "Adusumilli and Eckardt (2025), high-dimensional setup",
                "true_theta": THETA.tolist(),
                "basis_dim": args.basis_dim,
                "n_buses": args.n_buses,
                "seeds": args.seeds,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
