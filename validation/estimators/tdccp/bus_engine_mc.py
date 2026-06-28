#!/usr/bin/env python3
"""TD-CCP Monte Carlo on the Adusumilli-Eckardt (2025) bus-engine model.

Reproduces Online Appendix B.7.2 / Table B.1 of Adusumilli and Eckardt,
"Temporal-Difference Estimation of Dynamic Discrete Choice Models" (Review of
Economic Studies, 2025; arXiv 1912.09509). The DGP is a Rust-style bus-engine
replacement problem:

  keep  (a=0): payoff theta0 + theta1 * x + theta2 * s, mileage x -> x + 1
  replace (a=1): payoff normalized to 0,             mileage x -> 0

x is mileage (deterministic, +1 on keep, reset on replace), s in {1,2} is a
permanent bus type known to the econometrician, shocks are Type-1 EV, beta=0.9.
True parameters: theta0 = 2, theta1 = -0.15, theta2 = 1.

Estimator: linear semi-gradient with a third-order polynomial basis in (x, s)
and logit first-stage CCPs, run both without and with the locally robust
correction (two-fold cross-fitting). Reports per-parameter mean, empirical SD,
and MSE across 1000 simulations, matching the paper's Table B.1.

Usage:
    PYTHONPATH=src:. python validation/estimators/tdccp/bus_engine_mc.py --n-reps 1000
    PYTHONPATH=src:. python validation/estimators/tdccp/bus_engine_mc.py --n-reps 30  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON_OUT = ROOT / "validation" / "results" / "tdccp_bus_engine.json"

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from econirl.core.types import DDCProblem  # noqa: E402
from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator  # noqa: E402
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402
from validation.known_truth import to_jsonable  # noqa: E402

# Paper Table B.1 DGP --------------------------------------------------------
THETA_TRUE = np.array([2.0, -0.15, 1.0])          # theta0, theta1 (mileage), theta2 (type)
PARAM_NAMES = ["theta0_intercept", "theta1_mileage", "theta2_type"]
BETA = 0.9
X_MAX = 40                                          # mileage grid 0..X_MAX (replacement well before)
MILEAGE_SCALE = 20.0                                # basis normalization (~replacement region), conditions the polynomial
N_TYPES = 2                                         # s in {0, 1}
# Paper Table B.1 reported values (not locally robust | locally robust)
PAPER_TABLE_B1 = {
    "theta0_intercept": {"true": 2.0, "mean_nlr": 1.9788, "mse_nlr": 0.0080, "sd_nlr": 0.0868,
                         "mean_lr": 1.9778, "mse_lr": 0.0081, "sd_lr": 0.0870},
    "theta1_mileage":   {"true": -0.15, "mean_nlr": -0.1492, "mse_nlr": 1.2e-05, "sd_nlr": 0.0033,
                         "mean_lr": -0.1489, "mse_lr": 1.3e-05, "sd_lr": 0.0034},
    "theta2_type":      {"true": 1.0, "mean_nlr": 1.0044, "mse_nlr": 0.0034, "sd_nlr": 0.0583,
                         "mean_lr": 1.0032, "mse_lr": 0.0034, "sd_lr": 0.0584},
}


def state_index(x: int, s: int) -> int:
    return x * N_TYPES + s


def build_dgp() -> dict[str, Any]:
    """Bus-engine DGP: states (x, s), deterministic mileage, EV1 shocks."""
    num_states = (X_MAX + 1) * N_TYPES
    num_actions = 2  # 0 = keep, 1 = replace

    # state_encoder for the semi-gradient basis and logit CCP: (x normalized to
    # [0,1], s, x*s). The explicit x*s feature plus elementwise powers gives the
    # paper's "third-order polynomial in s, xt interacted with the binary
    # variables" (s binary => s^k = s, so x^k*s^k = x^k*s). A monotone reparam of
    # x spans the same polynomial space; theta uses raw x in the structural
    # features, so normalization only conditions the basis.
    xs = np.zeros((num_states, 3), dtype=np.float64)
    for x in range(X_MAX + 1):
        for s in range(N_TYPES):
            xn = x / MILEAGE_SCALE
            # Basis/CCP encode the binary type as a {0,1} indicator: same function
            # space as {1,2} but s^k = s collapses, so the third-order polynomial
            # stays the paper's 16 terms and well-conditioned. The structural
            # payoff below uses the paper's actual type value s in {1,2}.
            s_ind = float(s)
            xs[state_index(x, s)] = [xn, s_ind, xn * s_ind]
    xs_j = jnp.asarray(xs)

    def state_encoder(states: jnp.ndarray) -> jnp.ndarray:
        return xs_j[jnp.asarray(states, dtype=jnp.int32)]

    problem = DDCProblem(
        num_states=num_states,
        num_actions=num_actions,
        discount_factor=BETA,
        scale_parameter=1.0,
        state_dim=2,
        state_encoder=state_encoder,
    )

    # Structural features: keep reward = [1, x, s] . theta ; replace = 0.
    feat = np.zeros((num_states, num_actions, 3), dtype=np.float64)
    for x in range(X_MAX + 1):
        for s in range(N_TYPES):
            feat[state_index(x, s), 0, :] = [1.0, float(x), float(s + 1)]  # s in {1, 2}
    utility = ActionDependentReward(jnp.asarray(feat), list(PARAM_NAMES))

    # Deterministic transitions (A, S, S).
    T = np.zeros((num_actions, num_states, num_states), dtype=np.float64)
    for x in range(X_MAX + 1):
        for s in range(N_TYPES):
            i = state_index(x, s)
            T[0, i, state_index(min(x + 1, X_MAX), s)] = 1.0  # keep
            T[1, i, state_index(0, s)] = 1.0                  # replace
    transitions = jnp.asarray(T)

    return {
        "problem": problem,
        "utility": utility,
        "transitions": transitions,
        "num_states": num_states,
    }


def stationary_initial_distribution(problem, transitions, policy) -> jnp.ndarray:
    """Stationary distribution of the policy-induced chain (per-type 50/50 mix)."""
    pol = np.asarray(policy)
    T = np.asarray(transitions)
    P = np.einsum("sa,ast->st", pol, T)  # (S, S)
    # power iteration from a type-balanced start at x=0
    d = np.zeros(problem.num_states)
    d[state_index(0, 0)] = 0.5
    d[state_index(0, 1)] = 0.5
    for _ in range(5000):
        d_next = d @ P
        if np.abs(d_next - d).sum() < 1e-14:
            d = d_next
            break
        d = d_next
    d = np.clip(d, 0.0, None)
    d = d / d.sum()
    return jnp.asarray(d)


def estimator_config(locally_robust: bool, verbose: bool = False) -> TDCCPConfig:
    return TDCCPConfig(
        method="semigradient",
        basis_type="encoded",
        basis_dim=3,             # third-order polynomial in (x, s)
        basis_ridge=1e-5,
        ccp_method="logit",
        ccp_poly_degree=3,
        ccp_use_encoder=True,
        cross_fitting=locally_robust,
        robust_se=locally_robust,
        compute_se=locally_robust,   # MC needs only point estimates; SE sandwich is costly
        n_policy_iterations=1,
        outer_max_iter=500,
        outer_tol=1e-8,
        verbose=verbose,
    )


def run_one_rep(dgp, init_dist, seed: int, locally_robust: bool,
                n_buses: int, n_periods: int) -> np.ndarray:
    panel = simulate_panel_from_policy(
        dgp["problem"], dgp["transitions"], dgp["_policy"], init_dist,
        n_individuals=n_buses, n_periods=n_periods, seed=seed,
    )
    est = TDCCPEstimator(config=estimator_config(locally_robust), seed=seed)
    summary = est.estimate(
        panel=panel, utility=dgp["utility"],
        problem=dgp["problem"], transitions=dgp["transitions"],
    )
    return np.asarray(summary.parameters, dtype=np.float64)


def summarize(estimates: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(PARAM_NAMES):
        col = estimates[:, j]
        err = col - THETA_TRUE[j]
        out[name] = {
            "true": float(THETA_TRUE[j]),
            "mean": float(np.mean(col)),
            "sd": float(np.std(col, ddof=1)) if col.size > 1 else 0.0,
            "bias": float(np.mean(err)),
            "mse": float(np.mean(err**2)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--lr-reps", type=int, default=None,
                        help="Reps for the locally robust column (default: same as --n-reps). "
                             "The robust SE / cross-fitting path is much slower.")
    parser.add_argument("--n-buses", type=int, default=1000)
    parser.add_argument("--n-periods", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20250)
    parser.add_argument("--skip-lr", action="store_true", help="Skip the locally robust column.")
    args = parser.parse_args()
    lr_reps = args.lr_reps if args.lr_reps is not None else args.n_reps

    print("TD-CCP bus-engine Monte Carlo (Adusumilli-Eckardt 2025, Table B.1)")
    print(f"  DGP: X_MAX={X_MAX}, types={N_TYPES}, beta={BETA}, theta_true={THETA_TRUE.tolist()}")
    print(f"  panel: {args.n_buses} buses x {args.n_periods} periods   reps: {args.n_reps}")

    dgp = build_dgp()
    operator = SoftBellmanOperator(dgp["problem"], dgp["transitions"])
    true_reward = dgp["utility"].compute(jnp.asarray(THETA_TRUE))
    truth = value_iteration(operator, true_reward, tol=1e-12, max_iter=20_000)
    dgp["_policy"] = truth.policy
    init_dist = stationary_initial_distribution(dgp["problem"], dgp["transitions"], truth.policy)
    # report the stationary replacement / mileage profile as a sanity check
    repl_share = float(np.asarray(init_dist) @ np.asarray(truth.policy)[:, 1])
    print(f"  stationary replacement share: {repl_share:.4f}")

    columns = [("not_locally_robust", False)]
    if not args.skip_lr:
        columns.append(("locally_robust", True))

    results: dict[str, Any] = {}
    reps_used: dict[str, int] = {}
    t0 = time.time()
    for col_name, lr in columns:
        reps = lr_reps if lr else args.n_reps
        reps_used[col_name] = reps
        estimates = np.zeros((reps, 3))
        for rep in range(reps):
            seed = args.base_seed + rep
            estimates[rep] = run_one_rep(dgp, init_dist, seed, lr, args.n_buses, args.n_periods)
            if (rep + 1) % 25 == 0 or rep == 0:
                print(f"  [{col_name}] rep {rep + 1}/{reps}  ({time.time() - t0:.0f}s)", flush=True)
        results[col_name] = summarize(estimates)

    payload = {
        "estimator": "TD-CCP",
        "paper": "Adusumilli and Eckardt (2025), Table B.1 (bus-engine replacement)",
        "arxiv": "1912.09509",
        "dgp": {
            "model": "bus-engine replacement, deterministic mileage, permanent type",
            "x_max": X_MAX, "n_types": N_TYPES, "beta": BETA,
            "theta_true": THETA_TRUE.tolist(), "param_names": PARAM_NAMES,
            "n_buses": args.n_buses, "n_periods": args.n_periods,
            "n_reps": args.n_reps, "reps_by_column": reps_used,
        },
        "estimator_config": {
            "method": "semigradient", "basis_type": "encoded", "basis_dim": 3,
            "ccp_method": "logit", "ccp_poly_degree": 3,
        },
        "paper_table_b1": PAPER_TABLE_B1,
        "results": results,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Console comparison vs the paper
    print("\n  parameter        true     pkg mean   paper mean   pkg MSE    paper MSE   pkg SD    paper SD")
    for name in PARAM_NAMES:
        r = results["not_locally_robust"][name]
        p = PAPER_TABLE_B1[name]
        print(f"  {name:<16}{r['true']:>7.3f}{r['mean']:>11.4f}{p['mean_nlr']:>12.4f}"
              f"{r['mse']:>11.5f}{p['mse_nlr']:>11.5f}{r['sd']:>10.4f}{p['sd_nlr']:>10.4f}")
    print(f"\n  wrote: {JSON_OUT}")


if __name__ == "__main__":
    main()
