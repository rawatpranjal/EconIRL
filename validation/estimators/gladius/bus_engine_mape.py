#!/usr/bin/env python3
"""GLADIUS reward-MAPE replication on the Kang-Yoganarasimhan-Jain (2025) bus engine.

Reproduces Table 2 of "Gradients can train reward models: An Empirical Risk
Minimization Approach for Offline Inverse RL and Dynamic Discrete Choice Model"
(arXiv 2502.14131). GLADIUS learns a neural reward from expert trajectories and
is scored by the mean absolute percentage error (MAPE) of the recovered reward
against the truth, averaged over the state-action pairs visited in the data.

DGP (Section 7.1): a Rust-style bus-engine replacement problem.
  state: mileage x in {1, ..., 20}, every trajectory starts at x = 1.
  actions: maintain (a=0) and replace (a=1).
  reward: maintain pays -theta0 * x (theta0 = 1), replace pays -theta1 (theta1 = 5).
  transitions: maintain advances mileage by 1..4 each with prob 1/4 (capped at 20);
               replace resets mileage to 1.
  discount beta = 0.95. Type-1 EV shocks, 100 periods per trajectory.

MAPE = mean over observed (s, a) rows of |r_hat - r| / |r| * 100. No reward is
near zero (maintain <= -1, replace = -5), so the ratio is well defined.

Usage:
    PYTHONPATH=src:. python validation/estimators/gladius/bus_engine_mape.py --probe
    PYTHONPATH=src:. python validation/estimators/gladius/bus_engine_mape.py --n-traj 50 250 500 1000
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
JSON_OUT = ROOT / "validation" / "results" / "gladius_bus_engine_mape.json"

for path in (HERE.parent, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from econirl.core.types import DDCProblem  # noqa: E402
from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.estimation.gladius import GLADIUSConfig, GLADIUSEstimator  # noqa: E402
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from econirl.simulation.synthetic import simulate_panel_from_policy  # noqa: E402
from validation.known_truth import to_jsonable  # noqa: E402

# Paper Section 7.1 DGP -------------------------------------------------------
THETA0 = 1.0          # maintenance cost per unit mileage
THETA1 = 5.0          # fixed replacement cost
BETA = 0.95
MAX_MILEAGE = 20      # mileage in {1, ..., 20}
N_PERIODS = 100
# Paper Table 2 GLADIUS MAPE (%) by number of trajectories
PAPER_TABLE2_GLADIUS = {50: 3.44, 250: 0.84, 500: 0.55, 1000: 0.52, 2500: 0.13, 5000: 0.12}
PAPER_TABLE2_RUST = {50: 3.62, 250: 1.37, 500: 0.90, 1000: 0.71, 2500: 0.68, 5000: 0.40}


def build_dgp() -> dict[str, Any]:
    """Bus-engine DGP. State index i in 0..19 maps to mileage i+1."""
    S = MAX_MILEAGE
    A = 2  # 0 = maintain, 1 = replace
    mileage = np.arange(1, S + 1, dtype=np.float64)  # 1..20

    # True reward: maintain -> -theta0 * x ; replace -> -theta1.
    r_true = np.zeros((S, A), dtype=np.float64)
    r_true[:, 0] = -THETA0 * mileage
    r_true[:, 1] = -THETA1

    # Transitions (A, S, S).
    T = np.zeros((A, S, S), dtype=np.float64)
    for i in range(S):
        for k in (1, 2, 3, 4):
            j = min(i + k, S - 1)          # mileage advances by k, capped at 20
            T[0, i, j] += 0.25
        T[1, i, 0] = 1.0                   # replace -> mileage 1 (index 0)

    # state encoder: normalized mileage for the neural Q.
    enc = (mileage / MAX_MILEAGE).reshape(S, 1)
    enc_j = jnp.asarray(enc)

    def state_encoder(states: jnp.ndarray) -> jnp.ndarray:
        return enc_j[jnp.asarray(states, dtype=jnp.int32)]

    problem = DDCProblem(
        num_states=S, num_actions=A, discount_factor=BETA,
        scale_parameter=1.0, state_dim=1, state_encoder=state_encoder,
    )

    # Placeholder utility for GLADIUS's (unused-here) feature projection:
    # maintain feature = mileage, replace feature = constant.
    feat = np.zeros((S, A, 2), dtype=np.float64)
    feat[:, 0, 0] = mileage
    feat[:, 1, 1] = 1.0
    utility = ActionDependentReward(jnp.asarray(feat), ["maintain_mileage", "replace_const"])

    return {
        "problem": problem, "utility": utility,
        "transitions": jnp.asarray(T), "r_true": r_true, "mileage": mileage,
        "num_states": S, "num_actions": A,
    }


def gladius_config(anchor: bool) -> GLADIUSConfig:
    cfg = GLADIUSConfig(
        q_hidden_dim=64, q_num_layers=2, v_hidden_dim=64, v_num_layers=2,
        q_lr=1e-3, v_lr=1e-3, max_epochs=400, batch_size=512,
        patience=60, alternating_updates=True, compute_se=False, verbose=False,
    )
    if anchor:
        # Pin the reward level via the known fixed replacement cost.
        cfg.anchor_action = 1
        cfg.anchor_rewards = tuple([-THETA1] * MAX_MILEAGE)
        cfg.anchor_bellman_loss = True
    return cfg


def data_visited_mape(r_hat: np.ndarray, r_true: np.ndarray, panel) -> float:
    """Mean abs % error over observed (s, a) rows (the paper's metric)."""
    states = np.concatenate([np.asarray(t.states) for t in panel.trajectories])
    actions = np.concatenate([np.asarray(t.actions) for t in panel.trajectories])
    rt = r_true[states, actions]
    rh = r_hat[states, actions]
    return float(np.mean(np.abs(rh - rt) / np.abs(rt)) * 100.0)


def run_one(dgp, policy, init_dist, n_traj: int, anchor: bool, seed: int):
    panel = simulate_panel_from_policy(
        dgp["problem"], dgp["transitions"], policy, init_dist,
        n_individuals=n_traj, n_periods=N_PERIODS, seed=seed,
    )
    est = GLADIUSEstimator(config=gladius_config(anchor))
    summary = est.estimate(panel=panel, utility=dgp["utility"],
                           problem=dgp["problem"], transitions=dgp["transitions"])
    reward_table = np.asarray(summary.metadata["reward_table"], dtype=np.float64)
    mape = data_visited_mape(reward_table, dgp["r_true"], panel)
    return mape, reward_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="One small fit, print diagnostics.")
    parser.add_argument("--n-traj", type=int, nargs="+", default=[50, 250, 500, 1000])
    parser.add_argument("--anchor", action="store_true", help="Pin level via known replace cost.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reps", type=int, default=1, help="Reps (varying seed) averaged per N.")
    args = parser.parse_args()

    print("GLADIUS bus-engine reward MAPE (Kang-Yoganarasimhan-Jain 2025, Table 2)")
    print(f"  DGP: mileage 1..{MAX_MILEAGE}, theta0={THETA0}, theta1={THETA1}, beta={BETA}")
    print(f"  anchor (known replace cost): {args.anchor}")

    dgp = build_dgp()
    operator = SoftBellmanOperator(dgp["problem"], dgp["transitions"])
    truth = value_iteration(operator, jnp.asarray(dgp["r_true"]), tol=1e-12, max_iter=20_000)
    policy = truth.policy
    init_dist = jnp.zeros(dgp["num_states"]).at[0].set(1.0)  # all start at mileage 1

    if args.probe:
        t0 = time.time()
        mape, rhat = run_one(dgp, policy, init_dist, 250, args.anchor, args.seed)
        print(f"  probe N=250: MAPE={mape:.3f}%  (paper GLADIUS {PAPER_TABLE2_GLADIUS[250]}%)  [{time.time()-t0:.0f}s]")
        print("  recovered maintain reward (mileage 1..10) vs true:")
        for i in range(10):
            print(f"    mileage {i+1:2d}: r_hat={rhat[i,0]:+.3f}  true={dgp['r_true'][i,0]:+.1f}")
        print(f"  recovered replace reward r_hat[0,1]={rhat[0,1]:+.3f} true=-5")
        return

    results: dict[str, Any] = {}
    t0 = time.time()
    for n in args.n_traj:
        mapes = np.array([
            run_one(dgp, policy, init_dist, n, args.anchor, args.seed + r)[0]
            for r in range(args.reps)
        ])
        results[str(n)] = {
            "mape_mean": float(mapes.mean()), "mape_std": float(mapes.std()),
            "mape_min": float(mapes.min()), "mapes": mapes.tolist(), "reps": args.reps,
            "paper_gladius": PAPER_TABLE2_GLADIUS.get(n),
            "paper_rust_oracle": PAPER_TABLE2_RUST.get(n),
        }
        print(f"  N={n:5d}  pkg MAPE mean={mapes.mean():6.2f}% (std {mapes.std():5.2f}, "
              f"min {mapes.min():5.2f}, reps {args.reps})   paper GLADIUS={PAPER_TABLE2_GLADIUS.get(n)}%   "
              f"Rust={PAPER_TABLE2_RUST.get(n)}%   ({time.time()-t0:.0f}s)")

    payload = {
        "estimator": "GLADIUS", "paper": "Kang, Yoganarasimhan, Jain (2025), Table 2",
        "arxiv": "2502.14131",
        "dgp": {"model": "bus-engine replacement", "max_mileage": MAX_MILEAGE,
                "theta0": THETA0, "theta1": THETA1, "beta": BETA, "n_periods": N_PERIODS,
                "anchor": args.anchor, "seed": args.seed},
        "metric": "data-visited reward MAPE (%)",
        "results": results,
        "paper_table2_gladius": PAPER_TABLE2_GLADIUS,
        "paper_table2_rust_oracle": PAPER_TABLE2_RUST,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  wrote: {JSON_OUT}")


if __name__ == "__main__":
    main()
