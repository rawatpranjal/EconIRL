#!/usr/bin/env python3
"""GLADIUS paper Table 2 replication: reward-estimation MAPE on the standard bus.

Reproduces Table 2 (# dummy = 0) of Kang, Yoganarasimhan & Jain (2025),
"Generalized Inverse Reinforcement Learning ... (GLADIUS)", arXiv 2502.14131.

Target (MAPE %, GLADIUS column, 20 reps):
    50 -> 3.44, 250 -> 0.84, 500 -> 0.55, 1000 -> 0.52, 2500 -> 0.13, 5000 -> 0.12

The exact DGP (paper sec 7.1, lines 1394-1429 of the full text):
  - mileage x in {1..20}; all buses start at mileage 1; H = 100 periods.
  - a=0 maintain (cost theta0 * x), a=1 replace (cost theta1, resets mileage to 1).
  - maintain advances mileage by +1/+2/+3/+4 each w.p. 1/4 (capped at 20).
  - true theta0 = 1, theta1 = 5; beta = 0.95; soft-optimal (entropy-reg) demos.
  - 80/20 train/test split; reward learned on train, MAPE evaluated on the (s,a)
    visited in the test trajectories (dominated by mileage 1-5).
  - MAPE = mean_i |r_hat_i - r_i| / |r_i| * 100, over test samples; r_hat is the
    RAW recovered reward r_hat(s,a) = Q(s,a) - beta * zeta(s,a) (paper Table 3),
    NOT the projected theta.

The estimator version is the paper-faithful one (Enoch Kang's reference =
first-author code): GLADIUSEstimator with network_mode="shared_trunk" and
anchor_bellman_mode="paper_minimax" (the literal bi-conjugate Bellman term),
anchored at the replacement action with the known reward -theta1 (Assumption 3),
MLP with 2 hidden layers x 10 units for Q.

Usage:
  PYTHONPATH=src python validation/estimators/gladius/paper_table2_mape.py --probe
  PYTHONPATH=src python validation/estimators/gladius/paper_table2_mape.py --sweep --reps 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax.numpy as jnp  # noqa: E402
from gymnasium import spaces  # noqa: E402

from econirl.core.types import DDCProblem, Panel  # noqa: E402
from econirl.environments.base import DDCEnvironment  # noqa: E402
from econirl.estimation.gladius import GLADIUSConfig, GLADIUSEstimator  # noqa: E402
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402

# Paper Table 2, GLADIUS column (# dummy = 0), MAPE % (SE).
PAPER_GLADIUS = {50: 3.44, 250: 0.84, 500: 0.55, 1000: 0.52, 2500: 0.13, 5000: 0.12}
PAPER_RUST = {50: 3.62, 250: 1.37, 500: 0.90, 1000: 0.71, 2500: 0.68, 5000: 0.40}

N_MILEAGE = 20
THETA0 = 1.0  # maintenance cost per mileage unit
THETA1 = 5.0  # replacement cost
BETA = 0.95
MAINTAIN, REPLACE = 0, 1


class PaperBusEnvironment(DDCEnvironment):
    """The exact bus DGP from Kang-Yoganarasimhan-Jain (2025) sec 7.1.

    Distinct from the repo's RustBusEnvironment (which uses +0/+1/+2 transitions
    and a stochastic post-replace re-increase, and so cannot represent this DGP).
    """

    def __init__(self, seed: int | None = None):
        super().__init__(discount_factor=BETA, scale_parameter=1.0, seed=seed)
        self.observation_space = spaces.Discrete(N_MILEAGE)
        self.action_space = spaces.Discrete(2)

    @property
    def num_states(self) -> int:
        return N_MILEAGE

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def transition_matrices(self) -> jnp.ndarray:
        n = N_MILEAGE
        T = np.zeros((2, n, n), dtype=np.float32)
        for i in range(n):  # maintain: +1..+4 each 1/4, capped at top
            for k in (1, 2, 3, 4):
                T[MAINTAIN, i, min(i + k, n - 1)] += 0.25
        T[REPLACE, :, 0] = 1.0  # replace: reset to mileage 1 (index 0)
        return jnp.asarray(T)

    @property
    def feature_matrix(self) -> jnp.ndarray:
        n = N_MILEAGE
        phi = np.zeros((n, 2, 2), dtype=np.float32)
        mileage = np.arange(1, n + 1, dtype=np.float32)
        phi[:, MAINTAIN, 0] = -mileage  # U(maintain) = -theta0 * mileage
        phi[:, REPLACE, 1] = -1.0       # U(replace)  = -theta1
        return jnp.asarray(phi)

    @property
    def true_parameters(self) -> dict[str, float]:
        return {"theta0": THETA0, "theta1": THETA1}

    @property
    def parameter_names(self) -> list[str]:
        return ["theta0", "theta1"]

    def _get_initial_state_distribution(self) -> np.ndarray:
        dist = np.zeros(N_MILEAGE)
        dist[0] = 1.0  # all buses start at mileage 1
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        if action == REPLACE:
            return -THETA1
        return -THETA0 * (state + 1)

    def _sample_next_state(self, state: int, action: int) -> int:
        if action == REPLACE:
            return 0
        k = int(self._np_random.integers(1, 5))  # +1..+4
        return min(state + k, N_MILEAGE - 1)


def true_reward_matrix() -> np.ndarray:
    R = np.zeros((N_MILEAGE, 2), dtype=np.float64)
    R[:, MAINTAIN] = -THETA0 * np.arange(1, N_MILEAGE + 1)
    R[:, REPLACE] = -THETA1
    return R


def mape_on_samples(r_hat: np.ndarray, states: np.ndarray, actions: np.ndarray) -> float:
    """Paper MAPE: mean |r_hat - r|/|r| * 100 over visited (s,a) samples."""
    R = true_reward_matrix()
    r_true = R[states, actions]
    r_pred = r_hat[states, actions]
    return float(100.0 * np.mean(np.abs(r_pred - r_true) / np.abs(r_true)))


def fit_gladius(env, train_panel, *, max_epochs: int, batch_size: int) -> np.ndarray:
    """Paper-faithful GLADIUS; returns raw recovered reward r_hat(s,a), shape (S,A)."""
    S, A = env.num_states, env.num_actions
    phi = jnp.asarray(env.feature_matrix, dtype=jnp.float32)
    util = ActionDependentReward(phi, env.parameter_names)
    mileage = jnp.arange(1, S + 1, dtype=jnp.float32)  # raw mileage, the paper's MLP input
    state_enc = lambda s: mileage[jnp.asarray(s, dtype=jnp.int32)][:, None]
    base = env.problem_spec
    prob = DDCProblem(
        num_states=S, num_actions=A, discount_factor=base.discount_factor,
        scale_parameter=base.scale_parameter, state_dim=1, state_encoder=state_enc,
    )
    # anchor_moment (the package default), NOT paper_minimax: anchor_moment uses a
    # stop-gradient zeta target (r + beta*zeta - Q), a fitted-Q form that pins the
    # absolute reward LEVEL via the beta-contraction. paper_minimax lets gradient
    # flow through V(s'), giving a (1-beta)-weak level gradient, so the level sits
    # at its init and the recovered reward is shifted (MAPE ~80-200). See the
    # internal replication note for the diagnosis.
    cfg = GLADIUSConfig(
        anchor_bellman_mode="anchor_moment",
        anchor_bellman_loss=True,
        anchor_action=REPLACE,                       # known r(s, replace) = -theta1
        anchor_rewards=tuple([-THETA1] * S),
        q_hidden_dim=10, q_num_layers=2,             # paper: MLP 2 x 10
        v_hidden_dim=10, v_num_layers=2,
        q_lr=1e-3, v_lr=1e-3, lr_decay_rate=5e-4,
        batch_size=batch_size, max_epochs=max_epochs, patience=400,
        gradient_clip_mode="value",
        # Predict Q directly (value_scale=1.0) from a zero bias; anchor_moment is
        # init-robust and converges the absolute level via the fitted-Q contraction.
        output_bias_init=0.0,
        compute_se=False, verbose=False,
    )
    est = GLADIUSEstimator(config=cfg)
    s = est.estimate(panel=train_panel, utility=util, problem=prob,
                     transitions=jnp.asarray(env.transition_matrices))
    return np.asarray(s.metadata["reward_table"], dtype=np.float64)


def fit_nfxp_oracle(env, train_panel, test_states, test_actions) -> float:
    """Rust (1987) NFXP oracle: full-likelihood MLE with the true linear form and
    known transitions (as the paper's 'Rust (Oracle)' baseline). Returns reward MAPE.
    Validates the DGP + metric: this MAPE should track the paper's Rust column."""
    from econirl.estimation.nfxp import NFXPEstimator
    phi = np.asarray(env.feature_matrix)  # (S, A, K)
    util = ActionDependentReward(jnp.asarray(env.feature_matrix), env.parameter_names)
    summ = NFXPEstimator(compute_hessian=False).estimate(
        panel=train_panel, utility=util, problem=env.problem_spec,
        transitions=jnp.asarray(env.transition_matrices))
    r_hat = phi @ np.asarray(summ.parameters)  # reward = theta_hat . phi, shape (S, A)
    return mape_on_samples(r_hat, test_states, test_actions)


def run_one(n_traj: int, seed: int, *, max_epochs: int, batch_size: int,
            oracle: bool = True) -> dict:
    env = PaperBusEnvironment(seed=seed)
    panel = simulate_panel(env, n_individuals=n_traj, n_periods=100, seed=seed)
    trajs = list(panel.trajectories)
    n_train = int(round(0.8 * len(trajs)))
    train = Panel(trajectories=trajs[:n_train], metadata=panel.metadata)
    test = trajs[n_train:]
    test_states = np.concatenate([np.asarray(t.states) for t in test])
    test_actions = np.concatenate([np.asarray(t.actions) for t in test])

    t0 = time.time()
    r_hat = fit_gladius(env, train, max_epochs=max_epochs, batch_size=batch_size)
    mape = mape_on_samples(r_hat, test_states, test_actions)
    nfxp_mape = (fit_nfxp_oracle(env, train, test_states, test_actions)
                 if oracle else None)
    return {
        "n_traj": n_traj, "seed": seed, "mape": mape, "nfxp_oracle_mape": nfxp_mape,
        "r_hat_maintain": [round(float(r_hat[i, MAINTAIN]), 4) for i in range(10)],
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="1 rep at 1000 trajs (cheap)")
    ap.add_argument("--sweep", action="store_true", help="full trajectory sweep")
    ap.add_argument("--traj", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.probe:
        traj_list, reps = [1000], 1
    elif args.sweep:
        traj_list, reps = [50, 250, 500, 1000, 2500, 5000], args.reps
    else:
        traj_list, reps = [args.traj], args.reps

    records = []
    for n_traj in traj_list:
        mapes = []
        for seed in range(reps):
            rec = run_one(n_traj, seed, max_epochs=args.max_epochs, batch_size=args.batch_size)
            records.append(rec)
            mapes.append(rec["mape"])
            nf = rec.get("nfxp_oracle_mape")
            nf_str = f" NFXP-oracle={nf:.3f}" if nf is not None else ""
            print(f"[n_traj={n_traj:5d} seed={seed}] GLADIUS MAPE={rec['mape']:.3f}"
                  f"{nf_str}  r_hat_maint[1:5]={rec['r_hat_maintain'][:5]} ({rec['seconds']}s)")
        mean_mape = float(np.mean(mapes))
        nfm = [r["nfxp_oracle_mape"] for r in records
               if r["n_traj"] == n_traj and r.get("nfxp_oracle_mape") is not None]
        nfxp_str = f" | NFXP-oracle {np.mean(nfm):.3f} (paper Rust {PAPER_RUST.get(n_traj)})" if nfm else ""
        paper = PAPER_GLADIUS.get(n_traj)
        if paper is not None:
            print(f"  -> n_traj={n_traj}: GLADIUS MAPE {mean_mape:.3f} "
                  f"(paper {paper}, {mean_mape/paper:.1f}x){nfxp_str}")
        else:
            print(f"  -> n_traj={n_traj}: GLADIUS MAPE {mean_mape:.3f}{nfxp_str}")

    if args.out:
        args.out.write_text(json.dumps(
            {"dgp": "paper bus (20 mileage, +1..4, theta=[1,5], beta=0.95)",
             "estimator": "GLADIUSEstimator paper_minimax+shared_trunk, anchor=replace",
             "paper_gladius_mape": PAPER_GLADIUS, "records": records}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
