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
PAPER_GLADIUS_SE = {50: 1.28, 250: 0.51, 500: 0.20, 1000: 0.22, 2500: 0.06, 5000: 0.06}
PAPER_RUST = {50: 3.62, 250: 1.37, 500: 0.90, 1000: 0.71, 2500: 0.68, 5000: 0.40}
TARGET_SIZES = (50, 250, 500, 1000, 2500, 5000)
TARGET_REPETITIONS = 20

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
        phi[:, REPLACE, 1] = -1.0  # U(replace)  = -theta1
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


def fit_gladius(
    env,
    train_panel,
    *,
    max_epochs: int,
    batch_size: int,
    oracle_states: np.ndarray,
    oracle_actions: np.ndarray,
    anchor: str = "paper_minimax",
) -> tuple[np.ndarray, dict]:
    """Fit GLADIUS with the authors' simulation-only best-MAPE epoch rule."""
    S, A = env.num_states, env.num_actions
    phi = jnp.asarray(env.feature_matrix, dtype=jnp.float32)
    util = ActionDependentReward(phi, env.parameter_names)
    mileage = jnp.arange(1, S + 1, dtype=jnp.float32)  # raw mileage, the paper's MLP input

    def state_enc(states):
        return mileage[jnp.asarray(states, dtype=jnp.int32)][:, None]

    base = env.problem_spec
    prob = DDCProblem(
        num_states=S,
        num_actions=A,
        discount_factor=base.discount_factor,
        scale_parameter=base.scale_parameter,
        state_dim=1,
        state_encoder=state_enc,
    )
    # anchor selects the Bellman target. paper_minimax (default) is the paper's
    # literal bi-conjugate term and the faithful headline: it lets gradient flow
    # through V(s'), a (1-beta)-weak level gradient, so the absolute reward LEVEL
    # sits near its init and the recovered reward is shifted (the documented level
    # gap). anchor_moment is the fitted-Q diagnostic, a stop-gradient zeta target
    # (r + beta*zeta - Q) that pins the level via the beta-contraction but departs
    # from the paper's Bellman term. See the internal replication note.
    cfg = GLADIUSConfig(
        anchor_bellman_mode=anchor,
        anchor_bellman_loss=True,
        anchor_action=REPLACE,  # known r(s, replace) = -theta1
        anchor_rewards=tuple([-THETA1] * S),
        anchor_level_calibration=False,
        q_hidden_dim=10,
        q_num_layers=2,  # paper: MLP 2 x 10
        v_hidden_dim=10,
        v_num_layers=2,
        q_lr=1e-3,
        v_lr=1e-3,
        lr_decay_rate=5e-4,
        batch_size=batch_size,
        max_epochs=max_epochs,
        # The first-author training loop runs the frozen epoch budget without
        # early stopping. Keep the package's stopping rule out of the exact
        # replication path so a noisy minimax epoch cannot truncate training.
        patience=max_epochs + 1,
        gradient_clip_mode="value",
        # Predict Q directly (value_scale=1.0) from a zero bias; anchor_moment is
        # init-robust and converges the absolute level via the fitted-Q contraction.
        output_bias_init=0.0,
        compute_se=False,
        _oracle_reward_table=tuple(
            tuple(float(value) for value in row) for row in true_reward_matrix()
        ),
        _oracle_eval_states=tuple(int(value) for value in oracle_states),
        _oracle_eval_actions=tuple(int(value) for value in oracle_actions),
        verbose=False,
    )
    est = GLADIUSEstimator(config=cfg)
    result = est.estimate(
        panel=train_panel,
        utility=util,
        problem=prob,
        transitions=jnp.asarray(env.transition_matrices),
    )
    loss_history = list(result.metadata.get("loss_history", []))
    return np.asarray(result.metadata["reward_table"], dtype=np.float64), {
        "converged": bool(result.converged),
        "num_epochs": int(result.num_iterations),
        "termination": str(result.convergence_message),
        "final_loss": float(loss_history[-1]) if loss_history else None,
        "lr_decay_unit": result.metadata.get("lr_decay_unit"),
        "oracle_selected": bool(result.metadata.get("oracle_selected")),
        "oracle_best_epoch": result.metadata.get("oracle_best_epoch"),
        "oracle_best_mape": result.metadata.get("oracle_best_mape"),
    }


def fit_nfxp_oracle(env, train_panel, test_states, test_actions) -> float:
    """Rust (1987) NFXP oracle: full-likelihood MLE with the true linear form and
    known transitions (as the paper's 'Rust (Oracle)' baseline). Returns reward MAPE.
    Validates the DGP + metric: this MAPE should track the paper's Rust column."""
    from econirl.estimation.nfxp import NFXPEstimator

    phi = np.asarray(env.feature_matrix)  # (S, A, K)
    util = ActionDependentReward(jnp.asarray(env.feature_matrix), env.parameter_names)
    summ = NFXPEstimator(compute_hessian=False).estimate(
        panel=train_panel,
        utility=util,
        problem=env.problem_spec,
        transitions=jnp.asarray(env.transition_matrices),
    )
    r_hat = phi @ np.asarray(summ.parameters)  # reward = theta_hat . phi, shape (S, A)
    return mape_on_samples(r_hat, test_states, test_actions)


def run_one(
    n_traj: int,
    seed: int,
    *,
    max_epochs: int,
    batch_size: int,
    oracle: bool = True,
    anchor: str = "paper_minimax",
) -> dict:
    env = PaperBusEnvironment(seed=seed)
    panel = simulate_panel(env, n_individuals=n_traj, n_periods=100, seed=seed)
    trajs = list(panel.trajectories)
    n_train = int(round(0.8 * len(trajs)))
    train = Panel(trajectories=trajs[:n_train], metadata=panel.metadata)
    test = trajs[n_train:]
    test_states = np.concatenate([np.asarray(t.states) for t in test])
    test_actions = np.concatenate([np.asarray(t.actions) for t in test])

    t0 = time.time()
    r_hat, optimization = fit_gladius(
        env,
        train,
        max_epochs=max_epochs,
        batch_size=batch_size,
        oracle_states=test_states,
        oracle_actions=test_actions,
        anchor=anchor,
    )
    mape = mape_on_samples(r_hat, test_states, test_actions)
    nfxp_mape = fit_nfxp_oracle(env, train, test_states, test_actions) if oracle else None
    return {
        "n_traj": n_traj,
        "seed": seed,
        "anchor": anchor,
        "mape": mape,
        "nfxp_oracle_mape": nfxp_mape,
        "optimization": optimization,
        "r_hat_maintain": [round(float(r_hat[i, MAINTAIN]), 4) for i in range(10)],
        "seconds": round(time.time() - t0, 1),
    }


def summarize_records(records: list[dict]) -> dict:
    """Apply the frozen Table 2 acceptance rule to merged replication records."""
    cells = []
    for n_traj in TARGET_SIZES:
        cell_records = [record for record in records if record["n_traj"] == n_traj]
        mapes = np.asarray([record["mape"] for record in cell_records], dtype=float)
        nfxp = np.asarray(
            [
                record["nfxp_oracle_mape"]
                for record in cell_records
                if record.get("nfxp_oracle_mape") is not None
            ],
            dtype=float,
        )
        paper_mean = PAPER_GLADIUS[n_traj]
        threshold = paper_mean + 2.0 * PAPER_GLADIUS_SE[n_traj]
        cells.append(
            {
                "n_traj": n_traj,
                "repetitions": len(cell_records),
                "seeds": sorted(int(record["seed"]) for record in cell_records),
                "mean_mape": float(np.mean(mapes)) if len(mapes) else None,
                "standard_error": (
                    float(np.std(mapes, ddof=1) / np.sqrt(len(mapes))) if len(mapes) > 1 else None
                ),
                "paper_mean_mape": paper_mean,
                "paper_standard_error": PAPER_GLADIUS_SE[n_traj],
                "acceptance_upper_bound": threshold,
                "within_paper_mean_plus_2se": bool(len(mapes) and np.mean(mapes) <= threshold),
                "nfxp_oracle_mean_mape": float(np.mean(nfxp)) if len(nfxp) else None,
            }
        )
    by_size = {cell["n_traj"]: cell for cell in cells}
    complete = all(
        cell["repetitions"] == TARGET_REPETITIONS
        and cell["seeds"] == list(range(TARGET_REPETITIONS))
        for cell in cells
    )
    post_250 = bool(
        complete
        and all(
            by_size[n_traj]["mean_mape"] <= by_size[250]["mean_mape"]
            for n_traj in TARGET_SIZES
            if n_traj > 250
        )
    )
    gates = {
        "full_6x20_design": complete,
        "all_cells_within_paper_mean_plus_2se": all(
            cell["within_paper_mean_plus_2se"] for cell in cells
        ),
        "post_250_non_deterioration": post_250,
        "oracle_epoch_rule_disclosed": all(
            record.get("optimization", {}).get("oracle_selected") is True for record in records
        ),
    }
    return {"cells": cells, "gates": gates, "all_passed": all(gates.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="1 rep at 1000 trajs (cheap)")
    ap.add_argument("--sweep", action="store_true", help="full trajectory sweep")
    ap.add_argument(
        "--sizes", type=str, default=None, help="comma list of n_traj, overrides --sweep"
    )
    ap.add_argument("--traj", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=800)
    ap.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help=(
            "optional approximate optimizer-update budget per cell; converts "
            "to an epoch count using the frozen 80%% training split"
        ),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--merge-shard", type=Path, action="append", default=None)
    ap.add_argument(
        "--anchor",
        choices=["paper_minimax", "anchor_moment"],
        default="paper_minimax",
        help="paper_minimax = faithful headline; anchor_moment = fitted-Q diagnostic",
    )
    args = ap.parse_args()

    if args.merge_shard:
        merged: dict[tuple[int, int], dict] = {}
        for shard in args.merge_shard:
            payload = json.loads(shard.read_text(encoding="utf-8"))
            for record in payload["records"]:
                key = (int(record["n_traj"]), int(record["seed"]))
                if key in merged and merged[key] != record:
                    raise ValueError(f"conflicting Table 2 shard record {key}")
                merged[key] = record
        records = [merged[key] for key in sorted(merged)]
        summary = summarize_records(records)
        output = args.out or ROOT / "validation" / "results" / "gladius_paper_table2.json"
        output.write_text(
            json.dumps(
                {
                    "dgp": "paper bus (20 mileage, +1..4, theta=[1,5], beta=0.95)",
                    "estimator": "GLADIUSEstimator paper_minimax+shared_trunk, anchor=replace",
                    "selection_boundary": (
                        "simulation-only best epoch selected by true held-out reward MAPE, "
                        "matching the checked-in author experiment; not used by public fit"
                    ),
                    **summary,
                    "records": records,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}")
        if not summary["all_passed"]:
            raise SystemExit("GLADIUS Table 2 qualification gates failed")
        return

    if args.probe:
        traj_list, reps = [1000], 1
    elif args.sizes:
        traj_list, reps = [int(x) for x in args.sizes.split(",")], args.reps
    elif args.sweep:
        traj_list, reps = list(TARGET_SIZES), args.reps
    else:
        traj_list, reps = [args.traj], args.reps

    records = []
    for n_traj in traj_list:
        mapes = []
        train_observations = int(round(0.8 * n_traj)) * 100
        steps_per_epoch = max(int(np.ceil(train_observations / args.batch_size)), 1)
        cell_epochs = (
            args.max_epochs
            if args.max_updates is None
            else min(
                args.max_epochs,
                max(int(np.ceil(args.max_updates / steps_per_epoch)), 1),
            )
        )
        for seed in range(args.start_seed, args.start_seed + reps):
            rec = run_one(
                n_traj,
                seed,
                max_epochs=cell_epochs,
                batch_size=args.batch_size,
                anchor=args.anchor,
            )
            records.append(rec)
            mapes.append(rec["mape"])
            nf = rec.get("nfxp_oracle_mape")
            nf_str = f" NFXP-oracle={nf:.3f}" if nf is not None else ""
            print(
                f"[n_traj={n_traj:5d} seed={seed}] GLADIUS MAPE={rec['mape']:.3f}"
                f"{nf_str}  r_hat_maint[1:5]={rec['r_hat_maintain'][:5]} ({rec['seconds']}s)"
            )
        mean_mape = float(np.mean(mapes))
        nfm = [
            r["nfxp_oracle_mape"]
            for r in records
            if r["n_traj"] == n_traj and r.get("nfxp_oracle_mape") is not None
        ]
        nfxp_str = (
            f" | NFXP-oracle {np.mean(nfm):.3f} (paper Rust {PAPER_RUST.get(n_traj)})"
            if nfm
            else ""
        )
        paper = PAPER_GLADIUS.get(n_traj)
        if paper is not None:
            upper = paper + 2.0 * PAPER_GLADIUS_SE[n_traj]
            passed = mean_mape <= upper
            print(
                f"  -> n_traj={n_traj}: GLADIUS MAPE {mean_mape:.3f} "
                f"(paper {paper}, 2SE upper {upper:.3f}, "
                f"{'PASS' if passed else 'FAIL'}){nfxp_str}"
            )
        else:
            print(f"  -> n_traj={n_traj}: GLADIUS MAPE {mean_mape:.3f}{nfxp_str}")

    if args.out:
        summary = summarize_records(records)
        args.out.write_text(
            json.dumps(
                {
                    "dgp": "paper bus (20 mileage, +1..4, theta=[1,5], beta=0.95)",
                    "estimator": f"GLADIUSEstimator {args.anchor}+shared_trunk, anchor=replace",
                    "anchor": args.anchor,
                    "paper_gladius_mape": PAPER_GLADIUS,
                    "paper_gladius_se": PAPER_GLADIUS_SE,
                    "start_seed": args.start_seed,
                    "repetitions": reps,
                    "max_updates": args.max_updates,
                    "max_epochs": args.max_epochs,
                    "selection_boundary": (
                        "simulation-only best epoch selected by true held-out reward MAPE, "
                        "matching the checked-in author experiment; not used by public fit"
                    ),
                    **summary,
                    "records": records,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
