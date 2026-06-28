"""Canonical Ziebart R(s) recovery: linear MCE-IRL vs Neural MCE-IRL on the
deterministic gridworld, the evidence behind the replications-page table.

Metric: affine R^2 of the recovered reward against the true reward across states
(offset+scale invariant, matching "only the shape of the reward is identified").
Plus policy KL from re-solving the soft MDP with the recovered reward. Emits
validation/results/recovery_ziebart_deep.json.

Usage: python examples/ziebart-mce-irl/ziebart_rs_recovery.py --grid-size 12 --seeds 0 1 2
"""
import argparse, json
from pathlib import Path
import numpy as np, jax.numpy as jnp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_recovery_deep import CASES, generate_data, _split, _solve, _make_coord_encoder  # noqa: E402
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator  # noqa: E402
from econirl.estimators.mceirl_neural import MCEIRLNeural  # noqa: E402


def affine_r2(x, y):
    X = np.column_stack([x, np.ones(len(x))]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def solve_kl(R_sa, prob, trans, true_pol):
    _, sol = _solve(jnp.asarray(R_sa), prob, trans)
    ep, tp = np.asarray(sol.policy), np.asarray(true_pol)
    return float((tp * np.log((tp + 1e-12) / (ep + 1e-12))).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reward-num-layers", type=int, default=4)
    ap.add_argument("--reward-hidden-dim", type=int, default=128)
    ap.add_argument("--neural-lr", type=float, default=1e-2)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=Path("validation/results/recovery_ziebart_deep.json"))
    args = ap.parse_args()

    records = []
    for seed in args.seeds:
        prob, trans, rfn, names, theta = CASES["state_only"](args.grid_size, 0.95)
        full, true_pol = generate_data(prob, trans, rfn, theta, 30 * prob.num_states,
                                       4 * args.grid_size, seed)
        train, _ = _split(full, seed)
        true_R = np.asarray(rfn.compute(theta))[:, 0]
        A, Sn = prob.num_actions, prob.num_states

        mce = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=1000,
                              outer_tol=1e-8, inner_solver="hybrid", inner_tol=1e-10,
                              inner_max_iter=10000, use_adam=True, compute_se=False, verbose=False))
        rm = mce.estimate(panel=train, utility=rfn, problem=prob, transitions=trans, true_params=theta)
        R_mce = np.asarray(rfn.compute(rm.parameters))

        nm = MCEIRLNeural(n_states=Sn, n_actions=A, discount=0.95, reward_type="state",
                          reward_hidden_dim=args.reward_hidden_dim,
                          reward_num_layers=args.reward_num_layers, max_epochs=args.max_epochs,
                          lr=args.neural_lr, seed=seed, feature_names=names,
                          state_encoder=_make_coord_encoder(args.grid_size), state_dim=2, verbose=False)
        nm.fit(train, features=np.asarray(rfn.feature_matrix), transitions=np.asarray(trans))
        R_neu = np.broadcast_to(np.asarray(nm.reward_)[:, None], (Sn, A))

        rec = {
            "seed": seed,
            "MCE-IRL": {"reward_affine_r2": affine_r2(true_R, R_mce[:, 0]),
                        "policy_kl": solve_kl(R_mce, prob, trans, true_pol)},
            "Neural MCE-IRL": {"reward_affine_r2": affine_r2(true_R, np.asarray(nm.reward_)),
                               "policy_kl": solve_kl(R_neu, prob, trans, true_pol)},
        }
        records.append(rec)
        print(f"seed {seed}: MCE R2={rec['MCE-IRL']['reward_affine_r2']:+.3f} "
              f"KL={rec['MCE-IRL']['policy_kl']:.4f} | "
              f"Neural R2={rec['Neural MCE-IRL']['reward_affine_r2']:+.3f} "
              f"KL={rec['Neural MCE-IRL']['policy_kl']:.4f}", flush=True)

    def mean(est, k):
        return float(np.mean([r[est][k] for r in records]))
    summary = {est: {"reward_affine_r2": mean(est, "reward_affine_r2"),
                     "policy_kl": mean(est, "policy_kl")}
               for est in ["MCE-IRL", "Neural MCE-IRL"]}
    meta = {"task": "ziebart R(s) recovery", "grid_size": args.grid_size, "case": "state_only",
            "seeds": args.seeds, "reward_num_layers": args.reward_num_layers,
            "reward_hidden_dim": args.reward_hidden_dim, "neural_lr": args.neural_lr,
            "metric": "affine R^2 of recovered reward vs true reward across states; policy KL(true||est)"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, indent=2) + "\n")
    print(f"\nmean: MCE R2={summary['MCE-IRL']['reward_affine_r2']:+.3f} "
          f"Neural R2={summary['Neural MCE-IRL']['reward_affine_r2']:+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
