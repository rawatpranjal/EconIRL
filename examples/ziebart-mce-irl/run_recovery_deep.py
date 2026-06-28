"""Deep recovery: MCE-IRL vs Neural MCE-IRL on Ziebart's gridworld problem.

Replication track (not the simulation study). Ziebart's original taxi-route data
is withheld, so the controlled gridworld stands in for the soft-Bellman maximum
causal entropy recovery problem. We run the two estimators the user named, linear
MCE-IRL and Neural MCE-IRL, across the three reward parameterizations from
run_gridworld.py and report the full set of recovery metrics.

Ziebart's reward is R(s), linear in state features, on a deterministic MDP, so
the default case is state_only (the action-dependent cases remain available via
--cases but are off by default).

Neural MCE-IRL gets a (row, col) coordinate state encoder so its reward network
can represent 2-D grid structure (the default 1-D index encoder cannot). Its
learned reward is projected onto the same linear features as MCE-IRL, so the
reward-direction metrics are comparable; projection R^2 reports how linear the
learned reward is.

Usage:
    python run_recovery_deep.py --grid-size 8 --n-traj 2000 --seeds 0 1 2 \
        --reward-num-layers 4 --reward-hidden-dim 128 --neural-lr 1e-2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Reuse the environment builders and helpers from the existing harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gridworld import (  # noqa: E402
    build_case1_state_action,
    build_case2_rust_style,
    build_case3_state_only,
    generate_data,
)

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import hybrid_iteration  # noqa: E402
from econirl.core.types import DDCProblem, Panel  # noqa: E402
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator  # noqa: E402
from econirl.estimators.mceirl_neural import MCEIRLNeural  # noqa: E402

CASES = {
    "state_action": build_case1_state_action,
    "rust_style": build_case2_rust_style,
    "state_only": build_case3_state_only,
}


def _solve(reward, prob, trans):
    op = SoftBellmanOperator(prob, trans)
    sol = hybrid_iteration(op, reward, tol=1e-10)
    return op, sol


def _policy_value_under_true(policy, true_reward, prob, trans):
    """V^pi(s) under the TRUE reward for a (soft) policy pi: (I - g P^pi)^-1 r^pi."""
    g = prob.discount_factor
    rpi = (policy * true_reward).sum(axis=1)                     # (S,)
    Ppi = jnp.einsum("sa,ast->st", policy, trans)               # (S,S)
    S = prob.num_states
    V = jnp.linalg.solve(jnp.eye(S) - g * Ppi, rpi)
    return V


def _recovery_metrics(theta, coef, true_policy, est_policy, true_V_opt,
                      true_reward, prob, trans, train, test):
    """Full recovery metric set for one fit."""
    theta = jnp.asarray(theta)
    coef = jnp.asarray(coef)
    # --- reward direction ---
    cos = float(jnp.dot(coef, theta) / (jnp.linalg.norm(coef) * jnp.linalg.norm(theta) + 1e-12))
    rmse = float(jnp.sqrt(jnp.mean((coef - theta) ** 2)))
    sign_match = float(jnp.mean(jnp.sign(coef) == jnp.sign(theta)))
    # --- policy recovery ---
    eps = 1e-12
    kl = float((true_policy * jnp.log((true_policy + eps) / (est_policy + eps))).sum(axis=1).mean())
    tv = float(0.5 * jnp.abs(true_policy - est_policy).sum(axis=1).mean())
    acc = float((true_policy.argmax(1) == est_policy.argmax(1)).mean() * 100)
    # --- value recovery: expected value difference under the true reward ---
    d0 = jnp.zeros(prob.num_states).at[0].set(1.0)
    V_est_pol = _policy_value_under_true(est_policy, true_reward, prob, trans)
    evd = float((d0 * (true_V_opt - V_est_pol)).sum())
    # --- likelihood in/out of sample (using est policy choice probs) ---
    def ll_per_obs(panel):
        lp = jnp.log(est_policy + eps)
        return float(lp[panel.get_all_states(), panel.get_all_actions()].sum() / panel.num_observations)
    return {
        "reward_cosine": cos, "reward_rmse": rmse, "reward_sign_match": sign_match,
        "policy_kl": kl, "policy_tv": tv, "policy_accuracy": acc, "evd": evd,
        "ll_per_obs_in": ll_per_obs(train), "ll_per_obs_out": ll_per_obs(test),
    }


def _make_coord_encoder(grid_size):
    def enc(s):
        s = jnp.asarray(s)
        r = (s // grid_size).astype(jnp.float32) / grid_size
        c = (s % grid_size).astype(jnp.float32) / grid_size
        return jnp.stack([r, c], axis=-1)
    return enc


def _split(full, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(full.trajectories))
    n_tr = int(len(idx) * 0.7)
    train = Panel(trajectories=[full.trajectories[i] for i in idx[:n_tr]])
    test = Panel(trajectories=[full.trajectories[i] for i in idx[n_tr:]])
    return train, test


def run_one(case_name, grid_size, n_traj, n_periods, discount, seed, max_epochs,
            reward_hidden_dim=64, reward_num_layers=2, neural_lr=5e-4):
    prob, trans, rfn, names, theta = CASES[case_name](grid_size, discount)
    full, true_policy = generate_data(prob, trans, rfn, theta, n_traj, n_periods, seed)
    train, test = _split(full, seed)

    true_reward = rfn.compute(theta)
    _, true_sol = _solve(true_reward, prob, trans)
    true_V_opt = true_sol.V

    # transfer env: 10% transition noise
    rng_key = jnp.array(np.random.RandomState(seed).rand(*trans.shape))
    noisy = 0.9 * trans + 0.1 * (rng_key / rng_key.sum(axis=2, keepdims=True))
    noisy_prob = DDCProblem(num_states=prob.num_states, num_actions=prob.num_actions,
                            discount_factor=discount)
    transfer_panel, transfer_true = generate_data(noisy_prob, noisy, rfn, theta, 800, n_periods, seed + 10)

    out = {}

    # --- linear MCE-IRL ---
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05, outer_max_iter=1000, outer_tol=1e-8,
        inner_solver="hybrid", inner_tol=1e-10, inner_max_iter=10000,
        use_adam=True, compute_se=False, verbose=False))
    res = mce.estimate(panel=train, utility=rfn, problem=prob, transitions=trans, true_params=theta)
    _, mce_sol = _solve(rfn.compute(res.parameters), prob, trans)
    m = _recovery_metrics(theta, res.parameters, true_policy, mce_sol.policy, true_V_opt,
                          true_reward, prob, trans, train, test)
    # transfer accuracy: est reward re-solved under noisy dynamics
    _, mce_transfer = _solve(rfn.compute(res.parameters), noisy_prob, noisy)
    m["transfer_accuracy"] = float((transfer_true.argmax(1) == mce_transfer.policy.argmax(1)).mean() * 100)
    m["projection_r2"] = None
    m["runtime"] = time.time() - t0
    m["converged"] = bool(res.converged)
    out["MCE-IRL"] = m

    # --- Neural MCE-IRL ---
    t0 = time.time()
    nmce = MCEIRLNeural(
        n_states=prob.num_states, n_actions=prob.num_actions, discount=discount,
        reward_type="state_action", reward_hidden_dim=reward_hidden_dim,
        reward_num_layers=reward_num_layers,
        max_epochs=max_epochs, lr=neural_lr, seed=seed, feature_names=names,
        state_encoder=_make_coord_encoder(grid_size), state_dim=2, verbose=False)
    nmce.fit(train, features=np.asarray(rfn.feature_matrix), transitions=np.asarray(trans))
    npol = jnp.asarray(nmce.policy_)
    m = _recovery_metrics(theta, nmce.coef_, true_policy, npol, true_V_opt,
                          true_reward, prob, trans, train, test)
    # transfer: project neural reward onto features, re-solve under noisy dynamics
    _, nmce_transfer = _solve(rfn.compute(jnp.asarray(nmce.coef_)), noisy_prob, noisy)
    m["transfer_accuracy"] = float((transfer_true.argmax(1) == nmce_transfer.policy.argmax(1)).mean() * 100)
    m["projection_r2"] = None if nmce.projection_r2_ is None else float(nmce.projection_r2_)
    m["runtime"] = time.time() - t0
    m["converged"] = bool(nmce.converged_)
    out["Neural MCE-IRL"] = m

    return {"case": case_name, "seed": seed, "true_theta": [float(x) for x in theta],
            "names": names, "estimators": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--n-traj", type=int, default=1500)
    ap.add_argument("--n-periods", type=int, default=50)
    ap.add_argument("--discount", type=float, default=0.95)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--reward-num-layers", type=int, default=2)
    ap.add_argument("--reward-hidden-dim", type=int, default=64)
    ap.add_argument("--neural-lr", type=float, default=5e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # Ziebart's reward is R(s), linear in state features (state_only). The
    # action-dependent cases stay available but are off by default.
    ap.add_argument("--cases", nargs="+", default=["state_only"])
    ap.add_argument("--out", type=Path, default=Path("validation/results/recovery_ziebart_deep.json"))
    args = ap.parse_args()

    records = []
    for case in args.cases:
        for seed in args.seeds:
            t = time.time()
            rec = run_one(case, args.grid_size, args.n_traj, args.n_periods,
                          args.discount, seed, args.max_epochs,
                          reward_hidden_dim=args.reward_hidden_dim,
                          reward_num_layers=args.reward_num_layers,
                          neural_lr=args.neural_lr)
            records.append(rec)
            print(f"[{case} seed={seed}] done in {time.time()-t:.1f}s")
            for ename, m in rec["estimators"].items():
                print(f"   {ename:16s} cos={m['reward_cosine']:+.3f} rmse={m['reward_rmse']:.3f} "
                      f"acc={m['policy_accuracy']:5.1f} tv={m['policy_tv']:.3f} kl={m['policy_kl']:.3f} "
                      f"evd={m['evd']:+.3f} R2={m['projection_r2']} t={m['runtime']:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"grid_size": args.grid_size, "n_traj": args.n_traj, "n_periods": args.n_periods,
            "discount": args.discount, "max_epochs": args.max_epochs, "seeds": args.seeds,
            "reward_num_layers": args.reward_num_layers, "reward_hidden_dim": args.reward_hidden_dim,
            "neural_lr": args.neural_lr}
    args.out.write_text(json.dumps({"meta": meta, "records": records}, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
