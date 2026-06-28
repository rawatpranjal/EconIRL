"""AIRL (Fu, Luo, Levine 2018) section 7.1: recovering true rewards in a random
tabular MDP, plus the transfer test under new dynamics.

Paper setup: 16 states, 4 actions, random transitions, reward 1.0 for any action
from state 0, start state 1. We use deterministic transitions (the taxonomy's
"R(s) + deterministic -> identified" regime). Two estimator parameterizations:
  - AIRL-1, state-only reward: recovers the ground-truth reward, transfers to optimal.
  - AIRL-2 default, state-action reward: recovers a shaped advantage, fails transfer.

Config follows validation/known_truth.py (the known-good paper-identification
settings). Metrics: affine R^2 of recovered reward vs true reward across states;
transfer = fraction of hard-optimal return under a fresh transition matrix T'.

Usage: python examples/airl-fu2018/run_tabular_71.py --seeds 0 1 2
"""
import argparse, json
from pathlib import Path
import numpy as np, jax.numpy as jnp
from econirl.core.types import DDCProblem
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel_from_policy
from econirl.estimation.adversarial.airl import AIRLConfig, AIRLEstimator

S, A, GAMMA = 16, 4, 0.9
prob = DDCProblem(num_states=S, num_actions=A, discount_factor=GAMMA)
r_true = np.zeros((S, A), dtype=np.float32); r_true[0, :] = 1.0
r_true_j = jnp.asarray(r_true)


def rand_T(seed):
    """Random deterministic transitions, regenerated until the reward state 0 is
    reachable from every state. Deterministic dynamics give clean action-effect
    identification (state-only AIRL recovers); the reachability guard removes the
    non-ergodic dead-ends a raw random graph can produce, where state 0 would be
    rarely visited and its reward unidentifiable."""
    base = seed
    while True:
        r = np.random.RandomState(base)
        nxt = np.array([[r.randint(S) for _ in range(S)] for _ in range(A)])  # (A,S)
        reach = np.zeros(S, bool); reach[0] = True
        for _ in range(S):
            for a in range(A):
                reach |= reach[nxt[a]]
        if reach.all():
            break
        base += 10_000
    T = np.zeros((A, S, S), dtype=np.float32)
    for a in range(A):
        for s in range(S):
            T[a, s, nxt[a, s]] = 1.0
    return jnp.asarray(T)


def expert_state0_visitation(T):
    """Fraction of expert occupancy at the rewarding state 0 (coverage check)."""
    pol = np.asarray(soft(r_true_j, T).policy)
    Ppi = np.einsum("sa,ast->st", pol, np.asarray(T))
    d = np.ones(S) / S
    for _ in range(2000):
        dn = d @ Ppi
        if np.max(np.abs(dn - d)) < 1e-12:
            break
        d = dn
    return float(d[0])


def soft(R, T):
    return hybrid_iteration(SoftBellmanOperator(prob, T), jnp.asarray(R), tol=1e-10)


def affine_r2(x, y):
    X = np.column_stack([x, np.ones(len(x))]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def policy_return(policy, T2):
    pol = np.asarray(policy); rpi = (pol * r_true).sum(1)
    Ppi = np.einsum("sa,ast->st", pol, np.asarray(T2))
    return float(np.linalg.solve(np.eye(S) - GAMMA * Ppi, rpi)[1])


def hard_policy(R_sa, T2):
    """Greedy (hard-optimal) policy under reward R_sa in dynamics T2."""
    R = np.asarray(R_sa); V = np.zeros(S)
    for _ in range(5000):
        Q = R + GAMMA * np.einsum("ast,t->sa", np.asarray(T2), V)
        Vn = Q.max(1)
        if np.max(np.abs(Vn - V)) < 1e-12:
            break
        V = Vn
    Q = R + GAMMA * np.einsum("ast,t->sa", np.asarray(T2), V)
    pol = np.zeros((S, A)); pol[np.arange(S), Q.argmax(1)] = 1.0
    return pol


def hard_opt_return(T2):
    return policy_return(hard_policy(r_true, T2), T2)


def fit(arg, panel, dummy, T):
    pid = (arg == "state")
    est = AIRLEstimator(config=AIRLConfig(
        reward_type="tabular", reward_arg=arg, reward_lr=0.02, discriminator_steps=5,
        policy_step_size=0.1 if pid else 0.3, max_rounds=200,
        min_rounds=150 if pid else 20, convergence_tol=0.01 if pid else 1e-4,
        generator_reward="f" if pid else "recovered", use_shaping=True,
        compute_se=False, verbose=False))
    return np.asarray(est.estimate(panel, dummy, prob, T).parameters).reshape(S, A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("validation/results/airl_fu2018_71.json"))
    args = ap.parse_args()
    dummy = ActionDependentReward(feature_matrix=jnp.zeros((S, A, 1)), parameter_names=["d"])

    records = []
    for seed in args.seeds:
        # Pick a training MDP where the expert adequately visits the rewarding
        # state 0 (so the reward is identifiable). Bounded search, keep the best.
        best_T, best_vis, b = None, -1.0, 1 + seed
        for _ in range(40):
            cand = rand_T(b); vis = expert_state0_visitation(cand)
            if vis > best_vis:
                best_T, best_vis = cand, vis
            if vis >= 0.10:
                break
            b += 1
        T, Tp = best_T, rand_T(99 + seed)
        print(f"  [seed {seed}] train state-0 visitation={best_vis:.3f}", flush=True)
        expert = soft(r_true_j, T)
        panel = simulate_panel_from_policy(prob, T, expert.policy, jnp.zeros(S).at[1].set(1.0),
                                           n_individuals=1500, n_periods=40, seed=2 + seed)
        R_opt, R_rand = hard_opt_return(Tp), policy_return(np.ones((S, A)) / A, Tp)

        def transfer(R_sa):
            return (policy_return(hard_policy(R_sa, Tp), Tp) - R_rand) / (R_opt - R_rand + 1e-12)

        R_so = fit("state", panel, dummy, T)
        R_sa = fit("state_action", panel, dummy, T)
        rec = {
            "seed": seed,
            "AIRL-1 (state-only)": {"reward_affine_r2": affine_r2(r_true[:, 0], R_so.mean(1)),
                                    "transfer_frac_optimal": float(transfer(np.broadcast_to(R_so.mean(1)[:, None], (S, A))))},
            "AIRL-2 (state-action)": {"reward_affine_r2": affine_r2(r_true.ravel(), R_sa.ravel()),
                                      "transfer_frac_optimal": float(transfer(R_sa))},
        }
        records.append(rec)
        a1, a2 = rec["AIRL-1 (state-only)"], rec["AIRL-2 (state-action)"]
        print(f"seed {seed}: AIRL-1 R2={a1['reward_affine_r2']:+.3f} transfer={a1['transfer_frac_optimal']:+.3f} | "
              f"AIRL-2 R2={a2['reward_affine_r2']:+.3f} transfer={a2['transfer_frac_optimal']:+.3f}", flush=True)

    def mean(tag, k):
        return float(np.mean([r[tag][k] for r in records]))
    summary = {tag: {"reward_affine_r2": mean(tag, "reward_affine_r2"),
                     "transfer_frac_optimal": mean(tag, "transfer_frac_optimal")}
               for tag in ["AIRL-1 (state-only)", "AIRL-2 (state-action)"]}
    meta = {"task": "AIRL Fu-Luo-Levine 2018 section 7.1 (deterministic tabular)",
            "n_states": S, "n_actions": A, "reward": "1.0 at state 0", "discount": GAMMA,
            "seeds": args.seeds,
            "metric": "affine R^2 of recovered reward vs true; transfer = fraction of hard-optimal return under new T'"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, indent=2) + "\n")
    print(f"\nmean AIRL-1 R2={summary['AIRL-1 (state-only)']['reward_affine_r2']:+.3f} "
          f"transfer={summary['AIRL-1 (state-only)']['transfer_frac_optimal']:+.3f} | "
          f"AIRL-2 R2={summary['AIRL-2 (state-action)']['reward_affine_r2']:+.3f} "
          f"transfer={summary['AIRL-2 (state-action)']['transfer_frac_optimal']:+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
