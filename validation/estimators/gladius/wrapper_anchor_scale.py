#!/usr/bin/env python3
"""Does the public NeuralGLADIUS wrapper recover reward scale once the anchor is wired in?

Drives the sklearn-style ``NeuralGLADIUS`` (not the paper-API ``GLADIUSEstimator``)
on the Shapeshifter ss-spine DGP (S=32, A=3, K=4, beta=0.95, linear reward, stochastic
transitions). Varies only the anchor: off, or on with ``anchor_action=0`` and the true
r_0 (=0 for ss-spine). NFXP is the structural scale reference.

This is the acceptance-finding gate: without the anchor the wrapper recovers direction
but understates scale (~0.77); with the anchor it should recover both (cosine >= 0.95,
scale in ~[0.85, 1.2]), reproducing the paper-API rebuttal.

Usage:
  python wrapper_anchor_scale.py --smoke
  python wrapper_anchor_scale.py --seeds 3 --out ../../results/gladius_anchor_scale.json
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

from econirl.environments.shapeshifter import (  # noqa: E402
    ShapeshifterConfig,
    ShapeshifterEnvironment,
)
from econirl.estimators.neural_gladius import NeuralGLADIUS  # noqa: E402
from econirl.estimation.nfxp import NFXPEstimator  # noqa: E402
from econirl.preferences.action_reward import ActionDependentReward  # noqa: E402
from econirl.simulation.synthetic import _compute_optimal_policy, simulate_panel  # noqa: E402

sys.path.insert(0, str(ROOT / "validation" / "benchmark"))
from metrics import policy_tv  # noqa: E402


def cosine(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float("nan") if na < 1e-12 or nb < 1e-12 else float(a @ b / (na * nb))


def scale_ratio(hat, true):
    nt = np.linalg.norm(np.asarray(true, float))
    return float("nan") if nt < 1e-12 else float(np.linalg.norm(np.asarray(hat, float)) / nt)


def run_gladius(env, panel, true_theta, true_policy, anchor, smoke):
    S, A = env.num_states, env.num_actions
    phi = np.asarray(env.feature_matrix, dtype=np.float32)  # (S, A, K)
    K = phi.shape[-1]
    phi_state = phi[:, 1, :]  # reward-relevant per-state descriptor, (S, K)

    anchor_action = 0 if anchor else None
    anchor_rewards = (
        tuple(float(x) for x in np.asarray(env.true_reward_matrix)[:, 0]) if anchor else None
    )
    model = NeuralGLADIUS(
        n_actions=A,
        discount=float(env.problem_spec.discount_factor),
        state_encoder=lambda s: phi_state[np.asarray(s)],
        state_dim=K,
        anchor_action=anchor_action,
        anchor_rewards=anchor_rewards,
        q_hidden_dim=16 if smoke else 128,
        q_num_layers=1 if smoke else 3,
        ev_hidden_dim=16 if smoke else 128,
        ev_num_layers=1 if smoke else 3,
        max_epochs=15 if smoke else 500,
        batch_size=128 if smoke else 512,
    )
    t0 = time.time()
    model.fit(panel, features=jnp.asarray(phi))
    theta_hat = model.coef_
    return {
        "estimator": "NeuralGLADIUS",
        "anchor": anchor,
        "cosine": cosine(theta_hat, true_theta),
        "scale_ratio": scale_ratio(theta_hat, true_theta),
        "policy_tv": policy_tv(np.asarray(model.policy_), true_policy),
        "theta_hat": np.asarray(theta_hat).tolist(),
        "seconds": round(time.time() - t0, 1),
    }


def run_nfxp(env, panel, true_theta, true_policy):
    utility = ActionDependentReward(
        jnp.asarray(env.feature_matrix, dtype=jnp.float32), env.parameter_names
    )
    est = NFXPEstimator(compute_hessian=False)
    t0 = time.time()
    summ = est.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=jnp.asarray(env.transition_matrices),
    )
    pol = np.asarray(summ.policy) if summ.policy is not None else true_policy
    return {
        "estimator": "NFXP",
        "cosine": cosine(summ.parameters, true_theta),
        "scale_ratio": scale_ratio(summ.parameters, true_theta),
        "policy_tv": policy_tv(np.asarray(pol), true_policy),
        "theta_hat": np.asarray(summ.parameters).tolist(),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-ind", type=int, default=800)
    ap.add_argument("--n-per", type=int, default=40)
    ap.add_argument(
        "--out", type=Path, default=ROOT / "validation" / "results" / "gladius_anchor_scale.json"
    )
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_ind, args.n_per = 1, 200, 20

    env = ShapeshifterEnvironment(ShapeshifterConfig())
    true_theta = np.asarray(env.get_true_parameter_vector())
    true_policy = np.asarray(_compute_optimal_policy(env))
    print(f"ss-spine: S={env.num_states} A={env.num_actions} "
          f"K={env.feature_matrix.shape[-1]} theta_true={np.round(true_theta,3).tolist()}")

    records = []
    for seed in range(args.seeds):
        panel = simulate_panel(env, n_individuals=args.n_ind, n_periods=args.n_per, seed=seed)
        print(f"\n[seed {seed}] obs={panel.num_observations}")
        for anchor in (False, True):
            rec = run_gladius(env, panel, true_theta, true_policy, anchor, args.smoke)
            rec["seed"] = seed
            records.append(rec)
            print(f"  GLADIUS anchor={str(anchor):5s} cos={rec['cosine']:.3f} "
                  f"scale={rec['scale_ratio']:.3f} tv={rec['policy_tv']:.3f} ({rec['seconds']}s)")
        nf = run_nfxp(env, panel, true_theta, true_policy)
        nf["seed"] = seed
        records.append(nf)
        print(f"  NFXP                cos={nf['cosine']:.3f} scale={nf['scale_ratio']:.3f} "
              f"tv={nf['policy_tv']:.3f} ({nf['seconds']}s)")

    out = {
        "dgp": "shapeshifter ss-spine (S=32,A=3,K=4,beta=0.95,linear,stoch)",
        "estimator": "NeuralGLADIUS (public wrapper) anchor on/off",
        "true_theta": true_theta.tolist(),
        "n_individuals": args.n_ind,
        "n_periods": args.n_per,
        "seeds": list(range(args.seeds)),
        "records": records,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
