"""Simulation study: direct optimization approaches to dynamic discrete choice.

Three estimators that all LIFT the value function into the optimization and solve a
single problem (no nested Bellman fixed-point inner loop), contrasted on the same
anchored data:

  - tabular MPEC   linear reward, known P, HARD Bellman equality constraint (Su-Judd)
  - neural MPEC    neural reward u_theta(s,a) + neural value V_phi(s), known P, SOFT
                   Bellman-residual penalty over the full state grid (collocation)
  - GLADIUS        neural Q + neural EV, MODEL-FREE, soft consistency penalty on
                   observed transitions (the EV network stands in for unknown P)

Two regimes:
  Cell 1 (linear reward)    the true reward is linear in the state feature. The
                            correctly-specified 4-parameter tabular MPEC pools across
                            states and is the most efficient; neural MPEC is consistent
                            but pays a variance price; both use the anchor.
  Cell 2 (nonlinear reward) the true reward is a wave in the state feature. The linear
                            MPEC is now MISSPECIFIED (fed a reduced (1, x) map) and its
                            reward error plateaus at a bias floor that data cannot remove;
                            the flexible neural reward recovers the wave.

Metrics (no standard errors):
  - reward RMSE over the ESTIMATED actions {0, 1} vs the true reward (the reference
    action 2 is anchored to zero in both truth and estimate, so it is excluded from the
    reward error to avoid a free win).
  - value RMSE vs the soft-Bellman oracle value of the true reward (behavioral,
    comparable across all three).
  - a data-scaling sweep (reward RMSE vs N) that shows consistency (Cell 1) and the
    misspecification bias floor (Cell 2).

GLADIUS is run at the repo-standard net size (128 wide, 3 layers, 500 epochs) WITH the
same action-2 anchor the MPEC methods receive, so it is a fair, not a strawman, baseline.
Its reward is still model-free (it never uses the known P) and so sits in a different
identification gauge; that is the point of including it.

Run:  python scripts/sim_direct_optimization.py
      python scripts/sim_direct_optimization.py --multistart 10   # + local-optima probe
Writes validation/results/sim_direct_optimization.json and two figures under
docs/_static/simulation_studies/.
"""

from __future__ import annotations

import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.core.solvers import value_iteration  # noqa: E402
from econirl.environments import ArrayMDP  # noqa: E402
from econirl.preferences.linear import LinearUtility  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "sim_direct_optimization.json")
FIG_DIR = os.path.join(_ROOT, "docs", "_static", "simulation_studies")
FIG_REWARDS = os.path.join(FIG_DIR, "direct_optimization_rewards.png")
FIG_SCALING = os.path.join(FIG_DIR, "direct_optimization_scaling.png")

BETA, SIGMA, N_STATES = 0.95, 1.0, 20
REF_ACTION = 2  # zero-reward reference action (last index)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def oracle_value(env) -> np.ndarray:
    """Soft-Bellman fixed point of the TRUE reward -- the value RMSE target."""
    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    return np.asarray(value_iteration(operator, env.compute_utility_matrix()).V)


# ---------------------------------------------------------------------------
# DGP: anchored 3-action MDP, two reward regimes
# ---------------------------------------------------------------------------


def _transitions(S: int) -> np.ndarray:
    """Ergodic kernel: action 0 drifts up, action 1 drifts down, action 2 walks local."""
    T = np.zeros((3, S, S), dtype=np.float64)
    for s in range(S):
        for off, p in [(0, 0.2), (1, 0.6), (2, 0.2)]:
            T[0, s, min(s + off, S - 1)] += p
        for off, p in [(0, 0.2), (1, 0.6), (2, 0.2)]:
            T[1, s, max(s - off, 0)] += p
        for off, p in [(-1, 0.25), (0, 0.5), (1, 0.25)]:
            T[2, s, min(max(s + off, 0), S - 1)] += p
    return T


def build_linear_cell(S: int = N_STATES, seed: int = 0):
    """True reward LINEAR in x; the estimator's linear map is correctly specified."""
    x = np.arange(S, dtype=np.float64) / (S - 1)
    phi = np.zeros((S, 3, 4), dtype=np.float64)
    phi[:, 0, 0] = 1.0
    phi[:, 0, 1] = x
    phi[:, 1, 2] = 1.0
    phi[:, 1, 3] = x
    theta = np.array([0.4, -0.6, 0.2, -0.4], dtype=np.float64)
    names = ["a0_const", "a0_slope", "a1_const", "a1_slope"]
    env = ArrayMDP(_transitions(S), phi, theta, discount_factor=BETA,
                   scale_parameter=SIGMA, parameter_names=names, seed=seed)
    # Linear MPEC sees the SAME features -> correctly specified.
    return env, np.asarray(env.feature_matrix), names


def build_nonlinear_cell(S: int = N_STATES, seed: int = 1):
    """True reward a WAVE in x; the linear MPEC is fed a reduced (1, x) map -> misspecified."""
    x = np.arange(S, dtype=np.float64) / (S - 1)
    r0 = 0.7 * np.sin(2.0 * np.pi * x)
    r1 = 0.6 * np.cos(2.0 * np.pi * x) - 0.1
    # Exact encoding: one feature column per estimated action carries its reward directly.
    phi = np.zeros((S, 3, 2), dtype=np.float64)
    phi[:, 0, 0] = r0
    phi[:, 1, 1] = r1
    theta = np.array([1.0, 1.0], dtype=np.float64)
    env = ArrayMDP(_transitions(S), phi, theta, discount_factor=BETA,
                   scale_parameter=SIGMA, parameter_names=["r0_scale", "r1_scale"], seed=seed)
    # The linear MPEC only gets an AFFINE (1, x) map per action -> cannot fit the wave.
    est_phi = np.zeros((S, 3, 4), dtype=np.float64)
    est_phi[:, 0, 0] = 1.0
    est_phi[:, 0, 1] = x
    est_phi[:, 1, 2] = 1.0
    est_phi[:, 1, 3] = x
    return env, est_phi, ["a0_const", "a0_slope", "a1_const", "a1_slope"]


# ---------------------------------------------------------------------------
# Neural MPEC: neural reward + neural value, exact known-P Bellman penalty
# ---------------------------------------------------------------------------


class RewardNet(eqx.Module):
    """u_theta(s, a). Reference action (last index) hard-pinned to zero."""

    mlp: eqx.nn.MLP

    def __init__(self, n_states, n_actions, width, depth, *, key):
        self.mlp = eqx.nn.MLP(in_size=n_states, out_size=n_actions - 1,
                              width_size=width, depth=depth, activation=jax.nn.tanh, key=key)

    def all_actions(self, onehot):  # (S, S) -> (S, A)
        raw = jax.vmap(self.mlp)(onehot)
        zeros = jnp.zeros((raw.shape[0], 1), dtype=raw.dtype)
        return jnp.concatenate([raw, zeros], axis=1)


class ValueNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, n_states, width, depth, *, key):
        self.mlp = eqx.nn.MLP(in_size=n_states, out_size=1, width_size=width,
                              depth=depth, activation=jax.nn.tanh, key=key)

    def all_states(self, onehot):  # (S, S) -> (S,)
        return jax.vmap(self.mlp)(onehot).squeeze(-1)


class NeuralMPEC(eqx.Module):
    reward: RewardNet
    value: ValueNet


def run_neural_mpec(env, obs_states, obs_actions, *, width=32, depth=2, rho=1.0,
                    epochs=4000, lr=5e-3, seed=0) -> dict:
    """Co-train (u_theta, V_phi) with NLL + exact (known-P) Bellman-residual penalty."""
    S, A = env.num_states, env.num_actions
    onehot = jnp.eye(S, dtype=jnp.float64)
    T = jnp.asarray(env.transition_matrices, dtype=jnp.float64)
    obs_s = jnp.asarray(np.asarray(obs_states), dtype=jnp.int32)
    obs_a = jnp.asarray(np.asarray(obs_actions), dtype=jnp.int32)
    w = jnp.ones(S, dtype=jnp.float64) / S  # uniform collocation over the full grid

    key = jax.random.PRNGKey(seed)
    k_r, k_v = jax.random.split(key)
    model = NeuralMPEC(reward=RewardNet(S, A, width, depth, key=k_r),
                       value=ValueNet(S, width, depth, key=k_v))

    def loss_fn(m, rho_):
        u_all = m.reward.all_actions(onehot)
        V_all = m.value.all_states(onehot)
        EV = jnp.einsum("ast,t->as", T, V_all)
        Q = u_all + BETA * EV.T
        logp = jax.nn.log_softmax(Q / SIGMA, axis=1)
        nll = -logp[obs_s, obs_a].mean()
        resid = V_all - SIGMA * jax.scipy.special.logsumexp(Q / SIGMA, axis=1)
        return nll + 0.5 * rho_ * jnp.sum(w * resid**2), None

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, ostate, rho_):
        (loss, _), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(m, rho_)
        updates, ostate = opt.update(grads, ostate, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, updates), ostate

    rho_j = jnp.float64(rho)
    for _ in range(epochs):
        model, opt_state = step(model, opt_state, rho_j)

    u_all = np.asarray(model.reward.all_actions(onehot))
    V_all = np.asarray(model.value.all_states(onehot))
    EV = np.einsum("ast,t->as", np.asarray(T), V_all)
    Q = u_all + BETA * EV.T
    resid = V_all - SIGMA * np.asarray(jax.scipy.special.logsumexp(jnp.asarray(Q) / SIGMA, axis=1))
    true_R = np.asarray(env.true_reward_matrix)
    return {
        "reward_rmse": rmse(u_all[:, :A - 1], true_R[:, :A - 1]),  # estimated actions only
        "value_rmse": rmse(V_all, oracle_value(env)),
        "max_bellman_resid": float(np.abs(resid).max()),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def run_tabular_mpec(env, panel, est_features, est_names, initial_params=None) -> dict:
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    util = LinearUtility(feature_matrix=jnp.asarray(est_features), parameter_names=est_names)
    est = MPECEstimator(config=MPECConfig(solver="slsqp", outer_max_iter=200,
                                          constraint_tol=1e-6),
                        compute_hessian=False, verbose=False)
    # initial_params=None keeps the estimator's default start; passing a vector lets
    # the multi-start probe vary theta0 (V auto-inits at its Bellman fixed point).
    res = est.estimate(panel, util, env.problem_spec, env.transition_matrices,
                       initial_params=initial_params)
    est_R = np.einsum("sak,k->sa", np.asarray(est_features), np.asarray(res.parameters))
    true_R = np.asarray(env.true_reward_matrix)
    A = env.num_actions
    return {
        "reward_rmse": rmse(est_R[:, :A - 1], true_R[:, :A - 1]),
        "value_rmse": rmse(res.value_function, oracle_value(env)),
        "converged": bool(res.converged),
        "theta_hat": np.asarray(res.parameters, dtype=np.float64).tolist(),
        "constraint_violation": float(res.metadata.get("final_constraint_violation", float("nan"))),
    }


def run_gladius(env, panel) -> dict:
    from econirl.estimation.gladius import GLADIUSConfig, GLADIUSEstimator

    S, A = env.num_states, env.num_actions
    util = LinearUtility(feature_matrix=env.feature_matrix,
                         parameter_names=env.parameter_names)
    est = GLADIUSEstimator(config=GLADIUSConfig(
        q_hidden_dim=128, v_hidden_dim=128, q_num_layers=3, v_num_layers=3,
        max_epochs=500, batch_size=512,
        anchor_action=REF_ACTION, anchor_rewards=tuple(0.0 for _ in range(S)),
        anchor_bellman_mode="anchor_moment", compute_se=False, verbose=False,
    ))
    res = est.estimate(panel, util, env.problem_spec, env.transition_matrices)
    reward_table = np.asarray(res.metadata["reward_table"], dtype=np.float64)
    true_R = np.asarray(env.true_reward_matrix)
    return {
        "reward_rmse": rmse(reward_table[:, :A - 1], true_R[:, :A - 1]),
        "value_rmse": rmse(res.value_function, oracle_value(env)),
        "converged": bool(res.converged),
        "note": "model-free; reward in its own parameterization (no known P)",
    }


# ---------------------------------------------------------------------------
# Multi-start local-optima probe
# ---------------------------------------------------------------------------


def _agg(vals) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {"reward_rmse_mean": float(a.mean()), "reward_rmse_std": float(a.std()),
            "reward_rmse_min": float(a.min()), "reward_rmse_max": float(a.max())}


def multistart_probe(env, est_feat, est_names, *, n_starts=10, scale=0.5, seed=20240) -> dict:
    """Hold one linear-cell panel fixed; vary ONLY the optimizer start.

    Isolates optimization robustness from sampling noise. Koiso-Otani (2024) report
    that their MPEC search estimator "struggles with finding local optima"; this asks
    whether random starts on our soft-Bellman DDC likelihood agree on the same MLE.
    Linear MPEC varies theta0 ~ N(0, scale^2); neural MPEC varies the network init seed.
    """
    panel, obs_s, obs_a = _panel_obs(env, 400, 40, seed=12345)  # same panel as the main linear cell
    n_obs = int(len(obs_s))
    rng = np.random.default_rng(seed)
    n_par = len(est_names)
    print(f"\n=== multi-start probe: K={n_starts} on the linear cell ({n_obs} obs) ===")

    lin_starts = []
    for k in range(n_starts):
        theta0 = jnp.asarray(rng.normal(0.0, scale, size=n_par), dtype=jnp.float64)
        r = run_tabular_mpec(env, panel, est_feat, est_names, initial_params=theta0)
        lin_starts.append({"reward_rmse": r["reward_rmse"], "converged": r["converged"],
                           "constraint_violation": r["constraint_violation"],
                           "theta_hat": r["theta_hat"]})
        print(f"[multistart/linear] start {k}: reward {r['reward_rmse']:.4f} "
              f"conv={r['converged']} cviol={r['constraint_violation']:.1e}")

    neu_starts = []
    for k in range(n_starts):
        r = run_neural_mpec(env, obs_s, obs_a, seed=k)
        neu_starts.append({"reward_rmse": r["reward_rmse"],
                           "max_bellman_resid": r["max_bellman_resid"]})
        print(f"[multistart/neural] start {k}: reward {r['reward_rmse']:.4f} "
              f"resid={r['max_bellman_resid']:.1e}")

    thetas = np.asarray([s["theta_hat"] for s in lin_starts], dtype=np.float64)
    return {
        "cell": "linear reward", "n_obs": n_obs, "n_starts": n_starts,
        "theta_init_scale": scale,
        "linear_mpec": {
            **_agg([s["reward_rmse"] for s in lin_starts]),
            "n_converged": int(sum(s["converged"] for s in lin_starts)),
            "max_constraint_violation": float(max(s["constraint_violation"] for s in lin_starts)),
            "theta_max_component_std": float(thetas.std(axis=0).max()),
            "starts": lin_starts,
        },
        "neural_mpec": {
            **_agg([s["reward_rmse"] for s in neu_starts]),
            "max_bellman_resid": float(max(s["max_bellman_resid"] for s in neu_starts)),
            "starts": neu_starts,
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_rewards(cells, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, (title, env) in zip(axes, cells):
        true_R = np.asarray(env.true_reward_matrix)
        x = np.arange(env.num_states) / (env.num_states - 1)
        ax.plot(x, true_R[:, 0], lw=1.6, label="action 0")
        ax.plot(x, true_R[:, 1], lw=1.6, label="action 1")
        ax.axhline(0.0, lw=1.0, ls="--", color="0.6", label="action 2 (reference)")
        ax.set_title(title)
        ax.set_xlabel("state feature $x_s$")
        ax.set_ylabel("true reward $u(s,a)$")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure_scaling(scaling, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, (title, rows) in zip(axes, scaling):
        n = [r["n_obs"] for r in rows]
        ax.loglog(n, [r["neural_reward_rmse"] for r in rows], "o-", lw=1.6, label="neural MPEC")
        ax.loglog(n, [r["tabular_reward_rmse"] for r in rows], "s-", lw=1.6, label="tabular MPEC")
        ax.set_title(title)
        ax.set_xlabel("observations $N$")
        ax.set_ylabel("reward RMSE (estimated actions)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _panel_obs(env, n_ind, n_per, seed):
    p = simulate_panel(env, n_individuals=n_ind, n_periods=n_per, seed=seed)
    return p, np.asarray(p.get_all_states()), np.asarray(p.get_all_actions())


def main(multistart=None) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    cells = [("linear reward", *build_linear_cell()),
             ("nonlinear reward", *build_nonlinear_cell())]

    results = {"meta": {"beta": BETA, "sigma": SIGMA, "num_states": N_STATES,
                        "reference_action": REF_ACTION,
                        "neural_config": "width 32, depth 2, rho 1.0, 4000 epochs",
                        "gladius_config": "128 wide, 3 layers, 500 epochs, anchored at action 2"},
               "cells": []}

    # Main comparison table per cell at a fixed panel size.
    for title, env, est_feat, est_names in cells:
        panel, obs_s, obs_a = _panel_obs(env, 400, 40, seed=12345)
        shares = (np.bincount(obs_a, minlength=3) / len(obs_a)).round(3).tolist()
        covered = int(len(np.unique(obs_s)))
        print(f"\n=== {title} cell: {len(obs_s)} obs | shares {shares} | coverage {covered}/{N_STATES} ===")

        methods = {}
        t0 = time.time()
        methods["neural MPEC"] = run_neural_mpec(env, obs_s, obs_a)
        print(f"  neural MPEC   reward={methods['neural MPEC']['reward_rmse']:.4f} "
              f"value={methods['neural MPEC']['value_rmse']:.4f} ({time.time()-t0:.1f}s)")
        t0 = time.time()
        methods["tabular MPEC"] = run_tabular_mpec(env, panel, est_feat, est_names)
        print(f"  tabular MPEC  reward={methods['tabular MPEC']['reward_rmse']:.4f} "
              f"value={methods['tabular MPEC']['value_rmse']:.4f} ({time.time()-t0:.1f}s)")
        t0 = time.time()
        try:
            methods["GLADIUS"] = run_gladius(env, panel)
            print(f"  GLADIUS       reward={methods['GLADIUS']['reward_rmse']:.4f} "
                  f"value={methods['GLADIUS']['value_rmse']:.4f} ({time.time()-t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            methods["GLADIUS"] = {"error": str(exc)}
            print(f"  GLADIUS       FAILED: {exc}")

        results["cells"].append({"cell": title, "n_obs": int(len(obs_s)),
                                 "action_shares": shares, "state_coverage": covered,
                                 "methods": methods})

    # Data-scaling sweep (neural vs tabular) per cell: consistency + bias floor.
    scaling_fig = []
    for title, env, est_feat, est_names in cells:
        rows = []
        for n_ind in [100, 400, 1600]:
            panel, ss, aa = _panel_obs(env, n_ind, 40, seed=999)
            nm = run_neural_mpec(env, ss, aa)
            tb = run_tabular_mpec(env, panel, est_feat, est_names)
            rows.append({"n_obs": int(len(ss)),
                         "neural_reward_rmse": nm["reward_rmse"],
                         "neural_value_rmse": nm["value_rmse"],
                         "tabular_reward_rmse": tb["reward_rmse"],
                         "tabular_value_rmse": tb["value_rmse"]})
            print(f"[scaling/{title}] N={len(ss)}: neural {nm['reward_rmse']:.4f} "
                  f"tabular {tb['reward_rmse']:.4f}")
        # attach to the matching cell record
        for c in results["cells"]:
            if c["cell"] == title:
                c["data_scaling"] = rows
        scaling_fig.append((title, rows))

    figure_rewards([(t, e) for t, e, _, _ in cells], FIG_REWARDS)
    figure_scaling(scaling_fig, FIG_SCALING)

    # Optional local-optima probe on the linear cell (gated behind --multistart).
    if multistart:
        _, env0, feat0, names0 = cells[0]
        results["multistart"] = multistart_probe(env0, feat0, names0, n_starts=multistart)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")
    print(f"Wrote {FIG_REWARDS}")
    print(f"Wrote {FIG_SCALING}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multistart", type=int, nargs="?", const=10, default=None,
                        metavar="K",
                        help="also run the K-start local-optima probe on the linear "
                             "cell (default K=10 when the flag is given with no value)")
    args = parser.parse_args()
    main(multistart=args.multistart)
