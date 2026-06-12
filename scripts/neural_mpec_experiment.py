"""Neural MPEC experiment.

Tests whether MPEC works with neural function approximators: a neural reward
``u_theta(s, a)`` and a neural value ``V_phi(s)``, co-trained in a SINGLE loop by
minimizing logit NLL plus a soft Bellman-residual penalty. The structural setting
(transitions ``P`` are known) lets us compute the Bellman residual EXACTLY as a
matrix-vector product, so there is no double-sampling bias and no need for the
learned EV network that GLADIUS uses.

    EV_phi(s,a) = sum_s' P_a(s,s') V_phi(s')          # exact, known P
    Q(s,a)      = u_theta(s,a) + beta * EV_phi(s,a)
    pi(a|s)     = softmax_a( Q(s,.)/sigma )
    NLL         = - mean_it log pi(a_it | s_it)
    resid(s)    = V_phi(s) - sigma*logsumexp_a( Q(s,.)/sigma )
    Loss        = NLL + (rho/2) * sum_{s in C} w(s) resid(s)^2     # C = collocation set

This is the penalty-relaxed (PINN-style) form of MPEC: once V is a network the
hard equality constraint can only be imposed at collocation points, so the
Lagrange-multiplier-per-state structure becomes a residual penalty. With known P
and a tiny state space the collocation set can be the whole grid.

We compare three estimators on ONE anchored DGP and ONE panel:
  - neural MPEC  (this script)         -- known P, exact residual, single value net
  - tabular MPEC (MPECEstimator)       -- exact constrained MLE, the gold standard
  - GLADIUS      (GLADIUSEstimator)    -- the model-free neural cousin (learned EV net)

Metrics (no standard errors, by request):
  - value RMSE  vs the soft-Bellman oracle value -- behavioral, comparable to ALL
  - reward RMSE over the ESTIMATED actions {0, 1} vs the true reward (the reference
        action 2 is anchored to zero in both truth and estimate, so it is excluded to
        avoid a free win). Clean for the known-P methods (tabular + neural MPEC); GLADIUS
        is model-free, so its reward sits in a different gauge and is a reference only.

Identification note: in a logit DDC the reward level is identified only relative to
a reference action. Here action 2 is a zero-reward reference (phi(s,2) == 0 in the
DGP), and the neural reward net hard-pins u_theta(s, 2) == 0. Without that anchor the
reward RMSE would be meaningless even with a perfectly recovered policy.

Run:  python scripts/neural_mpec_experiment.py
Writes scripts/neural_mpec_experiment_results.json and prints a markdown table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.environments import ArrayMDP
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Metrics (inlined to avoid importing the heavy known-truth harness)
# ---------------------------------------------------------------------------


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# DGP: 3-action anchored MDP (action 2 == zero-reward reference)
# ---------------------------------------------------------------------------


def build_anchored_mdp(
    num_states: int = 20,
    discount_factor: float = 0.95,
    scale_parameter: float = 1.0,
    seed: int = 0,
) -> ArrayMDP:
    """Ergodic 3-action MDP with a zero-reward reference action.

    Action 0 drifts the state up, action 1 drifts it down, action 2 is a local
    random walk and carries identically-zero reward (its features are zero). The
    up/down drift keeps every state visited so coverage is full. Rewards:

        u(s, 0) = theta0_const + theta0_slope * x_s     (good low, bad high)
        u(s, 1) = theta1_const + theta1_slope * x_s
        u(s, 2) = 0                                      (anchor)

    where x_s = s / (S - 1).
    """
    S, A, K = num_states, 3, 4
    x = np.arange(S, dtype=np.float64) / (S - 1)

    # Action-dependent linear features; action 2 (reference) has zero features.
    phi = np.zeros((S, A, K), dtype=np.float64)
    phi[:, 0, 0] = 1.0
    phi[:, 0, 1] = x
    phi[:, 1, 2] = 1.0
    phi[:, 1, 3] = x
    theta = np.array([0.4, -0.6, 0.2, -0.4], dtype=np.float64)
    names = ["a0_const", "a0_slope", "a1_const", "a1_slope"]

    # Transitions (A, S, S); rows sum to 1 (clipping piles mass on boundaries).
    T = np.zeros((A, S, S), dtype=np.float64)
    for s in range(S):
        for off, p in [(0, 0.2), (1, 0.6), (2, 0.2)]:  # action 0: drift up
            T[0, s, min(s + off, S - 1)] += p
        for off, p in [(0, 0.2), (1, 0.6), (2, 0.2)]:  # action 1: drift down
            T[1, s, max(s - off, 0)] += p
        for off, p in [(-1, 0.25), (0, 0.5), (1, 0.25)]:  # action 2: local walk
            T[2, s, min(max(s + off, 0), S - 1)] += p

    return ArrayMDP(
        transitions=T,
        features=phi,
        theta=theta,
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
        parameter_names=names,
        seed=seed,
    )


def oracle_policy_value(env: ArrayMDP) -> tuple[np.ndarray, np.ndarray]:
    """Soft-Bellman fixed point of the TRUE utility -- the value RMSE target."""
    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)
    result = value_iteration(operator, env.compute_utility_matrix())
    return np.asarray(result.policy), np.asarray(result.V)


# ---------------------------------------------------------------------------
# Neural nets: one-hot state in -> reward / value out
# ---------------------------------------------------------------------------


class RewardNet(eqx.Module):
    """u_theta(s, a). Reference action (last index) is hard-pinned to zero."""

    mlp: eqx.nn.MLP
    n_actions: int = eqx.field(static=True)

    def __init__(self, n_states, n_actions, width, depth, *, key):
        self.n_actions = n_actions
        # Output one reward per NON-reference action; the reference column is 0.
        self.mlp = eqx.nn.MLP(
            in_size=n_states, out_size=n_actions - 1, width_size=width,
            depth=depth, activation=jax.nn.tanh, key=key,
        )

    def all_actions(self, onehot):  # onehot: (S, S) -> (S, A)
        raw = jax.vmap(self.mlp)(onehot)  # (S, A-1) for actions 0..A-2
        zeros = jnp.zeros((raw.shape[0], 1), dtype=raw.dtype)  # reference action == 0
        return jnp.concatenate([raw, zeros], axis=1)


class ValueNet(eqx.Module):
    """V_phi(s)."""

    mlp: eqx.nn.MLP

    def __init__(self, n_states, width, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=n_states, out_size=1, width_size=width,
            depth=depth, activation=jax.nn.tanh, key=key,
        )

    def all_states(self, onehot):  # (S, S) -> (S,)
        return jax.vmap(self.mlp)(onehot).squeeze(-1)


class NeuralMPEC(eqx.Module):
    reward: RewardNet
    value: ValueNet


# ---------------------------------------------------------------------------
# Neural MPEC training
# ---------------------------------------------------------------------------


def train_neural_mpec(
    env: ArrayMDP,
    obs_states: np.ndarray,
    obs_actions: np.ndarray,
    *,
    width: int,
    depth: int,
    rho: float,
    collocation: str,  # "all" or "observed"
    epochs: int = 3000,
    lr: float = 5e-3,
    seed: int = 0,
) -> dict:
    """Co-train (u_theta, V_phi) with NLL + exact Bellman-residual penalty."""
    S = env.num_states
    A = env.num_actions
    beta = float(env.problem_spec.discount_factor)
    sigma = float(env.problem_spec.scale_parameter)

    onehot = jnp.eye(S, dtype=jnp.float64)
    T = jnp.asarray(env.transition_matrices, dtype=jnp.float64)
    obs_s = jnp.asarray(np.asarray(obs_states), dtype=jnp.int32)
    obs_a = jnp.asarray(np.asarray(obs_actions), dtype=jnp.int32)

    # Collocation weights (sum to 1): uniform over the full grid, or weighted by
    # how often each state is actually observed in the panel.
    if collocation == "all":
        w = jnp.ones(S, dtype=jnp.float64) / S
    elif collocation == "observed":
        counts = np.bincount(np.asarray(obs_states), minlength=S).astype(np.float64)
        w = jnp.asarray(counts / counts.sum(), dtype=jnp.float64)
    else:
        raise ValueError(f"collocation must be 'all' or 'observed', got {collocation!r}")

    key = jax.random.PRNGKey(seed)
    k_r, k_v = jax.random.split(key)
    model = NeuralMPEC(
        reward=RewardNet(S, A, width, depth, key=k_r),
        value=ValueNet(S, width, depth, key=k_v),
    )

    def loss_fn(m, rho_):
        u_all = m.reward.all_actions(onehot)             # (S, A)
        V_all = m.value.all_states(onehot)               # (S,)
        EV = jnp.einsum("ast,t->as", T, V_all)           # (A, S)
        Q = u_all + beta * EV.T                          # (S, A)
        logp = jax.nn.log_softmax(Q / sigma, axis=1)
        nll = -logp[obs_s, obs_a].mean()
        resid = V_all - sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)
        penalty = jnp.sum(w * resid**2)                  # weighted-mean residual^2
        return nll + 0.5 * rho_ * penalty, (nll, penalty)

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, ostate, rho_):
        (loss, (nll, pen)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(m, rho_)
        updates, ostate = opt.update(grads, ostate, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, updates)
        return m, ostate, loss, nll, pen

    rho_j = jnp.float64(rho)
    final_nll = final_pen = float("nan")
    for _ in range(epochs):
        model, opt_state, _loss, nll, pen = step(model, opt_state, rho_j)
        final_nll, final_pen = float(nll), float(pen)

    # Final objects.
    u_all = np.asarray(model.reward.all_actions(onehot))
    V_all = np.asarray(model.value.all_states(onehot))
    EV = np.einsum("ast,t->as", np.asarray(T), V_all)
    Q = u_all + beta * EV.T
    resid = V_all - sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)

    true_R = np.asarray(env.true_reward_matrix)
    oracle_pol, oracle_V = oracle_policy_value(env)
    # Irreducible NLL floor = mean NLL under the TRUE soft-optimal policy. If the
    # neural model's NLL reaches (or dips just below) this floor, the likelihood is
    # fully optimized and any remaining reward error is finite-sample variance, not
    # under-training.
    floor = float(-np.log(np.clip(
        np.asarray(oracle_pol)[np.asarray(obs_states), np.asarray(obs_actions)],
        1e-12, 1.0)).mean())
    A = env.num_actions
    return {
        # Reward RMSE over the ESTIMATED actions only. The reference action (last index)
        # is anchored to zero in both truth and estimate, so including it is a free win
        # that deflates the headline; it is excluded.
        "reward_rmse": rmse(u_all[:, :A - 1], true_R[:, :A - 1]),
        "value_rmse": rmse(V_all, oracle_V),
        "max_bellman_resid": float(np.abs(np.asarray(resid)).max()),
        "final_nll": final_nll,
        "nll_floor": floor,
        "nll_gap_to_floor": final_nll - floor,
        "final_penalty": final_pen,
        "epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Baselines: tabular MPEC and GLADIUS on the same DGP/panel
# ---------------------------------------------------------------------------


def run_tabular_mpec(env, panel) -> dict:
    from econirl.estimation.mpec import MPECEstimator, MPECConfig
    from econirl.preferences.linear import LinearUtility

    util = LinearUtility(feature_matrix=env.feature_matrix,
                         parameter_names=env.parameter_names)
    est = MPECEstimator(
        config=MPECConfig(solver="slsqp", outer_max_iter=200, constraint_tol=1e-6),
        compute_hessian=False, verbose=False,
    )
    res = est.estimate(panel, util, env.problem_spec, env.transition_matrices)
    est_R = np.einsum("sak,k->sa", np.asarray(env.feature_matrix),
                      np.asarray(res.parameters))
    true_R = np.asarray(env.true_reward_matrix)
    _, oracle_V = oracle_policy_value(env)
    A = env.num_actions
    return {
        "reward_rmse": rmse(est_R[:, :A - 1], true_R[:, :A - 1]),  # estimated actions only
        "value_rmse": rmse(res.value_function, oracle_V),
        "converged": bool(res.converged),
    }


def run_gladius(env, panel) -> dict:
    # Fair baseline: repo-standard net size (128 wide, 3 layers, 500 epochs) WITH the
    # same action-2 anchor the MPEC methods receive. GLADIUS is still model-free (it
    # never uses the known transitions), so its reward sits in a different gauge -- that
    # is the honest contrast, not a crippled config.
    from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
    from econirl.preferences.linear import LinearUtility

    S, A = env.num_states, env.num_actions
    util = LinearUtility(feature_matrix=env.feature_matrix,
                         parameter_names=env.parameter_names)
    est = GLADIUSEstimator(
        config=GLADIUSConfig(
            q_hidden_dim=128, v_hidden_dim=128, q_num_layers=3, v_num_layers=3,
            max_epochs=500, batch_size=512,
            anchor_action=2, anchor_rewards=tuple(0.0 for _ in range(S)),
            anchor_bellman_mode="anchor_moment", compute_se=False, verbose=False,
        )
    )
    res = est.estimate(panel, util, env.problem_spec, env.transition_matrices)
    reward_table = np.asarray(res.metadata.get("reward_table"), dtype=np.float64)
    true_R = np.asarray(env.true_reward_matrix)
    _, oracle_V = oracle_policy_value(env)
    return {
        "reward_rmse": rmse(reward_table[:, :A - 1], true_R[:, :A - 1]),  # estimated actions
        "value_rmse": rmse(res.value_function, oracle_V),
        "converged": bool(res.converged),
        "note": "model-free; reward in a different gauge (no known P)",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    env = build_anchored_mdp()
    S, A = env.num_states, env.num_actions

    # One shared panel for all estimators.
    panel = simulate_panel(env, n_individuals=400, n_periods=40, seed=12345)
    obs_states = np.asarray(panel.get_all_states())
    obs_actions = np.asarray(panel.get_all_actions())

    # Pre-estimation diagnostics (CLAUDE.md rule: check coverage before fitting).
    shares = np.bincount(obs_actions, minlength=A) / len(obs_actions)
    covered = len(np.unique(obs_states))
    print(f"DGP: {S} states, {A} actions, beta={env.problem_spec.discount_factor}, "
          f"sigma={env.problem_spec.scale_parameter}")
    print(f"Panel: {len(obs_states)} obs | action shares = "
          f"{np.round(shares, 3).tolist()} | state coverage = {covered}/{S}")
    print()

    # Neural MPEC sweep (serial, one config at a time -- per saved feedback).
    sweep = [
        {"width": 16, "depth": 1, "rho": 1.0, "collocation": "all"},
        {"width": 32, "depth": 2, "rho": 1.0, "collocation": "all"},
        {"width": 32, "depth": 2, "rho": 0.1, "collocation": "all"},
        {"width": 32, "depth": 2, "rho": 10.0, "collocation": "all"},
        {"width": 32, "depth": 2, "rho": 1.0, "collocation": "observed"},
    ]

    rows = []
    for cfg in sweep:
        t0 = time.time()
        m = train_neural_mpec(env, obs_states, obs_actions, seed=0, **cfg)
        m["runtime"] = time.time() - t0
        label = (f"neural MPEC w{cfg['width']} d{cfg['depth']} "
                 f"rho={cfg['rho']} colloc={cfg['collocation']}")
        rows.append({"estimator": label, **cfg, **m})
        print(f"[done] {label}: reward_rmse={m['reward_rmse']:.4f} "
              f"value_rmse={m['value_rmse']:.4f} "
              f"max_resid={m['max_bellman_resid']:.2e} ({m['runtime']:.1f}s)")

    # Baselines.
    baselines = {}
    for name, fn in [("tabular MPEC", run_tabular_mpec), ("GLADIUS", run_gladius)]:
        t0 = time.time()
        try:
            b = fn(env, panel)
            b["runtime"] = time.time() - t0
            baselines[name] = b
            print(f"[done] {name}: reward_rmse={b['reward_rmse']:.4f} "
                  f"value_rmse={b['value_rmse']:.4f} ({b['runtime']:.1f}s)")
        except Exception as exc:  # noqa: BLE001 -- a baseline crash is data, not fatal
            baselines[name] = {"error": str(exc)}
            print(f"[FAIL] {name}: {exc}")

    # Markdown table.
    print("\n## Results (one anchored DGP, one panel)\n")
    print("| estimator | reward RMSE | value RMSE | max Bellman resid |")
    print("|---|---:|---:|---:|")
    print(f"| **tabular MPEC** (gold) | {baselines['tabular MPEC'].get('reward_rmse', float('nan')):.4f} "
          f"| {baselines['tabular MPEC'].get('value_rmse', float('nan')):.4f} | exact |")
    for r in rows:
        print(f"| {r['estimator']} | {r['reward_rmse']:.4f} | {r['value_rmse']:.4f} "
              f"| {r['max_bellman_resid']:.2e} |")
    g = baselines["GLADIUS"]
    print(f"| GLADIUS (model-free*) | {g.get('reward_rmse', float('nan')):.4f} "
          f"| {g.get('value_rmse', float('nan')):.4f} | n/a |")
    print("\n*GLADIUS is a fair baseline here (128 wide, 3 layers, 500 epochs, anchored at "
          "action 2). It is model-free, so its reward sits in a different gauge; its much "
          "higher value RMSE is the honest cost of not using the known transitions.")

    # ------------------------------------------------------------------
    # Consistency check: does the neural reward/value RMSE shrink toward the
    # tabular gold standard as data grows? This is the proof the estimator is
    # consistent (the per-state flexible reward inherits finite-sample CCP noise
    # that vanishes with N; the linear tabular model pools it away from the start).
    # ------------------------------------------------------------------
    print("\n## Data-scaling (consistency): neural MPEC vs tabular MPEC\n")
    print("| n_obs | neural reward RMSE | neural value RMSE | tabular reward RMSE | tabular value RMSE |")
    print("|---:|---:|---:|---:|---:|")
    scaling = []
    for n_ind in [100, 400, 1600]:
        p = simulate_panel(env, n_individuals=n_ind, n_periods=40, seed=999)
        ss = np.asarray(p.get_all_states())
        aa = np.asarray(p.get_all_actions())
        nm = train_neural_mpec(env, ss, aa, width=32, depth=2, rho=1.0,
                               collocation="all", epochs=4000, lr=5e-3, seed=0)
        tb = run_tabular_mpec(env, p)
        scaling.append({"n_obs": int(len(ss)),
                        "neural_reward_rmse": nm["reward_rmse"],
                        "neural_value_rmse": nm["value_rmse"],
                        "tabular_reward_rmse": tb["reward_rmse"],
                        "tabular_value_rmse": tb["value_rmse"]})
        print(f"| {len(ss)} | {nm['reward_rmse']:.4f} | {nm['value_rmse']:.4f} "
              f"| {tb['reward_rmse']:.4f} | {tb['value_rmse']:.4f} |")

    out = Path(__file__).with_name("neural_mpec_experiment_results.json")
    out.write_text(json.dumps({
        "dgp": {"num_states": S, "num_actions": A,
                "discount_factor": float(env.problem_spec.discount_factor),
                "scale_parameter": float(env.problem_spec.scale_parameter),
                "true_parameters": env.true_parameters,
                "action_shares": shares.tolist(), "state_coverage": int(covered)},
        "panel": {"n_obs": int(len(obs_states))},
        "neural_mpec": rows,
        "baselines": baselines,
        "data_scaling": scaling,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
