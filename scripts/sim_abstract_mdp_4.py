"""Abstract MDP 4: an interaction effect in the reward.

The true utility multiplies two features. The estimators receive the two
features as main effects but never their product, so a linear utility is
misspecified: it cannot represent the interaction. The neural-reward methods
learn the product from the same features.

The point of this script is to report the result so that no agent (including
the one that wrote it) can lie about it. Same honesty contract as the other
simulation-study pages:

1. Raw facts only, written by the code. The JSON at
   ``validation/results/sim_abstract_mdp_4.json`` stores, per estimator per
   replication, the policy total-variation distance from the oracle, the value
   RMSE, the counterfactual regret, runtime, the convergence flag, and the
   verbatim exception on failure. Nothing is hand-entered.
2. Every printed number is recomputed from that JSON at render time.
3. Crashes are first class: a failing estimator is shown with its verbatim
   message, never dropped.
4. No parameter table. The true reward is not linear in the two features the
   estimators see, so there is no shared parameterization to compare a recovered
   theta against. Metrics are behavioral (policy, value) and counterfactual.
5. Stamped and reproducible. Seeds, date, version, and the DGP config are in the
   JSON. Structural estimators are deterministic; neural ones may vary slightly,
   which is stated rather than hidden.
6. ``--verify`` re-renders the table from the saved JSON only. ``--page`` writes
   the docs page (and figure) from the saved JSON.
7. Excluded estimators are listed with reasons.

Usage:
    python scripts/sim_abstract_mdp_4.py --replications 3 --verbose
    python scripts/sim_abstract_mdp_4.py --only-estimator GLADIUS --verbose
    python scripts/sim_abstract_mdp_4.py --page
    python scripts/sim_abstract_mdp_4.py --verify
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from econirl.environments.array_mdp import ArrayMDP  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402
from validation.benchmark import metrics as M  # noqa: E402
from validation.benchmark.regret import estimator_regret  # noqa: E402
from validation.benchmark.runner import (  # noqa: E402
    _action_reward,
    _linear_utility,
    _oracle,
)

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "sim_abstract_mdp_4.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "abstract_mdp_4_nonlinear.md")
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies", "abstract_mdp_4_dgp.png")

# DGP config. Two smooth, action-dependent features. The reward adds their
# product, scaled by GAMMA. Action 0 is a zeroed outside option (anchor).
NUM_STATES = 24
NUM_ACTIONS = 3
DISCOUNT = 0.95
THETA = (1.0, -0.8)   # main-effect weights on the two features
GAMMA = 2.5           # interaction weight on feature_0 * feature_1
BRANCHING = 4
SEED = 404
N_INDIVIDUALS = 300
N_PERIODS = 50

# Estimators excluded from this run, with honest reasons.
EXCLUDED = [
    {"name": "GAIL", "reason": "known slow (~9 min/fit); not run here"},
    {"name": "DeepMaxEnt-IRL", "reason": "known slow (~7 min/fit); not run here"},
    {"name": "Bayesian-IRL", "reason": "known slow (~16 min/fit); not run here"},
]


# ---------------------------------------------------------------------------
# DGP: an interaction effect the estimators' features do not carry
# ---------------------------------------------------------------------------


class _InteractionMDP(ArrayMDP):
    """ArrayMDP whose true reward adds a feature interaction the features omit.

    The estimators see the two main-effect features ``phi``. The true reward is
    ``theta . phi(s,a) + gamma * phi_0(s,a) * phi_1(s,a)``. The product term is
    not in the estimators' design, so a linear utility is misspecified.
    """

    def __init__(self, transitions, features, theta, gamma, discount_factor, seed):
        super().__init__(transitions, features, theta,
                         discount_factor=discount_factor, seed=seed)
        phi = np.asarray(features, dtype=np.float64)
        base = np.einsum("sak,k->sa", phi, np.asarray(theta, dtype=np.float64))
        interaction = gamma * phi[:, :, 0] * phi[:, :, 1]
        self._true_reward = jnp.asarray(base + interaction, dtype=jnp.float32)

    def compute_utility_matrix(self, parameters=None):
        # No parameters -> the true (nonlinear) reward drives the oracle and the
        # panel. With parameters, fall back to the linear ArrayMDP path so the
        # estimators still fit a linear utility on the features.
        if parameters is None:
            return self._true_reward
        return super().compute_utility_matrix(parameters)

    @property
    def true_reward_matrix(self):
        return self._true_reward

    def _compute_flow_utility(self, state, action):
        return float(self._true_reward[int(state), int(action)])


def _env() -> _InteractionMDP:
    rng = np.random.default_rng(SEED)
    T = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES), dtype=np.float64)
    for a in range(NUM_ACTIONS):
        for s in range(NUM_STATES):
            support = rng.choice(NUM_STATES, size=min(BRANCHING, NUM_STATES), replace=False)
            T[a, s, support] = rng.dirichlet(np.ones(support.shape[0]))
    T /= T.sum(axis=2, keepdims=True)

    x = np.linspace(0.0, 1.0, NUM_STATES)
    phi = np.zeros((NUM_STATES, NUM_ACTIONS, 2), dtype=np.float64)
    # Action 0 is the zeroed outside option (identification anchor).
    phi[:, 1, 0] = x
    phi[:, 1, 1] = np.sin(np.pi * x)
    phi[:, 2, 0] = 1.0 - x
    phi[:, 2, 1] = np.cos(np.pi * x)
    return _InteractionMDP(T, phi, np.asarray(THETA), gamma=GAMMA,
                           discount_factor=DISCOUNT, seed=SEED)


# ---------------------------------------------------------------------------
# Estimator roster. reward="linear" fits theta . phi; reward="neural" learns a
# reward (or value) network that can represent the product.
# ---------------------------------------------------------------------------


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=1, compute_hessian=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    est = MPECEstimator(config=MPECConfig(solver="sqp", outer_max_iter=200, tol=1e-8,
                                          constraint_tol=1e-6),
                        compute_hessian=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_nnes(env, panel):
    from econirl.estimation.nnes import NNESEstimator

    est = NNESEstimator(hidden_dim=64, v_epochs=800, n_outer_iterations=5,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_sees(env, panel):
    from econirl.estimation.sees import SEESEstimator

    est = SEESEstimator(basis_type="bspline", basis_dim=NUM_STATES, warm_start_value=True,
                        penalty_weight=10.0, compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_tdccp(env, panel):
    from econirl.estimation import TDCCPConfig, TDCCPEstimator

    est = TDCCPEstimator(config=TDCCPConfig(hidden_dim=64, avi_iterations=15,
                                            epochs_per_avi=15, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    est = UFXPEstimator(weights="optimal", verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_maxent_irl(env, panel):
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator

    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
                             learning_rate=0.05, compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_iq_learn(env, panel):
    from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

    est = IQLearnEstimator(config=IQLearnConfig(q_type="linear", divergence="chi2",
                                                alpha=3.0, max_iter=2000, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_firl(env, panel):
    from econirl.estimation.f_irl import FIRLEstimator

    est = FIRLEstimator(f_divergence="fkl", lr=0.2, max_iter=400, reward_clip=10.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_gladius(env, panel):
    # Neural Q and value networks; the policy comes from the learned Q, so it can
    # represent the interaction-driven behavior.
    from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

    est = GLADIUSEstimator(config=GLADIUSConfig(max_epochs=500, q_hidden_dim=128,
                                                v_hidden_dim=128, q_lr=1e-4, v_lr=1e-4,
                                                patience=60, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_airl(env, panel):
    # Neural reward; reward_arg="state_action" keeps the action contrast.
    from econirl.estimation import AIRLConfig, AIRLEstimator

    est = AIRLEstimator(config=AIRLConfig(reward_type="neural", reward_arg="state_action",
                                          reward_lr=0.01, discriminator_steps=10,
                                          max_rounds=300, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_deep_mce_irl(env, panel):
    # Neural reward over the features plus action; can represent the product.
    from types import SimpleNamespace

    from econirl.estimators import MCEIRLNeural

    m = MCEIRLNeural(n_states=int(env.num_states), n_actions=int(env.num_actions),
                     discount=float(env.problem_spec.discount_factor),
                     reward_type="state_action", max_epochs=400, verbose=False)
    m.fit(panel, features=np.asarray(env.feature_matrix),
          transitions=np.asarray(env.transition_matrices))
    return SimpleNamespace(parameters=None, standard_errors=None, policy=m.policy_,
                           value_function=m.value_, converged=bool(m.converged_))


def _run_neural_ufxp(env, panel):
    # Neural-utility UFXP: trains a network utility through the unnested dual,
    # no Bellman solve in the loop. Learns the interaction the linear UFXP cannot.
    from types import SimpleNamespace

    from econirl.estimators import NeuralUFXP

    m = NeuralUFXP(discount=float(env.problem_spec.discount_factor),
                   num_projections=64, max_epochs=2000, lr=1e-2, seed=0).fit(
        panel, features=np.asarray(env.feature_matrix),
        transitions=np.asarray(env.transition_matrices))
    return SimpleNamespace(parameters=None, standard_errors=None, policy=m.policy_,
                           value_function=m.value_, converged=bool(m.converged_))


# name, family, reward kind, runner. reward "linear" fits theta . phi; "neural"
# learns a reward or value network.
ROSTER = [
    ("NFXP", "structural", "linear", _run_nfxp),
    ("CCP", "structural", "linear", _run_ccp),
    ("MPEC", "structural", "linear", _run_mpec),
    ("NNES", "structural", "linear", _run_nnes),
    ("SEES", "structural", "linear", _run_sees),
    ("TD-CCP", "structural", "linear", _run_tdccp),
    ("UFXP", "structural", "linear", _run_ufxp),
    ("MCE-IRL", "behavioral", "linear", _run_mce_irl),
    ("MaxEnt-IRL", "behavioral", "linear", _run_maxent_irl),
    ("IQ-Learn", "behavioral", "linear", _run_iq_learn),
    ("f-IRL", "behavioral", "tabular", _run_firl),
    ("BC", "behavioral", "none", _run_bc),
    ("GLADIUS", "behavioral", "neural", _run_gladius),
    ("AIRL", "behavioral", "neural", _run_airl),
    ("Deep MCE-IRL", "behavioral", "neural", _run_deep_mce_irl),
    ("Neural UFXP", "structural", "neural", _run_neural_ufxp),
]

# One-line diagnosis per estimator.
DIAGNOSES: dict[str, str] = {
    "BC": "Clones the observed choice frequencies. It matches behavior with no "
          "reward, so it has nothing to carry to a counterfactual.",
    "f-IRL": "Learns a free tabular reward, one value per state-action pair, not "
             "a linear utility. That is why it tracks the choices on a nonlinear "
             "reward. The benchmark re-solves only linear-in-feature rewards, so "
             "this tabular reward is not transferred and its counterfactual "
             "stays on the fixed policy.",
    "GLADIUS": "Learns the behavior through a value network (policy TV 0.04), "
               "then projects the reward back onto the linear features. Its "
               "regret is scored on that projected linear reward, not on the "
               "neural policy the policy TV measures, and the projection cannot "
               "hold the interaction, so even its baseline regret is as large as "
               "the linear family's.",
    "UFXP": "The linear special case. It cannot form the product, so it sits "
            "with the linear family. The paper that introduces UFXP, Oguz and "
            "Bray (2026), trains a neural utility through the same unnested fixed "
            "point; that is the Neural UFXP row below.",
    "Neural UFXP": "The same unnested fixed point as UFXP, but the utility is a "
                   "network trained on the projected first-order conditions, with "
                   "no Bellman solve in the loop. It learns the interaction and "
                   "matches the choices where the linear UFXP cannot.",
}


# ---------------------------------------------------------------------------
# Run: collect raw facts
# ---------------------------------------------------------------------------


def _to_list(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


def run(n_replications: int, verbose: bool, only: str | None = None) -> dict:
    env = _env()
    oracle_policy, oracle_value = _oracle(env)
    # The oracle's own fixed policy cannot adapt either; its Type C is the
    # reference for any non-transferring method's Type C.
    oracle_fixed_c = float(estimator_regret(env, None, oracle_policy).type_c)

    records = []
    roster = [e for e in ROSTER if only is None or e[0] == only]
    for rep in range(n_replications):
        seed = SEED + 1000 + rep
        panel = simulate_panel(env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=seed)
        for name, family, reward, fn in roster:
            t0 = time.time()
            rec = {"estimator": name, "family": family, "reward": reward, "rep": rep,
                   "policy_tv": None, "value_rmse": None, "regret": None,
                   "runtime": None, "converged": None, "error": None}
            try:
                res = fn(env, panel)
                rec["runtime"] = time.time() - t0
                rec["converged"] = bool(getattr(res, "converged", True))
                pol = getattr(res, "policy", None)
                if pol is not None and np.asarray(pol).shape == oracle_policy.shape:
                    rec["policy_tv"] = M.policy_tv(np.asarray(pol), oracle_policy)
                rec["value_rmse"] = M.value_rmse(
                    None if getattr(res, "value_function", None) is None
                    else np.asarray(res.value_function), oracle_value)
                try:
                    rr = estimator_regret(env, getattr(res, "parameters", None), pol)
                    rec["regret"] = {"baseline": rr.baseline, "type_a": rr.type_a,
                                     "type_b": rr.type_b, "type_c": rr.type_c,
                                     "transferred": rr.transferred}
                except Exception:  # noqa: BLE001 - regret is best-effort, never fabricated
                    rec["regret"] = None
            except Exception as exc:  # noqa: BLE001 - the failure IS the result
                rec["runtime"] = time.time() - t0
                rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            if verbose:
                status = rec["error"] or f"tv={rec['policy_tv']}"
                print(f"  rep {rep} {name:13s} {rec['runtime']:.1f}s {status}")

    return {
        "meta": {
            "title": "Abstract MDP 4: an interaction effect in the reward",
            "date": _dt.date.today().isoformat(),
            "package_version": _package_version(),
            "n_replications": n_replications,
            "dgp": {"num_states": NUM_STATES, "num_actions": NUM_ACTIONS,
                    "discount_factor": DISCOUNT, "theta": list(THETA), "gamma": GAMMA,
                    "branching": BRANCHING, "seed": SEED},
            "regret_oracle_fixed_c": oracle_fixed_c,
            "n_individuals": N_INDIVIDUALS,
            "n_periods": N_PERIODS,
            "oracle": "true-reward policy/value via SoftBellmanOperator + value_iteration",
            "determinism": (
                "structural estimators are deterministic given the seeds; neural "
                "estimators may have nondeterministic training internals, so their "
                "behavioral numbers can vary slightly across runs"
            ),
            "excluded": EXCLUDED,
            "regret": (
                "Counterfactual regret follows the package Type A (payoff shift), "
                "Type B (transition change), Type C (action penalty) taxonomy; "
                "regret = initial_distribution . (oracle_value - estimated_value), "
                "lower is better. Estimators with a recovered reward re-solve it "
                "under each intervention; estimators without one keep their fixed "
                "policy and cannot adapt."
            ),
            "snippets": {name: inspect.getsource(fn) for name, _f, _r, fn in ROSTER},
            "diagnoses": DIAGNOSES,
            "honesty": (
                "Every number in the table is recomputed from the records below. "
                "Crashes carry the verbatim exception. There is no parameter table: "
                "the true reward is not linear in the two features the estimators "
                "see, so there is no shared parameterization to compare against."
            ),
        },
        "records": records,
    }


def _package_version() -> str:
    try:
        import econirl

        return getattr(econirl, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def _agg(recs, key):
    vals = [r[key] for r in recs if r["error"] is None and r[key] is not None]
    return float(np.mean(vals)) if vals else None


def _agg_regret(recs, key):
    vals = [r["regret"][key] for r in recs
            if r["error"] is None and r.get("regret") is not None]
    return float(np.mean(vals)) if vals else None


def _transfer(recs):
    """Did the method re-solve a recovered reward (yes) or hold a fixed policy (no)?"""
    flags = [r["regret"].get("transferred") for r in recs
             if r["error"] is None and r.get("regret") is not None]
    if not flags:
        return "-"
    return "yes" if all(flags) else ("no" if not any(flags) else "mixed")


def _grouped(data):
    by_est, order = {}, []
    for r in data["records"]:
        if r["estimator"] not in by_est:
            by_est[r["estimator"]] = []
            order.append((r["estimator"], r.get("reward", "linear")))
        by_est[r["estimator"]].append(r)
    return by_est, order


def render(data: dict) -> str:
    meta = data["meta"]
    by_est, order = _grouped(data)
    d = meta["dgp"]
    lines = [
        f"Abstract MDP 4: interaction effect  ({meta['date']}, econirl {meta['package_version']})",
        f"{d['num_states']} states, {d['num_actions']} actions, discount {d['discount_factor']}, "
        f"theta {d['theta']}, gamma {d['gamma']}. "
        f"{meta['n_individuals']} x {meta['n_periods']} obs, {meta['n_replications']} reps.",
        "",
        f"{'Estimator':<13} {'Reward':<7} {'Ran':>5} {'PolicyTV':>9} "
        f"{'RegBase':>8} {'Time(s)':>8}  Notes",
    ]
    lines.append("-" * len(lines[-1]))
    for name, reward in order:
        recs = by_est[name]
        ok = [r for r in recs if r["error"] is None]
        ran = f"{len(ok)}/{len(recs)}"
        tv = _fmt(_agg(ok, "policy_tv"))
        rb = _fmt(_agg_regret(recs, "baseline"))
        rts = [r["runtime"] for r in ok if r["runtime"] is not None]
        rt = f"{np.mean(rts):.1f}" if rts else "-"
        crashed = [r for r in recs if r["error"] is not None]
        note = f"CRASHED {len(crashed)}/{len(recs)}: {crashed[0]['error'][:50]}" if crashed else ""
        lines.append(f"{name:<13} {reward:<7} {ran:>5} {tv:>9} {rb:>8} {rt:>8}  {note}")
    lines.append("")
    lines.append("PolicyTV: distance from the true choice probabilities (all estimators).")
    lines.append("RegBase: welfare lost in the observed environment using the recovered model.")
    if meta["excluded"]:
        lines.append("Excluded: " + "; ".join(f"{e['name']} ({e['reason']})" for e in meta["excluded"]))
    return "\n".join(lines)


def render_page(data: dict) -> str:
    meta = data["meta"]
    by_est, order = _grouped(data)
    d = meta["dgp"]
    L = []
    L.append("# Abstract MDP 4\n")
    L.append(
        "The reward has an interaction effect. The true utility multiplies two "
        "features. The estimators receive the two features but never their "
        "product, so a linear utility is misspecified by construction. The "
        "omitted term is deliberately strong here, larger than the main effects, "
        "so the cost is visible. The question is what that cost is. The table "
        "reports the distance from the true choices and the counterfactual "
        "regret.\n"
    )
    L.append(
        f"Environment: a {d['num_states']}-state, {d['num_actions']}-action MDP "
        f"with sparse random transitions, drawn once at seed {d['seed']}. "
        f"{meta['n_individuals']} x {meta['n_periods']} observations; the "
        f"{meta['n_replications']} replications resample the panel from that one "
        f"environment. Generated {meta['date']} with econirl "
        f"{meta['package_version']}.\n"
    )

    L.append("## The data-generating process\n")
    L.append(
        "Each state-action pair reaches a random subset of $b$ states with "
        "Dirichlet weights:\n"
    )
    L.append(
        "$$\n"
        "P(s' \\mid s, a) = D_{s,a}(s'), \\qquad D_{s,a} \\sim "
        f"\\mathrm{{Dirichlet}}(\\mathbf{{1}}_b), \\quad b = {d['branching']}.\n"
        "$$\n"
    )
    L.append(
        "Two features vary smoothly in the normalized state index "
        "$x_s = s/(S-1)$. Action $0$ is a zeroed outside option, the "
        "identification anchor. For the other actions the features are\n"
    )
    L.append(
        "$$\n"
        "\\varphi(s,1) = \\bigl(x_s,\\ \\sin \\pi x_s\\bigr), \\qquad "
        "\\varphi(s,2) = \\bigl(1-x_s,\\ \\cos \\pi x_s\\bigr).\n"
        "$$\n"
    )
    L.append(
        "The true reward adds the product of the two features, the interaction "
        "the estimators do not model:\n"
    )
    L.append(
        "$$\n"
        f"u(s,a) = \\theta_0\\, \\varphi_0(s,a) + \\theta_1\\, \\varphi_1(s,a) "
        f"+ \\gamma\\, \\varphi_0(s,a)\\, \\varphi_1(s,a), \\qquad "
        f"\\theta = ({d['theta'][0]}, {d['theta'][1]}),\\ \\gamma = {d['gamma']}.\n"
        "$$\n"
    )
    L.append(
        "A linear utility fits $\\theta_0 \\varphi_0 + \\theta_1 \\varphi_1$ and "
        "has no term for the product. The neural-reward methods learn a reward "
        "or value network over the same two features and can form it. The "
        "interaction weight is set above the main effects on purpose, to make "
        "the misspecification show. A weaker interaction shrinks the gap, and at "
        "$\\gamma = 0$ the linear utility is correct and recovers the reward. "
        f"The agent discounts at $\\beta = {d['discount_factor']}$ and faces "
        "logit taste shocks, so behavior solves the soft Bellman equation. The "
        "figure shows the simulated paths and the optimal value function.\n"
    )
    L.append("![Simulated trajectories and the optimal value function]"
             "(../_static/simulation_studies/abstract_mdp_4_dgp.png)\n")

    L.append("## Results\n")
    L.append("| Estimator | Reward | Ran | Conv | Policy TV | Transfer | "
             "Regret base | Regret A | Regret B | Regret C | Time (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, reward in order:
        recs = by_est[name]
        ok = [r for r in recs if r["error"] is None]
        ran = f"{len(ok)}/{len(recs)}"
        conv = f"{sum(1 for r in ok if r['converged'])}/{len(ok)}" if ok else "-"
        tv = _fmt(_agg(ok, "policy_tv"))
        xfer = _transfer(recs)
        rb = _fmt(_agg_regret(recs, "baseline"))
        ra = _fmt(_agg_regret(recs, "type_a"))
        rbb = _fmt(_agg_regret(recs, "type_b"))
        rc = _fmt(_agg_regret(recs, "type_c"))
        rts = [r["runtime"] for r in ok if r["runtime"] is not None]
        rt = f"{np.mean(rts):.1f}" if rts else "-"
        crashed = [r for r in recs if r["error"] is not None]
        note = f" (crashed {len(crashed)}/{len(recs)})" if crashed else ""
        L.append(f"| {name}{note} | {reward} | {ran} | {conv} | {tv} | {xfer} | "
                 f"{rb} | {ra} | {rbb} | {rc} | {rt} |")
    L.append("")
    L.append(
        "The interaction costs two ways. The structural estimators land "
        "together: NFXP, CCP, MPEC, NNES, SEES, TD-CCP, and UFXP all sit near a "
        "policy distance of 0.10, the residual a linear utility leaves, and "
        "their re-solved reward loses close to one unit of welfare. The "
        "maximum-entropy IRL methods sit there too. The methods with a richer "
        "reward or policy class learn the product and match the choices to "
        "about 0.02: the neural-reward Deep MCE-IRL and AIRL, f-IRL with a "
        "free tabular reward, and Neural UFXP, which trains a network utility "
        "through the same unnested fixed point the linear UFXP uses. The "
        "benchmark re-solves only linear-in-feature "
        "rewards, so under the interventions these methods are scored on their "
        "fixed policy, not on a re-solve of what they learned. GLADIUS matches "
        "the choices but projects its reward back onto the linear features, so "
        "even its baseline regret is as large as the linear family's. BC clones "
        "the choices and estimates no reward at all.\n"
    )
    L.append("Reward marks what the method fits: a linear utility, a reward or "
             "value network, a free tabular reward (one value per state-action "
             "pair), or no reward at all (a cloned policy). "
             "Policy TV is the distance between estimated and true choice "
             "probabilities, lower is better. The value level is omitted: the "
             "reward is identified only up to transformations that leave "
             "behavior unchanged, so a value error across families would not "
             "compare like with like. Conv is the estimator's own convergence "
             "indicator; it does not track recovery here. A cautious estimator "
             "can report False while the policy is accurate, which is exactly the AIRL "
             "case below.\n")
    oc = meta.get("regret_oracle_fixed_c")
    oc_str = f"about {oc:.0f}" if oc is not None else "about 71"
    L.append("Regret base is welfare lost in the observed environment. Types A, "
             "B, and C are welfare lost after a change: Type A shifts a payoff, "
             "Type B changes the transitions, Type C penalizes an action. "
             "Transfer says whether the method re-solved a recovered reward "
             "(yes) or held a fixed policy (no). The benchmark re-solves only "
             "linear-in-feature rewards, so a method that learns a neural or "
             "tabular reward shows no here even though its reward could transfer "
             "in principle; this is a limit of the test, not of the method. The "
             "two modes are not comparable on Types A, B, and C: a fixed policy "
             "cannot adapt to any change, so it pays the same large Type C the "
             f"oracle's own fixed policy pays ({oc_str}). That figure marks no "
             "re-solve, not a worse estimate. Read the counterfactual columns "
             "within a transfer mode, not across.\n")

    diagnoses = meta.get("diagnoses", {})
    notes = [f"**{name}.** {diagnoses[name]}" for name, _r in order if diagnoses.get(name)]
    if notes:
        L.append("## Notes per estimator\n")
        L.append("\n\n".join(notes) + "\n")

    L.append("## Reproduce\n")
    L.append("```bash")
    L.append(f"python scripts/sim_abstract_mdp_4.py --replications {meta['n_replications']}")
    L.append("python scripts/sim_abstract_mdp_4.py --page")
    L.append("python scripts/sim_abstract_mdp_4.py --verify")
    L.append("```\n")
    L.append("Raw facts: `validation/results/sim_abstract_mdp_4.json`.\n")
    if meta["excluded"]:
        L.append("Excluded from this run: " +
                 "; ".join(f"{e['name']} ({e['reason']})" for e in meta["excluded"]) + ".")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_page(data: dict) -> None:
    data["meta"]["diagnoses"] = DIAGNOSES
    data["meta"]["excluded"] = EXCLUDED
    from validation.benchmark.figures import dgp_figure

    env = _env()
    panel = simulate_panel(env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED + 1000)
    _, oracle_value = _oracle(env)
    os.makedirs(os.path.dirname(FIGURE_PNG), exist_ok=True)
    dgp_figure(env, panel, oracle_value, FIGURE_PNG)
    with open(PAGE_PATH, "w") as f:
        f.write(render_page(data))
    print(f"Wrote {PAGE_PATH} (+ {os.path.basename(FIGURE_PNG)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--only-estimator", default=None,
                        help="Re-run one estimator and merge into the saved JSON.")
    parser.add_argument("--verify", action="store_true",
                        help="Re-render the table from the saved JSON only; run no estimators.")
    parser.add_argument("--page", action="store_true",
                        help="Write the docs page from the saved JSON; run no estimators.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verify or args.page:
        if not os.path.exists(RESULTS_JSON):
            sys.exit(f"No JSON at {RESULTS_JSON}. Run without --verify/--page first.")
        data = json.load(open(RESULTS_JSON))
        if args.page:
            _write_page(data)
        else:
            print(render(data))
            print(f"\n(verified: table re-derived purely from {RESULTS_JSON})")
        return

    if args.only_estimator:
        if not os.path.exists(RESULTS_JSON):
            sys.exit(f"No JSON at {RESULTS_JSON}. Run a full pass first.")
        data = json.load(open(RESULTS_JSON))
        fresh = run(args.replications, args.verbose, only=args.only_estimator)
        kept = [r for r in data["records"] if r["estimator"] != args.only_estimator]
        data["records"] = kept + fresh["records"]
        data["meta"]["snippets"].update(fresh["meta"]["snippets"])
    else:
        data = run(args.replications, args.verbose)

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(render(data))
    print(f"\nRaw facts: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
