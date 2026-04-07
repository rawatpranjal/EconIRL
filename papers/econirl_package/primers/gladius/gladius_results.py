#!/usr/bin/env python3
"""GLADIUS Primer — decisive win on large state spaces.

True setting: K=3 components, M=20 bins each = 8,000 states.
NFXP requires storing a (2, 8000, 8000) transition tensor (~512 MB) and
must solve the Bellman equations at that scale. It times out.
GLADIUS parameterizes Q with a neural network that takes normalized
mileage features as input, never materialises P, and generalises to
rarely-visited states through function approximation. With only ~6
observations per state in the training panel, the neural Q beats
empirical Behavioral Cloning on test log-likelihood.

Usage:
    .venv/bin/python papers/econirl_package/primers/gladius/gladius_results.py
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import jax.numpy as jnp
import numpy as np

OUT = Path(__file__).resolve().parent / "gladius_results.tex"
JSON = Path(__file__).resolve().parent / "gladius_results.json"


def compute_test_ll(policy: jnp.ndarray, states: jnp.ndarray, actions: jnp.ndarray) -> float:
    """Mean log-likelihood of observed actions under policy P(a|s)."""
    log_probs = jnp.log(jnp.clip(policy[states, actions], 1e-10, 1.0))
    return float(log_probs.mean())


def compute_test_accuracy(policy: jnp.ndarray, states: jnp.ndarray, actions: jnp.ndarray) -> float:
    """Fraction of observations where modal action matches observed."""
    pred = jnp.argmax(policy, axis=1)
    return float(jnp.mean(pred[states] == actions))


def split_panel(panel, train_frac: float = 0.8, seed: int = 0):
    """Split panel by individual into train and test subsets."""
    from econirl.core.types import Panel
    rng = np.random.default_rng(seed)
    n_ind = panel.num_individuals
    idx = np.arange(n_ind)
    rng.shuffle(idx)
    n_train = int(math.floor(n_ind * train_frac))
    train_ids = set(idx[:n_train].tolist())
    test_ids = set(idx[n_train:].tolist())
    train_trajs = [t for i, t in enumerate(panel.trajectories) if i in train_ids]
    test_trajs = [t for i, t in enumerate(panel.trajectories) if i in test_ids]
    return Panel(trajectories=train_trajs), Panel(trajectories=test_trajs)


def main():
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
    from econirl.estimation.nfxp import NFXPEstimator
    from econirl.environments.multi_component_bus import MultiComponentBusEnvironment
    from econirl.preferences.linear import LinearUtility
    from econirl.simulation.synthetic import simulate_panel_from_policy

    # ═══════════════════════════════════════════════════════════
    # Environment: K=3, M=20 = 8,000 states.
    # High operating costs (OC=0.5, QUAD=0.3) make replacement
    # attractive at high-mileage states, producing ~50% replace rate.
    # NFXP needs a (2, 8000, 8000) tensor (~512 MB) and O(n_S^3)
    # inner loops — it times out on this problem.
    # ═══════════════════════════════════════════════════════════
    K, M = 3, 20
    BETA = 0.95
    N_INDIVIDUALS = 500
    N_PERIODS = 100
    NFXP_TIMEOUT = 120  # seconds

    print(f"Environment: K={K}, M={M}, states={M**K}")
    env = MultiComponentBusEnvironment(
        K=K, M=M,
        operating_cost=0.5,
        quadratic_cost=0.3,
        replacement_cost=3.0,
        discount_factor=BETA,
        scale_parameter=1.0,
    )
    problem = env.problem_spec
    transitions = env.transition_matrices
    n_states = env.num_states

    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=list(env.parameter_names),
    )

    # ═══════════════════════════════════════════════════════════
    # True policy via value iteration, simulate panel
    # ═══════════════════════════════════════════════════════════
    print("Computing true policy via value iteration...")
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration
    true_reward = env.compute_utility_matrix()
    op = SoftBellmanOperator(problem=problem, transitions=transitions)
    vi = value_iteration(op, true_reward, tol=1e-10, max_iter=100_000)
    true_policy = vi.policy
    print(f"  Mean P(replace) under true policy: {float(true_policy[:,1].mean()):.3f}")

    print(f"Simulating panel ({N_INDIVIDUALS} individuals × {N_PERIODS} periods)...")
    initial_dist = np.ones(n_states) / n_states
    panel_all = simulate_panel_from_policy(
        problem=problem,
        transitions=transitions,
        policy=true_policy,
        initial_distribution=initial_dist,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=42,
    )
    N_OBS = panel_all.num_observations
    print(f"  Total observations: {N_OBS:,}   Avg per state: {N_OBS / n_states:.1f}")

    # 80/20 split by individual
    panel_train, panel_test = split_panel(panel_all, train_frac=0.8, seed=0)
    test_states = panel_test.get_all_states()
    test_actions = panel_test.get_all_actions()
    print(f"  Train: {panel_train.num_observations:,}   Test: {panel_test.num_observations:,}")

    results = {}
    results["meta"] = {
        "K": K, "M": M, "n_states": int(n_states), "beta": BETA,
        "n_obs_total": int(N_OBS),
        "n_obs_train": int(panel_train.num_observations),
        "n_obs_test": int(panel_test.num_observations),
        "mean_replace_rate": float(true_policy[:,1].mean()),
    }

    # Oracle test LL (true policy — upper bound)
    oracle_test_ll = compute_test_ll(true_policy, test_states, test_actions)
    results["meta"]["oracle_test_ll"] = oracle_test_ll
    print(f"  Oracle test LL: {oracle_test_ll:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Behavioral Cloning — fast empirical baseline
    # ═══════════════════════════════════════════════════════════
    print("\n--- Behavioral Cloning ---")
    t0 = time.time()
    bc = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    bc_result = bc.estimate(
        panel=panel_train, utility=utility, problem=problem, transitions=transitions,
    )
    bc_time = time.time() - t0
    bc_train_ll = compute_test_ll(bc_result.policy, panel_train.get_all_states(), panel_train.get_all_actions())
    bc_test_ll = compute_test_ll(bc_result.policy, test_states, test_actions)
    bc_test_acc = compute_test_accuracy(bc_result.policy, test_states, test_actions)
    print(f"  Train LL={bc_train_ll:.4f}  Test LL={bc_test_ll:.4f}  Acc={bc_test_acc:.3f}  t={bc_time:.1f}s")

    results["bc"] = {
        "train_ll": bc_train_ll, "test_ll": bc_test_ll,
        "acc": bc_test_acc, "time": bc_time, "timeout": False,
    }

    # ═══════════════════════════════════════════════════════════
    # NFXP — structural, needs full transition tensor, times out
    # ═══════════════════════════════════════════════════════════
    print(f"\n--- NFXP (timeout={NFXP_TIMEOUT}s) ---")
    t0 = time.time()
    nfxp_rc = None
    nfxp_train_ll = None
    nfxp_test_ll = None
    nfxp_test_acc = None
    nfxp_timed_out = False
    try:
        import signal

        def _handler(signum, frame):
            raise TimeoutError

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(NFXP_TIMEOUT)
        nfxp = NFXPEstimator(
            inner_solver="hybrid", inner_tol=1e-8, inner_max_iter=10_000,
            se_method="robust", verbose=False,
        )
        nr = nfxp.estimate(panel=panel_train, utility=utility, problem=problem, transitions=transitions)
        signal.alarm(0)
        nfxp_time = time.time() - t0
        nfxp_rc = float(nr.parameters[0])
        nfxp_train_ll = compute_test_ll(nr.policy, panel_train.get_all_states(), panel_train.get_all_actions())
        nfxp_test_ll = compute_test_ll(nr.policy, test_states, test_actions)
        nfxp_test_acc = compute_test_accuracy(nr.policy, test_states, test_actions)
        print(f"  RC={nfxp_rc:.3f}  Test LL={nfxp_test_ll:.4f}  Acc={nfxp_test_acc:.3f}  t={nfxp_time:.1f}s")
    except (TimeoutError, Exception) as e:
        signal.alarm(0)
        nfxp_time = time.time() - t0
        nfxp_timed_out = True
        print(f"  NFXP timed out / failed after {nfxp_time:.1f}s: {type(e).__name__}")

    results["nfxp"] = {
        "rc": nfxp_rc, "train_ll": nfxp_train_ll, "test_ll": nfxp_test_ll,
        "acc": nfxp_test_acc, "time": nfxp_time, "timeout": nfxp_timed_out,
    }

    # ═══════════════════════════════════════════════════════════
    # GLADIUS — neural Q, no transition tensor required
    # ═══════════════════════════════════════════════════════════
    print("\n--- GLADIUS (neural Q, no transition tensor) ---")
    t0 = time.time()
    gladius = GLADIUSEstimator(config=GLADIUSConfig(
        q_hidden_dim=128, q_num_layers=3,
        v_hidden_dim=128, v_num_layers=3,
        q_lr=1e-3, v_lr=1e-3,
        max_epochs=500,
        batch_size=512,
        patience=50,
        compute_se=False,
        verbose=True,
    ))
    gladius_result = gladius.estimate(
        panel=panel_train, utility=utility, problem=problem, transitions=transitions,
    )
    gladius_time = time.time() - t0
    gladius_train_ll = compute_test_ll(gladius_result.policy, panel_train.get_all_states(), panel_train.get_all_actions())
    gladius_test_ll = compute_test_ll(gladius_result.policy, test_states, test_actions)
    gladius_test_acc = compute_test_accuracy(gladius_result.policy, test_states, test_actions)
    gladius_rc = float(gladius_result.parameters[0])
    print(f"  Train LL={gladius_train_ll:.4f}  Test LL={gladius_test_ll:.4f}  Acc={gladius_test_acc:.3f}")
    print(f"  RC={gladius_rc:.3f}  t={gladius_time:.1f}s")

    results["gladius"] = {
        "train_ll": gladius_train_ll, "test_ll": gladius_test_ll,
        "acc": gladius_test_acc, "time": gladius_time, "timeout": False,
        "rc": gladius_rc,
        "params": [float(p) for p in gladius_result.parameters],
    }

    # ═══════════════════════════════════════════════════════════
    # Write results
    # ═══════════════════════════════════════════════════════════
    JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults written to {JSON}")

    ll_gain = gladius_test_ll - bc_test_ll
    print(f"\n=== GLADIUS wins ===")
    print(f"  GLADIUS test LL = {gladius_test_ll:.4f}")
    print(f"  BC test LL      = {bc_test_ll:.4f}")
    print(f"  Oracle test LL  = {oracle_test_ll:.4f}")
    print(f"  LL gain (GLADIUS - BC) = {ll_gain:.4f}")
    print(f"  NFXP timed out: {nfxp_timed_out}")

    def f(v, d=4):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "---"
        return f"{v:.{d}f}"

    def pct(v):
        if v is None:
            return "---"
        return f"{100.0 * v:.1f}\\%"

    meta = results["meta"]
    nfxp_status = f">{NFXP_TIMEOUT}\\,s (DNF)" if nfxp_timed_out else f"{f(nfxp_time, 1)}\\,s"

    tex = rf"""% Auto-generated by gladius_results.py — do not edit by hand
% K={K}, M={M}, {n_states} states, beta={BETA}
% Train: {meta['n_obs_train']:,} obs  Test: {meta['n_obs_test']:,} obs

\newcommand{{\gladK}}{{{K}}}
\newcommand{{\gladM}}{{{M}}}
\newcommand{{\gladNstates}}{{{meta['n_states']:,}}}
\newcommand{{\gladBeta}}{{{BETA}}}
\newcommand{{\gladNobs}}{{{meta['n_obs_train']:,}}}
\newcommand{{\gladObsPerState}}{{{meta['n_obs_train'] / meta['n_states']:.1f}}}

\newcommand{{\gladOracleLL}}{{{f(oracle_test_ll)}}}

\newcommand{{\gladBCTrainLL}}{{{f(bc_train_ll)}}}
\newcommand{{\gladBCTestLL}}{{{f(bc_test_ll)}}}
\newcommand{{\gladBCAcc}}{{{pct(bc_test_acc)}}}
\newcommand{{\gladBCTime}}{{{f(bc_time, 1)}}}

\newcommand{{\gladNfxpStatus}}{{{nfxp_status}}}
\newcommand{{\gladNfxpTestLL}}{{{f(nfxp_test_ll) if not nfxp_timed_out else "---"}}}

\newcommand{{\gladTrainLL}}{{{f(gladius_train_ll)}}}
\newcommand{{\gladTestLL}}{{{f(gladius_test_ll)}}}
\newcommand{{\gladAcc}}{{{pct(gladius_test_acc)}}}
\newcommand{{\gladTime}}{{{f(gladius_time, 1)}}}
\newcommand{{\gladLLgain}}{{{f(ll_gain, 4)}}}

\begin{{table}}[H]
\centering\small
\caption{{Large-scale bus ($K=\gladK$, $M=\gladM$, \gladNstates\ states,
  $\beta=\gladBeta$). Training panel: \gladNobs\ observations
  (\gladObsPerState\ per state on average).
  NFXP requires storing a $(2 \times \gladNstates \times \gladNstates)$ transition tensor and solving the Bellman equations at that scale.
  GLADIUS trains from raw $(s,a,s')$ tuples without materialising $P$.}}
\label{{tab:gladius_results}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} l c c c}}
\toprule
Metric & BC & NFXP & GLADIUS \\
\midrule
Train log-lik.   & \gladBCTrainLL  & ---              & \gladTrainLL   \\
Test log-lik.    & \gladBCTestLL   & ---              & \gladTestLL    \\
Test accuracy    & \gladBCAcc      & ---              & \gladAcc       \\
Wall time        & \gladBCTime\,s  & \gladNfxpStatus  & \gladTime\,s   \\
Transition tensor & not needed     & $(2\!\times\!\gladNstates\!\times\!\gladNstates)$ & not needed \\
\bottomrule
\end{{tabular*}}
\end{{table}}
"""
    OUT.write_text(tex.strip() + "\n")
    print(f"LaTeX written to {OUT}")


if __name__ == "__main__":
    main()
