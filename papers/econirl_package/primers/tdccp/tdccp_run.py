#!/usr/bin/env python3
"""TD-CCP Primer — decisive win: continuous state with discretization bias.

The experiment: a 1-D firm value problem where productivity follows AR(1).
CCP must discretize the continuous state into K bins, introducing
approximation bias that grows with bin width. TD-CCP uses polynomial
basis functions directly on the continuous state and never estimates the
transition density. Per Adusumilli and Eckardt (2025) Table 2, TD-CCP
achieves ~10x lower parameter RMSE than coarsely-discretized CCP.

Usage:
    cd /path/to/econirl
    .venv/bin/python papers/econirl_package/primers/tdccp/tdccp_run.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm as scipy_norm

OUT = Path(__file__).resolve().parent / "tdccp_results.tex"

# True structural parameters
# theta_v: productivity value (slope of utility in z_norm)
# theta_FC: fixed operating cost (negative = cost)
TRUE_THETA_V = 5.0
TRUE_THETA_FC = -1.0
TRUE_PARAMS = np.array([TRUE_THETA_V, TRUE_THETA_FC])

# Model parameters
N_FINE = 100      # states on [0,1] for the "true" continuous model
BETA = 0.95
PERSISTENCE = 0.8  # AR(1) persistence rho
NOISE_STD = 0.10   # innovation std (sigma_eps), chosen so sigma_z ~ 0.17

# Data
N_INDIVIDUALS = 3000
N_PERIODS = 5
SEED = 42

# CCP discretization levels
CCP_K_VALUES = [5, 20]


def build_tauchen_transitions(n_bins: int, persistence: float, noise_std: float) -> np.ndarray:
    """Tauchen (1986) discretization of AR(1) z_{t+1} = rho*z_t + (1-rho)*0.5 + eps.

    Grid is evenly spaced on [0, 1] with long-run mean 0.5.
    noise_std is the innovation standard deviation (sigma_eps).
    """
    grid = np.linspace(0.0, 1.0, n_bins)
    half_h = (grid[1] - grid[0]) / 2.0 if n_bins > 1 else 0.5

    trans = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i in range(n_bins):
        mu_next = persistence * grid[i] + (1.0 - persistence) * 0.5  # AR(1) mean
        for j in range(n_bins):
            lo = (grid[j] - half_h) if j > 0 else -np.inf
            hi = (grid[j] + half_h) if j < n_bins - 1 else np.inf
            trans[i, j] = scipy_norm.cdf((hi - mu_next) / noise_std) - scipy_norm.cdf(
                (lo - mu_next) / noise_std
            )
        trans[i] /= trans[i].sum()  # normalize rows
    return trans.astype(np.float32)


def build_firm_transitions(n_states: int) -> np.ndarray:
    """Build (2, n_states, n_states) transition matrix for the firm model.

    Both actions (exit=0, operate=1) share the same productivity transition:
    productivity is exogenous and unaffected by the firm's decision.
    """
    ar1 = build_tauchen_transitions(n_states, PERSISTENCE, NOISE_STD)
    transitions = np.zeros((2, n_states, n_states), dtype=np.float32)
    transitions[0] = ar1  # exit
    transitions[1] = ar1  # operate
    return transitions


def build_firm_features(n_states: int) -> np.ndarray:
    """Build (n_states, 2, 2) feature matrix.

    Features:
      phi(s, operate=1, 0) = z_norm = s / (n_states-1)   [theta_v coefficient]
      phi(s, operate=1, 1) = 1.0                          [theta_FC coefficient]
      phi(s, exit=0, :)    = 0.0                          [exit: zero utility]

    So u(s, operate) = theta_v * z_norm + theta_FC * 1.0
       u(s, exit)    = 0.0  (normalized to zero)
    """
    features = np.zeros((n_states, 2, 2), dtype=np.float32)
    for s in range(n_states):
        z_norm = s / max(n_states - 1, 1)
        features[s, 1, 0] = z_norm  # theta_v
        features[s, 1, 1] = 1.0     # theta_FC (enters as +1, true theta_FC is negative)
    return features


def coarsen_panel(panel, K: int, N: int):
    """Remap panel states from N fine bins to K coarse bins."""
    from econirl.core.types import Trajectory, Panel as EconPanel
    coarse_trajs = []
    for traj in panel.trajectories:
        s_c = np.floor(np.array(traj.states, dtype=float) * K / N).clip(0, K - 1).astype(int)
        n_c = np.floor(np.array(traj.next_states, dtype=float) * K / N).clip(0, K - 1).astype(int)
        coarse_trajs.append(
            Trajectory(
                states=jnp.array(s_c, dtype=jnp.int32),
                actions=traj.actions,
                next_states=jnp.array(n_c, dtype=jnp.int32),
                individual_id=traj.individual_id,
            )
        )
    return EconPanel(coarse_trajs)


def estimate_freq_transitions(panel, K: int) -> np.ndarray:
    """Frequency-estimate (2, K, K) transition matrix with Laplace smoothing."""
    counts = np.ones((2, K, K), dtype=np.float64) * 0.5  # add-half smoothing
    for traj in panel.trajectories:
        for s, a, sp in zip(traj.states, traj.actions, traj.next_states):
            counts[int(a), int(s), int(sp)] += 1.0
    P = counts / counts.sum(axis=-1, keepdims=True)
    return P.astype(np.float32)


def build_coarse_features(K: int) -> np.ndarray:
    """(K, 2, 2) feature matrix using bin centroids for z_norm."""
    features = np.zeros((K, 2, 2), dtype=np.float32)
    for j in range(K):
        z_centroid = (j + 0.5) / K
        features[j, 1, 0] = z_centroid  # theta_v
        features[j, 1, 1] = 1.0         # theta_FC
    return features


def policy_accuracy(est_policy: np.ndarray, true_policy: np.ndarray) -> float:
    """Fraction of states where argmax actions agree."""
    return float(np.mean(np.argmax(est_policy, axis=1) == np.argmax(true_policy, axis=1)))


def param_rmse(params: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error across parameters."""
    return float(np.sqrt(np.mean((params - true) ** 2)))


def main():
    from econirl.core.types import DDCProblem
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration
    from econirl.estimation.ccp import CCPEstimator
    from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from econirl.preferences.linear import LinearUtility
    from econirl.simulation.synthetic import simulate_panel_from_policy

    n_obs = N_INDIVIDUALS * N_PERIODS
    print("TD-CCP Primer — firm value model (continuous state demonstration)")
    print(f"  N_FINE={N_FINE} states, beta={BETA}, {n_obs:,} obs")
    print(f"  True: theta_v={TRUE_THETA_V}, theta_FC={TRUE_THETA_FC}")

    # ── Build fine environment ──
    trans_fine = build_firm_transitions(N_FINE)
    feat_fine = build_firm_features(N_FINE)
    transitions_jnp = jnp.array(trans_fine)
    features_jnp = jnp.array(feat_fine)

    problem_fine = DDCProblem(num_states=N_FINE, num_actions=2, discount_factor=BETA)
    utility_fine = LinearUtility(features_jnp, parameter_names=["theta_v", "theta_fc"])

    # ── Compute true optimal policy ──
    print("\n  Computing true optimal policy...")
    true_params_jnp = jnp.array(TRUE_PARAMS, dtype=jnp.float32)
    true_reward = utility_fine.compute(true_params_jnp)
    operator = SoftBellmanOperator(problem_fine, transitions_jnp)
    vi_true = value_iteration(operator, true_reward, tol=1e-10, max_iter=5000)
    true_policy = np.array(vi_true.policy)
    print(f"  Converged in {vi_true.num_iterations} VI iterations")

    # ── Simulate panel ──
    print(f"\n  Simulating {N_INDIVIDUALS} x {N_PERIODS} = {n_obs:,} obs...")
    initial_dist = jnp.ones(N_FINE) / N_FINE  # uniform over all states
    panel_fine = simulate_panel_from_policy(
        problem=problem_fine,
        transitions=transitions_jnp,
        policy=vi_true.policy,
        initial_distribution=initial_dist,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=SEED,
    )

    results = {}

    # ── BC baseline ──
    print("\n  [1/4] Running BC baseline...")
    t0 = time.time()
    bc = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    bc_res = bc.estimate(panel_fine, utility_fine, problem_fine, transitions_jnp)
    bc_time = time.time() - t0
    bc_acc = policy_accuracy(np.array(bc_res.policy), true_policy)
    results["bc"] = {"acc": bc_acc, "time": bc_time}
    print(f"    policy_acc={bc_acc:.3f}, time={bc_time:.2f}s")

    # ── CCP with coarse discretizations ──
    results["ccp"] = {}
    for run_i, K in enumerate(CCP_K_VALUES, start=2):
        print(f"\n  [{run_i}/4] Running CCP (K={K} bins)...")
        t0 = time.time()

        panel_K = coarsen_panel(panel_fine, K, N_FINE)
        P_hat_K = jnp.array(estimate_freq_transitions(panel_K, K))
        features_K = jnp.array(build_coarse_features(K))
        problem_K = DDCProblem(num_states=K, num_actions=2, discount_factor=BETA)
        utility_K = LinearUtility(features_K, parameter_names=["theta_v", "theta_fc"])

        ccp = CCPEstimator(num_policy_iterations=2, verbose=False)
        ccp_res = ccp.estimate(panel_K, utility_K, problem_K, P_hat_K)
        ccp_time = time.time() - t0

        ccp_params = np.array(ccp_res.parameters)

        # Policy accuracy: re-solve on fine grid with estimated params
        fine_reward_hat = utility_fine.compute(jnp.array(ccp_params, dtype=jnp.float32))
        vi_hat = value_iteration(operator, fine_reward_hat, tol=1e-10, max_iter=5000)
        ccp_acc = policy_accuracy(np.array(vi_hat.policy), true_policy)

        rmse = param_rmse(ccp_params, TRUE_PARAMS)
        results["ccp"][K] = {
            "theta_v": float(ccp_params[0]),
            "theta_fc": float(ccp_params[1]),
            "rmse": rmse,
            "acc": ccp_acc,
            "time": ccp_time,
        }
        print(
            f"    theta_v={ccp_params[0]:.4f}, theta_fc={ccp_params[1]:.4f}, "
            f"RMSE={rmse:.4f}, acc={ccp_acc:.3f}, time={ccp_time:.2f}s"
        )

    # ── TD-CCP (no transitions needed) ──
    print("\n  [4/4] Running TD-CCP (polynomial basis, no P)...")
    t0 = time.time()
    tdccp = TDCCPEstimator(
        config=TDCCPConfig(
            method="semigradient",
            basis_dim=8,
            ccp_method="logit",
            ccp_poly_degree=3,
            cross_fitting=False,
            robust_se=True,
            verbose=False,
        )
    )
    # TD-CCP only uses transitions for the final policy solve, not for theta estimation
    tr = tdccp.estimate(panel_fine, utility_fine, problem_fine, transitions_jnp)
    tdccp_time = time.time() - t0

    td_params = np.array(tr.parameters)
    td_acc = policy_accuracy(np.array(tr.policy), true_policy)
    td_rmse = param_rmse(td_params, TRUE_PARAMS)
    results["tdccp"] = {
        "theta_v": float(td_params[0]),
        "theta_fc": float(td_params[1]),
        "se_theta_v": float(tr.standard_errors[0]),
        "se_theta_fc": float(tr.standard_errors[1]),
        "rmse": td_rmse,
        "acc": td_acc,
        "time": tdccp_time,
    }
    print(
        f"    theta_v={td_params[0]:.4f} (SE {tr.standard_errors[0]:.4f}), "
        f"theta_fc={td_params[1]:.4f} (SE {tr.standard_errors[1]:.4f})"
    )
    print(f"    RMSE={td_rmse:.4f}, acc={td_acc:.3f}, time={tdccp_time:.2f}s")

    # ── Write LaTeX ──
    _write_latex(results, n_obs)
    print(f"\n  Wrote {OUT}")
    OUT.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print("  Done.")


def _write_latex(results: dict, n_obs: int) -> None:
    """Write tdccp_results.tex with \newcommand macros and auto-generated table."""

    def fmt(v: float, d: int = 4) -> str:
        return f"{v:.{d}f}"

    ccp5 = results["ccp"][5]
    ccp20 = results["ccp"][20]
    td = results["tdccp"]

    tex = [
        "% Auto-generated by tdccp_run.py — do not edit by hand",
        f"% Firm value model: {N_FINE} states, beta={BETA}, {n_obs:,} obs",
        "",
        f"\\newcommand{{\\tdNstates}}{{{N_FINE}}}",
        f"\\newcommand{{\\tdBeta}}{{{BETA}}}",
        f"\\newcommand{{\\tdNobs}}{{{n_obs:,}}}",
        f"\\newcommand{{\\tdTrueV}}{{{TRUE_THETA_V}}}",
        f"\\newcommand{{\\tdTrueFC}}{{{TRUE_THETA_FC}}}",
        "",
        f"\\newcommand{{\\bcAcc}}{{{fmt(results['bc']['acc'] * 100, 1)}\\%}}",
        "",
        f"\\newcommand{{\\ccpFiveV}}{{{fmt(ccp5['theta_v'])}}}",
        f"\\newcommand{{\\ccpFiveFC}}{{{fmt(ccp5['theta_fc'])}}}",
        f"\\newcommand{{\\ccpFiveRMSE}}{{{fmt(ccp5['rmse'])}}}",
        f"\\newcommand{{\\ccpFiveAcc}}{{{fmt(ccp5['acc'] * 100, 1)}\\%}}",
        f"\\newcommand{{\\ccpFiveTime}}{{{fmt(ccp5['time'], 1)}}}",
        "",
        f"\\newcommand{{\\ccpTwentyV}}{{{fmt(ccp20['theta_v'])}}}",
        f"\\newcommand{{\\ccpTwentyFC}}{{{fmt(ccp20['theta_fc'])}}}",
        f"\\newcommand{{\\ccpTwentyRMSE}}{{{fmt(ccp20['rmse'])}}}",
        f"\\newcommand{{\\ccpTwentyAcc}}{{{fmt(ccp20['acc'] * 100, 1)}\\%}}",
        f"\\newcommand{{\\ccpTwentyTime}}{{{fmt(ccp20['time'], 1)}}}",
        "",
        f"\\newcommand{{\\tdV}}{{{fmt(td['theta_v'])}}}",
        f"\\newcommand{{\\tdFC}}{{{fmt(td['theta_fc'])}}}",
        f"\\newcommand{{\\tdSEV}}{{{fmt(td['se_theta_v'])}}}",
        f"\\newcommand{{\\tdSEFC}}{{{fmt(td['se_theta_fc'])}}}",
        f"\\newcommand{{\\tdRMSE}}{{{fmt(td['rmse'])}}}",
        f"\\newcommand{{\\tdAcc}}{{{fmt(td['acc'] * 100, 1)}\\%}}",
        f"\\newcommand{{\\tdTime}}{{{fmt(td['time'], 1)}}}",
        "",
        "\\begin{table}[H]",
        "\\centering\\small",
        (
            f"\\caption{{Firm value model (\\tdNstates\\ states, $\\beta=\\tdBeta$, "
            f"\\tdNobs\\ obs). True: $\\theta_v={TRUE_THETA_V}$, "
            f"$\\theta_{{FC}}={TRUE_THETA_FC}$. "
            f"CCP must discretize the continuous state; TD-CCP does not.}}"
        ),
        "\\label{tab:tdccp_results}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r r r r r}",
        "\\toprule",
        "Method & $\\hat\\theta_v$ & $\\hat\\theta_{FC}$ & RMSE & Acc & Time (s) \\\\",
        "\\midrule",
        f"True values & {TRUE_THETA_V:.1f} & {TRUE_THETA_FC:.1f} & --- & --- & --- \\\\",
        "BC & --- & --- & --- & \\bcAcc & --- \\\\",
        "\\midrule",
        "CCP ($K=5$ bins) & \\ccpFiveV & \\ccpFiveFC & \\ccpFiveRMSE & \\ccpFiveAcc & \\ccpFiveTime \\\\",
        "CCP ($K=20$ bins) & \\ccpTwentyV & \\ccpTwentyFC & \\ccpTwentyRMSE & \\ccpTwentyAcc & \\ccpTwentyTime \\\\",
        "\\midrule",
        "TD-CCP & \\tdV\\ (\\tdSEV) & \\tdFC\\ (\\tdSEFC) & \\tdRMSE & \\tdAcc & \\tdTime \\\\",
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table}",
    ]

    OUT.write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
