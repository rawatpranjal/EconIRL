#!/usr/bin/env python3
"""NNES Primer — auto-generate results for the LaTeX document.

Demonstrates NNES's core advantage: scalability to large state spaces
where NFXP is prohibitively slow, while preserving valid standard errors
via the block-diagonal information matrix (Neyman orthogonality).

On a 2-component bus engine (400 states), NFXP must solve a 400x400
Bellman fixed point at every outer step. NNES trains a small V-network
that takes 2D mileage input and generalizes across the state space,
running an order of magnitude faster while matching NFXP's MLE.

This replicates the scalability argument from Nguyen (2025, Section 6).

Usage:
    python papers/econirl_package/primers/nnes/nnes_run.py
"""

import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "nnes_results.tex"

# ---------- DGP: Multi-component bus (large state space) ----------
K_COMPONENTS = 2       # number of independent bus components
M_BINS = 20            # mileage bins per component → 400 total states
DISCOUNT = 0.9999
SEED = 42
N_BUSES = 200
N_PERIODS = 100


def main():
    import jax.numpy as jnp
    from econirl.environments.multi_component_bus import MultiComponentBusEnvironment
    from econirl.estimation.nfxp import NFXPEstimator
    from econirl.estimation.nnes import NNESConfig, NNESEstimator
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from econirl.preferences.linear import LinearUtility
    from econirl.simulation.synthetic import simulate_panel
    from econirl.core.types import DDCProblem
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import policy_iteration as pi_solve

    env = MultiComponentBusEnvironment(
        K=K_COMPONENTS, M=M_BINS, discount_factor=DISCOUNT,
    )
    n_states = env.num_states
    n_obs = N_BUSES * N_PERIODS
    true_params = env.get_true_parameter_vector()
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    pnames = env.parameter_names

    print("NNES Primer — generating results")
    print(f"  Environment: {K_COMPONENTS}-component bus, {M_BINS} bins/component, "
          f"{n_states} total states, beta={DISCOUNT}")
    print(f"  Data: {N_BUSES} x {N_PERIODS} = {n_obs:,} obs")
    print(f"  True params: {dict(zip(pnames, np.asarray(true_params)))}")

    panel = simulate_panel(
        env, n_individuals=N_BUSES, n_periods=N_PERIODS, seed=SEED,
    )

    # True policy for accuracy comparison
    transitions_f64 = jnp.asarray(transitions, dtype=jnp.float64)
    operator = SoftBellmanOperator(problem, transitions_f64)
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = pi_solve(operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix")
    true_actions = np.argmax(np.array(true_result.policy), axis=1)

    results = {}

    # -- BC baseline --
    print("\n  Running BC baseline...")
    t0 = time.time()
    bc = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    bc_res = bc.estimate(panel, utility, problem, jnp.asarray(transitions))
    bc_time = time.time() - t0
    bc_acc = float(np.mean(np.argmax(np.array(bc_res.policy), axis=1) == true_actions))
    results["bc"] = {"ll": float(bc_res.log_likelihood), "acc": bc_acc, "time": bc_time}
    print(f"    LL={results['bc']['ll']:.2f}, acc={bc_acc:.2%}, time={bc_time:.2f}s")

    # -- NFXP oracle --
    print("\n  Running NFXP (oracle)...")
    nfxp = NFXPEstimator(
        inner_solver="hybrid", inner_tol=1e-10, inner_max_iter=100000,
        switch_tol=1e-3, outer_max_iter=200,
        compute_hessian=True, verbose=False,
    )
    t0 = time.time()
    nfxp_res = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    nfxp_p = np.asarray(nfxp_res.parameters)
    nfxp_se = np.asarray(nfxp_res.standard_errors)
    results["nfxp"] = {
        "params": [float(x) for x in nfxp_p],
        "se": [float(x) for x in nfxp_se],
        "ll": float(nfxp_res.log_likelihood),
        "time": nfxp_time,
    }
    print(f"    params={[f'{x:.4f}' for x in nfxp_p]}, "
          f"LL={nfxp_res.log_likelihood:.1f}, time={nfxp_time:.1f}s")

    # -- NNES (NPL Bellman) --
    # State encoder: decode flat state index into K normalized mileage values
    def state_encoder(s):
        """Map flat state indices to K-dimensional normalized mileage vectors."""
        s = jnp.asarray(s, dtype=jnp.int32)
        components = []
        remainder = s
        for _ in range(K_COMPONENTS):
            components.append((remainder % M_BINS).astype(jnp.float32) / max(M_BINS - 1, 1))
            remainder = remainder // M_BINS
        return jnp.stack(components, axis=-1)  # shape (..., K)

    problem_nnes = DDCProblem(
        num_states=n_states,
        num_actions=problem.num_actions,
        discount_factor=DISCOUNT,
        scale_parameter=1.0,
        state_dim=K_COMPONENTS,
        state_encoder=state_encoder,
    )

    print(f"\n  Running NNES (NPL, 20 outer iterations, {K_COMPONENTS}D state encoder)...")
    nnes_cfg = NNESConfig(
        hidden_dim=64, num_layers=2,
        v_lr=1e-3, v_epochs=500,
        n_outer_iterations=20,
        compute_se=True, se_method="asymptotic",
        verbose=False,
    )
    nnes_est = NNESEstimator(config=nnes_cfg)
    t0 = time.time()
    nnes_res = nnes_est.estimate(panel, utility, problem_nnes, transitions)
    nnes_time = time.time() - t0
    nnes_p = np.asarray(nnes_res.parameters)
    nnes_se = np.asarray(nnes_res.standard_errors)
    results["nnes"] = {
        "params": [float(x) for x in nnes_p],
        "se": [float(x) for x in nnes_se],
        "ll": float(nnes_res.log_likelihood),
        "time": nnes_time,
    }
    print(f"    params={[f'{x:.4f}' for x in nnes_p]}, "
          f"LL={nnes_res.log_likelihood:.1f}, time={nnes_time:.1f}s")

    ll_gap = abs(float(nnes_res.log_likelihood) - float(nfxp_res.log_likelihood))
    speedup = nfxp_time / nnes_time if nnes_time > 0 else float("inf")
    print(f"\n  LL gap: {ll_gap:.2f}, Speedup: {speedup:.1f}x")

    # -- Write LaTeX macros + table --
    true_vals = [float(x) for x in true_params]
    tex = []
    tex.append("% Auto-generated by nnes_run.py — do not edit by hand")
    tex.append(f"% {K_COMPONENTS}-component bus, {M_BINS} bins/comp, "
               f"{n_states} states, {n_obs} obs, seed={SEED}")
    tex.append("")
    tex.append(f"\\newcommand{{\\nnesKcomp}}{{{K_COMPONENTS}}}")
    tex.append(f"\\newcommand{{\\nnesMbins}}{{{M_BINS}}}")
    tex.append(f"\\newcommand{{\\nnesNstates}}{{{n_states}}}")
    tex.append(f"\\newcommand{{\\nnesBeta}}{{{DISCOUNT}}}")
    tex.append(f"\\newcommand{{\\nnesNobs}}{{{n_obs:,}}}")
    tex.append(f"\\newcommand{{\\nnesLLgap}}{{{ll_gap:.2f}}}")
    tex.append(f"\\newcommand{{\\nnesSpeedup}}{{{speedup:.0f}}}")
    tex.append(f"\\newcommand{{\\nnesNfxpTime}}{{{nfxp_time:.1f}}}")
    tex.append(f"\\newcommand{{\\nnesTime}}{{{nnes_time:.1f}}}")
    tex.append(f"\\newcommand{{\\nnesBCacc}}{{{results['bc']['acc']*100:.1f}\\%}}")
    tex.append(f"\\newcommand{{\\nnesBCll}}{{{results['bc']['ll']:.1f}}}")
    tex.append("")

    # Results table
    tex.append("\\begin{table}[H]")
    tex.append("\\centering\\small")
    tex.append(f"\\caption{{\\nnesKcomp-component bus engine "
               f"(\\nnesNstates\\ states, $\\beta=\\nnesBeta$, \\nnesNobs\\ obs). "
               f"NNES uses a 2-layer ReLU V-network (64 hidden), 20 outer iterations.}}")
    tex.append("\\label{tab:nnes_results}")
    tex.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r r r}")
    tex.append("\\toprule")
    tex.append("& True & NFXP & NNES \\\\")
    tex.append("\\midrule")
    for i, name in enumerate(pnames):
        true_val = true_vals[i]
        nfxp_val = float(nfxp_p[i])
        nnes_val = float(nnes_p[i])
        nfxp_se_val = float(nfxp_se[i])
        nnes_se_val = float(nnes_se[i])
        tex.append(f"${name}$ & {true_val:.4f} & {nfxp_val:.4f} & {nnes_val:.4f} \\\\")
        se_nfxp = f"{nfxp_se_val:.4f}" if not np.isnan(nfxp_se_val) else "---"
        se_nnes = f"{nnes_se_val:.4f}" if not np.isnan(nnes_se_val) else "---"
        tex.append(f"\\quad SE & --- & {se_nfxp} & {se_nnes} \\\\")
    tex.append(f"Log-lik & --- & ${float(nfxp_res.log_likelihood):.1f}$ "
               f"& ${float(nnes_res.log_likelihood):.1f}$ \\\\")
    tex.append(f"Time (s) & --- & {nfxp_time:.1f} & {nnes_time:.1f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular*}")
    tex.append("\\end{table}")

    OUT.write_text("\n".join(tex) + "\n")
    print(f"\n  Wrote {OUT}")
    OUT.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print(f"  Wrote {OUT.with_suffix('.json')}")


if __name__ == "__main__":
    main()
