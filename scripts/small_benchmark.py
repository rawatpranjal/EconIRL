"""Small MDP benchmark: all 17 estimators on a 20-bin Rust bus.

MDP: RustBus(20 bins, OC=0.01, RC=2.0, discount=0.99)
Data: 200 agents x 100 periods = 20,000 observations (saved to NPZ)
Goal: verify all estimators run, recover reward, produce sensible policies.
"""

import json
import os
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation import (
    NFXPEstimator,
    CCPEstimator,
    MCEIRLEstimator, MCEIRLConfig,
    TDCCPEstimator, TDCCPConfig,
    GLADIUSEstimator, GLADIUSConfig,
    AIRLEstimator, AIRLConfig,
)
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.estimation.nnes import NNESEstimator
from econirl.estimation.sees import SEESEstimator
from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
from econirl.estimation.f_irl import FIRLEstimator
from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
from econirl.contrib.maxent_irl import MaxEntIRLEstimator
from econirl.contrib.deep_maxent_irl import DeepMaxEntIRLEstimator
from econirl.contrib.bayesian_irl import BayesianIRLEstimator
from econirl.contrib.gail import GAILEstimator, GAILConfig
from econirl.estimators.mceirl_neural import MCEIRLNeural
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.reward import LinearReward
from econirl.simulation.synthetic import simulate_panel
from econirl.core.types import Panel


# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark")
DATA_PATH = os.path.join(DATA_DIR, "small_rust_bus_20.npz")
RESULTS_PATH = os.path.join(DATA_DIR, "small_benchmark_results.json")


# ── Helpers ──────────────────────────────────────────────────────────────────

def rmse(est, true):
    return float(jnp.sqrt(jnp.mean((jnp.asarray(est) - jnp.asarray(true)) ** 2)))


def cosine_sim(est, true):
    e, t = jnp.asarray(est), jnp.asarray(true)
    return float(jnp.dot(e, t) / (jnp.linalg.norm(e) * jnp.linalg.norm(t)))


# ── Environment and data ─────────────────────────────────────────────────────

env = RustBusEnvironment(
    num_mileage_bins=20,
    operating_cost=0.01,
    replacement_cost=2.0,
    discount_factor=0.99,
)
true_params = env.get_true_parameter_vector()
n_states = env.problem_spec.num_states
n_actions = env.problem_spec.num_actions

N_IND = 200
N_PER = 100
N_OBS = N_IND * N_PER

# Generate or load data
if os.path.exists(DATA_PATH):
    print(f"Loading saved data from {DATA_PATH}")
    panel = Panel.load_npz(DATA_PATH)
else:
    print(f"Simulating data: {N_IND} agents x {N_PER} periods = {N_OBS:,} obs")
    panel = simulate_panel(env, n_individuals=N_IND, n_periods=N_PER, seed=42)
    os.makedirs(DATA_DIR, exist_ok=True)
    panel.save_npz(DATA_PATH)
    print(f"Saved to {DATA_PATH}")

# Build utility/reward specs
utility = LinearUtility.from_environment(env)
reward = ActionDependentReward.from_rust_environment(env)
df = panel.to_dataframe()

print(f"\nMDP: Rust bus, {n_states} bins, OC={env._operating_cost}, RC={env._replacement_cost}, gamma={env.problem_spec.discount_factor}")
print(f"Data: {N_IND} agents x {N_PER} periods = {N_OBS:,} observations")
print(f"True params: {true_params}\n")


# ── Estimator runners ────────────────────────────────────────────────────────

def run_nfxp():
    est = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10, inner_max_iter=100000, compute_hessian=False, verbose=True)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_ccp():
    est = CCPEstimator(num_policy_iterations=1, compute_hessian=False, verbose=True)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_mpec():
    config = MPECConfig(solver="slsqp", max_iter=200, constraint_tol=1e-6)
    est = MPECEstimator(config=config, compute_hessian=False, verbose=True)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_nnes():
    est = NNESEstimator(hidden_dim=32, v_epochs=200, n_outer_iterations=3, compute_se=False, verbose=True)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_sees():
    est = SEESEstimator(basis_type="fourier", basis_dim=8, penalty_weight=10.0, compute_se=False, verbose=True)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_tdccp():
    config = TDCCPConfig(hidden_dim=64, avi_iterations=20, epochs_per_avi=20, compute_se=False, verbose=True)
    est = TDCCPEstimator(config=config)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_mce_irl():
    config = MCEIRLConfig(learning_rate=0.05, outer_max_iter=100, inner_max_iter=2000, compute_se=False, verbose=True)
    est = MCEIRLEstimator(config=config)
    return est.estimate(panel, reward, env.problem_spec, env.transition_matrices)

def run_maxent_irl():
    state_features = jnp.stack([
        jnp.arange(n_states, dtype=jnp.float64) / n_states,
        jnp.ones(n_states, dtype=jnp.float64),
    ], axis=1)
    lr = LinearReward(state_features=state_features, parameter_names=["operating_cost", "replacement_cost"], n_actions=2)
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=100, compute_hessian=False, verbose=True)
    return est.estimate(panel, lr, env.problem_spec, env.transition_matrices)

def run_iq_learn():
    config = IQLearnConfig(q_type="linear", divergence="chi2", optimizer="L-BFGS-B", max_iter=1000, verbose=True)
    est = IQLearnEstimator(config=config)
    return est.estimate(panel, reward, env.problem_spec, env.transition_matrices)

def run_gladius():
    config = GLADIUSConfig(max_epochs=100, q_hidden_dim=64, v_hidden_dim=64, compute_se=False, verbose=True)
    est = GLADIUSEstimator(config=config)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_gail():
    config = GAILConfig(discriminator_type="tabular", max_rounds=10, compute_se=False, verbose=True)
    est = GAILEstimator(config=config)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_airl():
    config = AIRLConfig(reward_type="tabular", max_rounds=10, compute_se=False, verbose=True)
    est = AIRLEstimator(config=config)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_firl():
    est = FIRLEstimator(f_divergence="kl", lr=0.05, max_iter=300, verbose=True)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_bc():
    est = BehavioralCloningEstimator(smoothing=1.0, verbose=True)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_deep_maxent():
    est = DeepMaxEntIRLEstimator(hidden_dims=[32, 32], max_epochs=100, lr=1e-3, verbose=True, seed=42)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_bayesian_irl():
    est = BayesianIRLEstimator(n_samples=500, burnin=150, proposal_sigma=0.05, verbose=True, seed=42)
    return est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

def run_mceirl_neural():
    model = MCEIRLNeural(
        n_states=n_states,
        n_actions=n_actions,
        discount=0.99,
        reward_type="state_action",
        reward_hidden_dim=32,
        reward_num_layers=2,
        max_epochs=100,
        lr=1e-3,
        seed=42,
        verbose=True,
    )
    model.fit(df, state="state", action="action", id="id", transitions=jnp.asarray(env.transition_matrices))
    class R:
        pass
    r = R()
    r.parameters = model.params_ if model.params_ is not None else None
    r.policy = model.policy_
    r.converged = True
    return r


# ── Estimator list ───────────────────────────────────────────────────────────

estimators = [
    ("NFXP",          run_nfxp,          "structural"),
    ("CCP",           run_ccp,           "structural"),
    ("MPEC",          run_mpec,          "structural"),
    ("NNES",          run_nnes,          "structural"),
    ("SEES",          run_sees,          "structural"),
    ("TD-CCP",        run_tdccp,         "structural"),
    ("MCE-IRL",       run_mce_irl,       "structural"),
    ("MaxEnt-IRL",    run_maxent_irl,    "irl"),
    ("IQ-Learn",      run_iq_learn,      "irl"),
    ("GLADIUS",       run_gladius,       "policy_only"),
    ("GAIL",          run_gail,          "policy_only"),
    ("AIRL",          run_airl,          "policy_only"),
    ("f-IRL",         run_firl,          "policy_only"),
    ("BC",            run_bc,            "policy_only"),
    ("DeepMaxEnt",    run_deep_maxent,   "policy_only"),
    ("BayesianIRL",   run_bayesian_irl,  "irl"),
    ("MCE-IRL-NN",    run_mceirl_neural, "policy_only"),
]


# ── Run all ──────────────────────────────────────────────────────────────────

results = []

for idx, (name, runner, ptype) in enumerate(estimators):
    print(f"\n{'='*70}")
    print(f"[{idx+1}/{len(estimators)}] {name} (type={ptype})")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        r = runner()
        dt = time.time() - t0

        has_params = (r.parameters is not None and hasattr(r.parameters, '__len__') and len(r.parameters) >= 2)

        if has_params:
            oc = float(r.parameters[0])
            rc = float(r.parameters[1])
            p_est = jnp.asarray(r.parameters[:2])
            p_true = jnp.asarray(true_params[:2])
            r_val = rmse(p_est, p_true)
            cos = cosine_sim(p_est, p_true)
        else:
            oc, rc, r_val, cos = None, None, None, None

        if r.policy is not None:
            pol = jnp.asarray(r.policy)
            n_lo = min(5, pol.shape[0])
            n_hi = min(5, pol.shape[0])
            low = float(pol[:n_lo, 1].mean())
            high = float(pol[-n_hi:, 1].mean())
            direction = "YES" if high > low else "NO"
        else:
            low, high, direction = None, None, "N/A"

        converged = getattr(r, 'converged', None)
        results.append({
            "name": name, "type": ptype, "oc": oc, "rc": rc,
            "rmse": r_val, "cosine": cos, "direction": direction,
            "p_replace_lo": low, "p_replace_hi": high,
            "time": dt, "converged": converged, "error": None,
        })
        status = f"RMSE={r_val:.4f}" if r_val is not None else f"Direction={direction}"
        print(f"  -> Done in {dt:.1f}s. {status}")

    except Exception as e:
        dt = time.time() - t0
        results.append({
            "name": name, "type": ptype, "oc": None, "rc": None,
            "rmse": None, "cosine": None, "direction": "CRASH",
            "p_replace_lo": None, "p_replace_hi": None,
            "time": dt, "converged": False, "error": str(e),
        })
        print(f"  -> FAILED in {dt:.1f}s: {e}")
        traceback.print_exc()


# ── Print results table ──────────────────────────────────────────────────────

print("\n\n")
print("=" * 130)
print(f"SMALL BENCHMARK — Rust Bus, {n_states} bins, OC={float(true_params[0])}, RC={float(true_params[1])}, gamma=0.99")
print(f"Data: {N_IND} agents x {N_PER} periods = {N_OBS:,} observations")
print("=" * 130)
print(f"{'Estimator':<14} {'Type':<11} {'OC_hat':>10} {'RC_hat':>10} {'RMSE':>8} {'Cosine':>8} {'Direct':>6} {'P(R|lo)':>8} {'P(R|hi)':>8} {'Time':>7} {'Conv':>5}")
print("-" * 130)

for r in results:
    oc_s = f"{r['oc']:.6f}" if r['oc'] is not None else "---"
    rc_s = f"{r['rc']:.4f}" if r['rc'] is not None else "---"
    r_s = f"{r['rmse']:.4f}" if r['rmse'] is not None else "---"
    c_s = f"{r['cosine']:.4f}" if r['cosine'] is not None else "---"
    lo_s = f"{r['p_replace_lo']:.4f}" if r['p_replace_lo'] is not None else "---"
    hi_s = f"{r['p_replace_hi']:.4f}" if r['p_replace_hi'] is not None else "---"
    dt_s = f"{r['time']:.0f}s"
    conv_s = "Y" if r['converged'] else ("N" if r['converged'] is not None else "---")
    if r['error']:
        print(f"{r['name']:<14} {'CRASHED':<11} {'':>10} {'':>10} {'':>8} {'':>8} {'':>6} {'':>8} {'':>8} {dt_s:>7} {'':>5}  {r['error'][:50]}")
    else:
        print(f"{r['name']:<14} {r['type']:<11} {oc_s:>10} {rc_s:>10} {r_s:>8} {c_s:>8} {r['direction']:>6} {lo_s:>8} {hi_s:>8} {dt_s:>7} {conv_s:>5}")

print("-" * 130)
print()
print("Legend:")
print("  OC_hat / RC_hat = recovered operating cost / replacement cost")
print("  RMSE = root mean squared error vs true parameters")
print("  Cosine = cosine similarity between estimated and true param vectors")
print("  Direct = P(replace) increases with mileage?")
print("  P(R|lo) / P(R|hi) = mean P(replace) at low/high mileage bins")
print("  Conv = converged (Y/N)")
print("  structural = recovers params directly | irl = up to scale | policy_only = no params")


# ── Save results to JSON ─────────────────────────────────────────────────────

# Convert for JSON serialization
for r in results:
    for k, v in r.items():
        if isinstance(v, (np.floating, jnp.floating)):
            r[k] = float(v)
        elif isinstance(v, (np.integer, jnp.integer)):
            r[k] = int(v)

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "mdp": {"bins": n_states, "oc": float(true_params[0]), "rc": float(true_params[1]), "gamma": 0.99},
        "data": {"n_individuals": N_IND, "n_periods": N_PER, "n_obs": N_OBS},
        "results": results,
    }, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
