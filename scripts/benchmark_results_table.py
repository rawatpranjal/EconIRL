"""Run ALL estimators on Rust bus and print a results table.

True parameters: operating_cost=0.001, replacement_cost=3.0
Data: 2000 individuals x 200 periods = 400,000 observations (generous)
"""

import time
import traceback
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

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


def rmse(est, true):
    return float(jnp.sqrt(jnp.mean((jnp.asarray(est) - jnp.asarray(true)) ** 2)))


def cosine_sim(est, true):
    e, t = jnp.asarray(est), jnp.asarray(true)
    return float(jnp.dot(e, t) / (jnp.linalg.norm(e) * jnp.linalg.norm(t)))


# ── Data generation ──────────────────────────────────────────────────────────

print("Simulating data...", flush=True)
env_hi = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999)
env_lo = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0, discount_factor=0.99)
true_hi = env_hi.get_true_parameter_vector()
true_lo = env_lo.get_true_parameter_vector()

N_IND = 2000
N_PER = 200
N_OBS = N_IND * N_PER

panel_hi = simulate_panel(env_hi, n_individuals=N_IND, n_periods=N_PER, seed=42)
panel_lo = simulate_panel(env_lo, n_individuals=N_IND, n_periods=N_PER, seed=42)

utility_hi = LinearUtility.from_environment(env_hi)
utility_lo = LinearUtility.from_environment(env_lo)
reward_hi = ActionDependentReward.from_rust_environment(env_hi)
reward_lo = ActionDependentReward.from_rust_environment(env_lo)

# DataFrame for sklearn-API estimators
df_lo = panel_lo.to_dataframe()

print(f"Data ready: {N_IND} individuals x {N_PER} periods = {N_OBS:,} obs\n")


# ── Estimator runners (all verbose=True for tqdm) ───────────────────────────

def run_nfxp():
    est = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10, inner_max_iter=100000, compute_hessian=False, verbose=True)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_ccp():
    est = CCPEstimator(num_policy_iterations=1, compute_hessian=False, verbose=True)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_mpec():
    config = MPECConfig(solver="slsqp", max_iter=500, constraint_tol=1e-8)
    est = MPECEstimator(config=config, compute_hessian=False, verbose=True)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_nnes():
    est = NNESEstimator(hidden_dim=64, v_epochs=1000, n_outer_iterations=5, compute_se=False, verbose=True)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_sees():
    est = SEESEstimator(basis_type="fourier", basis_dim=12, penalty_weight=10.0, compute_se=False, verbose=True)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_tdccp():
    config = TDCCPConfig(hidden_dim=128, avi_iterations=30, epochs_per_avi=30, compute_se=False, verbose=True)
    est = TDCCPEstimator(config=config)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_mce_irl():
    config = MCEIRLConfig(learning_rate=0.05, outer_max_iter=1000, inner_max_iter=5000, compute_se=False, verbose=True)
    est = MCEIRLEstimator(config=config)
    return est.estimate(panel_lo, reward_lo, env_lo.problem_spec, env_lo.transition_matrices)

def run_maxent_irl():
    n_states = env_lo.problem_spec.num_states
    state_features = jnp.stack([
        jnp.arange(n_states, dtype=jnp.float32) / n_states,
        jnp.ones(n_states, dtype=jnp.float32),
    ], axis=1)
    reward = LinearReward(state_features=state_features, parameter_names=["operating_cost", "replacement_cost"], n_actions=2)
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=10000, outer_max_iter=1000, compute_hessian=False, verbose=True)
    return est.estimate(panel_lo, reward, env_lo.problem_spec, env_lo.transition_matrices)

def run_iq_learn():
    config = IQLearnConfig(q_type="linear", divergence="chi2", optimizer="L-BFGS-B", max_iter=2000, verbose=True)
    est = IQLearnEstimator(config=config)
    return est.estimate(panel_lo, reward_lo, env_lo.problem_spec, env_lo.transition_matrices)

def run_gladius():
    config = GLADIUSConfig(max_epochs=500, q_hidden_dim=128, v_hidden_dim=128, compute_se=False, verbose=True)
    est = GLADIUSEstimator(config=config)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_gail():
    config = GAILConfig(discriminator_type="tabular", max_rounds=100, compute_se=False, verbose=True)
    est = GAILEstimator(config=config)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_airl():
    config = AIRLConfig(reward_type="tabular", max_rounds=100, compute_se=False, verbose=True)
    est = AIRLEstimator(config=config)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_firl():
    est = FIRLEstimator(f_divergence="kl", lr=0.05, max_iter=500, verbose=True)
    return est.estimate(panel_lo, utility_lo, env_lo.problem_spec, env_lo.transition_matrices)

def run_bc():
    est = BehavioralCloningEstimator(smoothing=1.0, verbose=True)
    return est.estimate(panel_hi, utility_hi, env_hi.problem_spec, env_hi.transition_matrices)

def run_deep_maxent():
    est = DeepMaxEntIRLEstimator(hidden_dims=[64, 64], max_epochs=300, lr=1e-3, verbose=True, seed=42)
    return est.estimate(panel_lo, utility_lo, env_lo.problem_spec, env_lo.transition_matrices)

def run_bayesian_irl():
    est = BayesianIRLEstimator(n_samples=2000, burnin=500, proposal_sigma=0.05, verbose=True, seed=42)
    return est.estimate(panel_lo, utility_lo, env_lo.problem_spec, env_lo.transition_matrices)

def run_mceirl_neural():
    model = MCEIRLNeural(
        n_states=env_lo.problem_spec.num_states,
        n_actions=env_lo.problem_spec.num_actions,
        discount=0.99,
        reward_type="state_action",
        reward_hidden_dim=64,
        reward_num_layers=2,
        max_epochs=200,
        lr=1e-3,
        seed=42,
        verbose=True,
    )
    model.fit(df_lo, state="state", action="action", id="id", transitions=jnp.asarray(env_lo.transition_matrices))
    class R:
        pass
    r = R()
    r.parameters = model.params_ if model.params_ is not None else None
    r.policy = model.policy_
    r.converged = True
    return r


# ── Run all estimators ───────────────────────────────────────────────────────

estimators = [
    ("NFXP",          run_nfxp,          0.9999, true_hi, "structural"),
    ("CCP",           run_ccp,           0.9999, true_hi, "structural"),
    ("MPEC",          run_mpec,          0.9999, true_hi, "structural"),
    ("NNES",          run_nnes,          0.9999, true_hi, "structural"),
    ("SEES",          run_sees,          0.9999, true_hi, "structural"),
    ("TD-CCP",        run_tdccp,         0.9999, true_hi, "structural"),
    ("MCE-IRL",       run_mce_irl,       0.99,   true_lo, "structural"),
    ("MaxEnt-IRL",    run_maxent_irl,    0.99,   true_lo, "irl"),
    ("IQ-Learn",      run_iq_learn,      0.99,   true_lo, "irl"),
    ("GLADIUS",       run_gladius,       0.9999, true_hi, "structural"),
    ("GAIL",          run_gail,          0.9999, true_hi, "policy_only"),
    ("AIRL",          run_airl,          0.9999, true_hi, "policy_only"),
    ("f-IRL",         run_firl,          0.99,   true_lo, "policy_only"),
    ("BC",            run_bc,            0.9999, true_hi, "policy_only"),
    ("DeepMaxEnt",    run_deep_maxent,   0.99,   true_lo, "policy_only"),
    ("BayesianIRL",   run_bayesian_irl,  0.99,   true_lo, "irl"),
    ("MCE-IRL-NN",    run_mceirl_neural, 0.99,   true_lo, "policy_only"),
]

results = []

for idx, (name, runner, gamma, true_p, ptype) in enumerate(estimators):
    print(f"\n{'='*70}")
    print(f"[{idx+1}/{len(estimators)}] {name} (gamma={gamma}, type={ptype})")
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
            p_true = jnp.asarray(true_p[:2])
            r_val = rmse(p_est, p_true)
            cos = cosine_sim(p_est, p_true)
        else:
            oc, rc, r_val, cos = None, None, None, None

        if r.policy is not None:
            low = float(jnp.asarray(r.policy)[:10, 1].mean())
            high = float(jnp.asarray(r.policy)[-10:, 1].mean())
            direction = "YES" if high > low else "NO"
        else:
            low, high, direction = None, None, "N/A"

        results.append((name, gamma, ptype, oc, rc, r_val, cos, direction, low, high, dt, None))
        print(f"  -> Done in {dt:.0f}s. RMSE={r_val:.4f}" if r_val else f"  -> Done in {dt:.0f}s. Direction={direction}")

    except Exception as e:
        dt = time.time() - t0
        results.append((name, gamma, ptype, None, None, None, None, "CRASH", None, None, dt, str(e)))
        print(f"  -> FAILED in {dt:.0f}s: {e}")
        traceback.print_exc()

# ── Print results table ──────────────────────────────────────────────────────

print("\n\n")
print("=" * 130)
print(f"PARAMETER RECOVERY BENCHMARK — Rust Bus Engine")
print(f"True: operating_cost = 0.001, replacement_cost = 3.000")
print(f"Data: {N_IND} individuals x {N_PER} periods = {N_OBS:,} observations")
print("=" * 130)
print(f"{'Estimator':<14} {'Type':<11} {'gamma':>6} {'OC_hat':>10} {'RC_hat':>10} {'RMSE':>8} {'Cosine':>8} {'Direct':>6} {'P(R|lo)':>8} {'P(R|hi)':>8} {'Time':>7}")
print("-" * 130)

for name, gamma, ptype, oc, rc, r_val, cos, direction, low, high, dt, err in results:
    oc_s = f"{oc:.6f}" if oc is not None else "---"
    rc_s = f"{rc:.4f}" if rc is not None else "---"
    r_s = f"{r_val:.4f}" if r_val is not None else "---"
    c_s = f"{cos:.4f}" if cos is not None else "---"
    lo_s = f"{low:.4f}" if low is not None else "---"
    hi_s = f"{high:.4f}" if high is not None else "---"
    dt_s = f"{dt:.0f}s"
    if err:
        print(f"{name:<14} {'CRASHED':<11} {gamma:>6.4f} {'':>10} {'':>10} {'':>8} {'':>8} {'':>6} {'':>8} {'':>8} {dt_s:>7}  {err[:50]}")
    else:
        print(f"{name:<14} {ptype:<11} {gamma:>6.4f} {oc_s:>10} {rc_s:>10} {r_s:>8} {c_s:>8} {direction:>6} {lo_s:>8} {hi_s:>8} {dt_s:>7}")

print("-" * 130)
print()
print("Legend:")
print("  OC_hat / RC_hat = recovered operating cost / replacement cost")
print("  RMSE = root mean squared error vs true parameters")
print("  Cosine = cosine similarity between estimated and true parameter vectors")
print("  Direct = directional check: does P(replace) increase with mileage?")
print("  P(R|lo) = mean P(replace) at mileage bins 0-9")
print("  P(R|hi) = mean P(replace) at mileage bins 80-89")
print("  structural = recovers OC/RC directly")
print("  irl = recovers up to scale/constant (cosine more meaningful than RMSE)")
print("  policy_only = no interpretable parameters returned")
print("  --- = not applicable or not returned by estimator")
