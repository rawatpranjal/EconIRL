"""Post-estimation diagnostics on the Rust bus engine dataset.

Demonstrates all 10 inference tools: hypothesis tests, fit metrics,
reward comparison, and model comparison tables.

Run: python examples/post_estimation_diagnostics.py
"""

import jax.numpy as jnp
import numpy as np

from econirl import NFXP, CCP
from econirl.datasets import load_rust_bus
from econirl.inference import (
    likelihood_ratio_test,
    vuong_test,
    brier_score,
    kl_divergence,
    efron_pseudo_r_squared,
    ccp_consistency_test,
    epic_distance,
    detect_reward_shaping,
    etable,
)

# ── Load data and fit two models ─────────────────────────────────────────

df = load_rust_bus()
print(f"Loaded Rust bus data: {len(df):,} observations, {df['bus_id'].nunique()} buses\n")

print("Fitting NFXP...")
nfxp = NFXP(discount=0.9999).fit(
    df, state="mileage_bin", action="replaced", id="bus_id"
)

print("Fitting CCP...")
ccp = CCP(discount=0.9999, num_policy_iterations=3).fit(
    df, state="mileage_bin", action="replaced", id="bus_id"
)

# ── 1. Side-by-side comparison table ─────────────────────────────────────

print("\n" + "=" * 60)
print("1. MODEL COMPARISON TABLE (etable)")
print("=" * 60)
print(etable(nfxp._result, ccp._result, model_names=["NFXP", "CCP"]))

# ── 2. Likelihood ratio test ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. LIKELIHOOD RATIO TEST")
print("=" * 60)

# Test whether the CCP model (fewer effective iterations) fits as well as NFXP.
# Both have the same number of structural parameters, so we compare the log-likelihoods.
ll_nfxp = nfxp._result.log_likelihood
ll_ccp = ccp._result.log_likelihood
print(f"Log-likelihood NFXP:  {ll_nfxp:,.2f}")
print(f"Log-likelihood CCP:   {ll_ccp:,.2f}")
print(f"Difference:           {abs(ll_nfxp - ll_ccp):.4f}")
print("(Both estimators have the same structural parameters, so the LR test")
print(" applies when one model is a restricted version of the other.)")

# ── 3. Vuong test ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. VUONG TEST (non-nested comparison)")
print("=" * 60)

obs_states = jnp.array(df["mileage_bin"].values, dtype=jnp.int32)
obs_actions = jnp.array(df["replaced"].values, dtype=jnp.int32)

vt = vuong_test(
    jnp.array(nfxp.policy_),
    jnp.array(ccp.policy_),
    obs_states,
    obs_actions,
    num_params_1=nfxp._result.num_parameters,
    num_params_2=ccp._result.num_parameters,
)
print(f"Vuong z-statistic:    {vt['statistic']:.4f}")
print(f"p-value:              {vt['p_value']:.4f}")
print(f"Direction:            {vt['direction']}")
if "corrected_statistic" in vt:
    print(f"Corrected z-stat:     {vt['corrected_statistic']:.4f}")
    print(f"Corrected p-value:    {vt['corrected_p_value']:.4f}")

# ── 4. Brier score ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. BRIER SCORE")
print("=" * 60)

bs_nfxp = brier_score(jnp.array(nfxp.policy_), obs_states, obs_actions)
bs_ccp = brier_score(jnp.array(ccp.policy_), obs_states, obs_actions)
print(f"Brier score NFXP:     {bs_nfxp['brier_score']:.6f}")
print(f"Brier score CCP:      {bs_ccp['brier_score']:.6f}")
print(f"  Per-action NFXP:    keep={bs_nfxp['brier_score_per_action'][0]:.6f}, "
      f"replace={bs_nfxp['brier_score_per_action'][1]:.6f}")

# ── 5. KL divergence ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. KL DIVERGENCE")
print("=" * 60)

num_states = nfxp.policy_.shape[0]
num_actions = nfxp.policy_.shape[1]

# Compute empirical CCPs from the data
state_counts = np.zeros(num_states)
choice_counts = np.zeros((num_states, num_actions))
for s, a in zip(df["mileage_bin"].values, df["replaced"].values):
    state_counts[s] += 1
    choice_counts[s, a] += 1

state_freq = jnp.array(state_counts / state_counts.sum())
data_ccps = jnp.array(
    choice_counts / np.maximum(state_counts[:, None], 1)
)

kl_nfxp = kl_divergence(data_ccps, jnp.array(nfxp.policy_), state_freq)
kl_ccp = kl_divergence(data_ccps, jnp.array(ccp.policy_), state_freq)
print(f"KL divergence NFXP:   {kl_nfxp['kl_divergence']:.6f}")
print(f"KL divergence CCP:    {kl_ccp['kl_divergence']:.6f}")

# ── 6. Efron pseudo R-squared ───────────────────────────────────────────

print("\n" + "=" * 60)
print("6. EFRON PSEUDO R-SQUARED")
print("=" * 60)

er2_nfxp = efron_pseudo_r_squared(jnp.array(nfxp.policy_), obs_states, obs_actions)
er2_ccp = efron_pseudo_r_squared(jnp.array(ccp.policy_), obs_states, obs_actions)
print(f"Efron R² NFXP:        {er2_nfxp['efron_r_squared']:.4f}")
print(f"Efron R² CCP:         {er2_ccp['efron_r_squared']:.4f}")

# ── 7. CCP consistency test ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. CCP CONSISTENCY TEST")
print("=" * 60)

ccp_test_nfxp = ccp_consistency_test(
    data_ccps, jnp.array(nfxp.policy_),
    jnp.array(state_counts),
    num_estimated_params=nfxp._result.num_parameters,
)
print(f"Chi-squared stat:     {ccp_test_nfxp['statistic']:.2f}")
print(f"Degrees of freedom:   {ccp_test_nfxp['df']}")
print(f"p-value:              {ccp_test_nfxp['p_value']:.4f}")

# ── 8. EPIC distance ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. EPIC DISTANCE (reward comparison)")
print("=" * 60)

# Build reward vectors from estimated parameters for each model
# R(s, a=0) = -theta_c * s (operating cost), R(s, a=1) = -RC (replacement)
theta_c_nfxp = nfxp.params_["theta_c"]
rc_nfxp = nfxp.params_["RC"]
theta_c_ccp = ccp.params_["theta_c"]
rc_ccp = ccp.params_["RC"]

states_arr = np.arange(num_states)
reward_nfxp = np.zeros((num_states, num_actions))
reward_nfxp[:, 0] = -theta_c_nfxp * states_arr  # operating cost
reward_nfxp[:, 1] = -rc_nfxp  # replacement cost

reward_ccp = np.zeros((num_states, num_actions))
reward_ccp[:, 0] = -theta_c_ccp * states_arr
reward_ccp[:, 1] = -rc_ccp

epic = epic_distance(jnp.array(reward_nfxp), jnp.array(reward_ccp), 0.9999)
print(f"EPIC distance:        {epic['epic_distance']:.6f}")
print(f"Pearson correlation:  {epic['pearson_correlation']:.6f}")

# ── 9. Reward shaping detection ─────────────────────────────────────────

print("\n" + "=" * 60)
print("9. REWARD SHAPING DETECTION")
print("=" * 60)

# Synthetic demonstration: construct a shaped reward and verify detection.
# Start with the NFXP reward R(s,a,s'), add gamma*Phi(s') - Phi(s),
# and confirm the detector recovers Phi.
S = 5
gamma = 0.99
r_base = jnp.zeros((S, 2, S))
phi_true = jnp.array([0.0, 1.5, -0.8, 2.3, -1.1])
shaped = r_base + gamma * phi_true[None, None, :] - phi_true[:, None, None]

shaping = detect_reward_shaping(r_base, shaped, gamma)
print(f"Is shaping only:      {shaping['is_shaping']}")
print(f"Relative residual:    {shaping['relative_residual']:.6f}")
print(f"Max abs residual:     {shaping['max_absolute_residual']:.6f}")
print(f"Recovered potential:  {[f'{x:.2f}' for x in shaping['potential']]}")
print(f"True potential:       {[f'{float(x):.2f}' for x in phi_true]}")

# ── 10. LaTeX table ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("10. LATEX COMPARISON TABLE")
print("=" * 60)
print(etable(nfxp._result, ccp._result, model_names=["NFXP", "CCP"], output="latex"))

print("\n" + "=" * 60)
print("Done. All 10 post-estimation tools demonstrated.")
print("=" * 60)
