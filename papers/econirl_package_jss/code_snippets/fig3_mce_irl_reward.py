"""Figure 3: MCE-IRL recovered reward overlaid on the NFXP structural utility.
The two curves coincide up to an additive constant on the Rust bus panel.

Reproduces: figures/fig3_mce_irl_reward.pdf.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig

env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
panel = load_rust_bus(as_panel=True)
linear = LinearUtility.from_environment(env)
adr = ActionDependentReward(
    feature_matrix=linear.feature_matrix,
    parameter_names=linear.parameter_names,
)
problem = env.problem_spec
transitions = env.transition_matrices

nfxp = NFXPEstimator(optimizer="BHHH", inner_solver="policy", compute_hessian=True)
nfxp_res = nfxp.estimate(panel=panel, utility=linear, problem=problem, transitions=transitions)

mce = MCEIRLEstimator(config=MCEIRLConfig(optimizer="L-BFGS-B", inner_solver="policy", compute_se=False))
mce_res = mce.estimate(panel=panel, utility=adr, problem=problem, transitions=transitions)

theta_nfxp = jnp.asarray(nfxp_res.parameters, dtype=jnp.float64)
theta_mce = jnp.asarray(mce_res.parameters, dtype=jnp.float64)

r_nfxp = np.asarray(linear.compute(theta_nfxp))[:, 0]
r_mce = np.asarray(adr.compute(theta_mce))[:, 0]
r_mce = r_mce - (r_mce.mean() - r_nfxp.mean())

mileage_bins = np.arange(env.num_mileage_bins)
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(mileage_bins, r_nfxp, label="NFXP structural utility", color="#1F4E79", lw=2)
ax.plot(mileage_bins, r_mce, label="MCE-IRL recovered reward", color="#8B0000", lw=2, ls="--")
ax.set_xlabel("Mileage bin")
ax.set_ylabel("Reward (action: continue)")
ax.legend(frameon=False)
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "fig3_mce_irl_reward.pdf"
fig.savefig(out)
print(f"wrote {out}")
