"""Figure 1: replacement probability vs mileage on the Rust bus panel. Shows
the baseline CCP and a counterfactual with the replacement cost reduced by
twenty-five percent.

Reproduces: figures/fig1_rust_bus_ccp.pdf.
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
from econirl.estimation.nfxp import NFXPEstimator
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration

env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
panel = load_rust_bus(as_panel=True)
utility = LinearUtility.from_environment(env)
problem = env.problem_spec
transitions = jnp.asarray(env.transition_matrices, dtype=jnp.float64)

est = NFXPEstimator(optimizer="BHHH", inner_solver="policy", compute_hessian=True)
result = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)

theta_hat = jnp.asarray(result.parameters, dtype=jnp.float64)
theta_cf = theta_hat.at[1].set(theta_hat[1] * 0.75)

operator = SoftBellmanOperator(problem, transitions)
r_hat = jnp.asarray(utility.compute(theta_hat), dtype=jnp.float64)
r_cf = jnp.asarray(utility.compute(theta_cf), dtype=jnp.float64)
pol_hat = policy_iteration(operator, r_hat, tol=1e-10, max_iter=200, eval_method="matrix").policy
pol_cf = policy_iteration(operator, r_cf, tol=1e-10, max_iter=200, eval_method="matrix").policy

mileage_bins = np.arange(env.num_mileage_bins)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(mileage_bins, np.asarray(pol_hat[:, 1]), label="baseline", color="#1F4E79", lw=2)
ax.plot(mileage_bins, np.asarray(pol_cf[:, 1]), label="RC reduced 25 percent", color="#8B0000", lw=2, ls="--")
ax.set_xlabel("Mileage bin")
ax.set_ylabel("Replacement probability")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "fig1_rust_bus_ccp.pdf"
fig.savefig(out)
print(f"wrote {out}")
