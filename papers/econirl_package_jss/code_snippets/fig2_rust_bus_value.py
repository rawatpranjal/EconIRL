"""Figure 2: converged integrated value function V(s) vs mileage on the Rust
bus panel after NFXP.

Reproduces: figures/fig2_rust_bus_value.pdf.
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

env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
panel = load_rust_bus(as_panel=True)
utility = LinearUtility.from_environment(env)
problem = env.problem_spec
transitions = jnp.asarray(env.transition_matrices, dtype=jnp.float64)

est = NFXPEstimator(optimizer="BHHH", inner_solver="policy", compute_hessian=True)
result = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)

V = np.asarray(result.value_function)
mileage_bins = np.arange(V.shape[0])

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(mileage_bins, V, color="#1F4E79", lw=2)
ax.set_xlabel("Mileage bin")
ax.set_ylabel("Integrated value V(s)")
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "fig2_rust_bus_value.pdf"
fig.savefig(out)
print(f"wrote {out}")
