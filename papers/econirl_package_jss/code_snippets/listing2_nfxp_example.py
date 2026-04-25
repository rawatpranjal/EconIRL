"""Listing 2: full NFXP summary on the Rust bus panel. The console output is
the same text reproduced in Section 4.1 of the paper.

Reproduces: Listing 2.
"""
from __future__ import annotations

import numpy as np

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
transitions = env.transition_matrices

est = NFXPEstimator(
    optimizer="BHHH",
    inner_solver="policy",
    inner_tol=1e-12,
    inner_max_iter=200,
    compute_hessian=True,
    outer_tol=1e-3,
)
result = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
print(result.summary())
