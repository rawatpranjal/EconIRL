"""Listing 2b: bootstrap standard errors on the Rust bus NFXP fit. Demonstrates
that switching standard error method requires a single argument change.

Reproduces: follow-up listing in Section 4.1.
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

est = NFXPEstimator(optimizer="BHHH", compute_hessian=True)
result = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)

asym = result.standard_errors(method="asymptotic")
boot = result.standard_errors(method="bootstrap", n_bootstrap=200, seed=SEED)
print("asymptotic:", asym)
print("bootstrap :", boot)
