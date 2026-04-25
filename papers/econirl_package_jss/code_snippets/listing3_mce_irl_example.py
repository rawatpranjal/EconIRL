"""Listing 3: MCE-IRL on the Rust bus panel. Same workflow as Listing 2 but
recovers a reward function rather than a structural utility.

Reproduces: Listing 3.
"""
from __future__ import annotations

import numpy as np

SEED = 42
np.random.seed(SEED)

from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig

env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
panel = load_rust_bus(as_panel=True)
linear = LinearUtility.from_environment(env)
utility = ActionDependentReward(
    feature_matrix=linear.feature_matrix,
    parameter_names=linear.parameter_names,
)
problem = env.problem_spec
transitions = env.transition_matrices

est = MCEIRLEstimator(config=MCEIRLConfig(
    optimizer="L-BFGS-B",
    inner_solver="policy",
    inner_max_iter=200,
    outer_max_iter=500,
    outer_tol=1e-6,
    compute_se=True,
))
result = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
print(result.summary())
