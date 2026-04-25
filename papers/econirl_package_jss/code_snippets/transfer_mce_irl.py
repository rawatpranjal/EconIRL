"""TT1 transfer test for MCE-IRL on the Rust bus panel under perturbed
transitions. Reports the percent transfer value of the recovered reward
relative to the structural ground-truth value.

Reproduces: row TT1 of transfer_results.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import jax.numpy as jnp

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


def perturb(P, key_seed):
    rng = np.random.default_rng(key_seed)
    noise = rng.beta(2, 2, size=P.shape)
    Q = np.asarray(P) * noise
    Q = Q / Q.sum(axis=-1, keepdims=True)
    return jnp.asarray(Q, dtype=jnp.float64)


env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
panel = load_rust_bus(as_panel=True)
linear = LinearUtility.from_environment(env)
adr = ActionDependentReward(
    feature_matrix=linear.feature_matrix,
    parameter_names=linear.parameter_names,
)
problem = env.problem_spec
P_train = jnp.asarray(env.transition_matrices, dtype=jnp.float64)
P_test = perturb(P_train, SEED + 1)

est = MCEIRLEstimator(config=MCEIRLConfig(optimizer="L-BFGS-B", inner_solver="policy", compute_se=False))
res = est.estimate(panel=panel, utility=adr, problem=problem, transitions=P_train)
theta = jnp.asarray(res.parameters, dtype=jnp.float64)
r = jnp.asarray(adr.compute(theta), dtype=jnp.float64)

op_test = SoftBellmanOperator(problem, P_test)
val_test = policy_iteration(op_test, r, tol=1e-10, max_iter=200, eval_method="matrix")

# percent of optimal value: optimal is the value under the true reward and same transitions.
linear_true = LinearUtility.from_environment(env)
true_r = jnp.asarray(linear_true.compute(jnp.array([0.001, 3.0])), dtype=jnp.float64)
opt_val = policy_iteration(op_test, true_r, tol=1e-10, max_iter=200, eval_method="matrix").value_function

pct = float(np.mean(np.asarray(val_test.value_function) / np.asarray(opt_val)) * 100)
row = {"estimator": "MCE-IRL", "percent_transfer_value": f"{pct:.2f}"}
out = Path(__file__).resolve().parents[1] / "transfer_results.csv"
write_header = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(row)
