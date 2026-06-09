# Rust Bus Engine Example

The Rust bus-engine replacement problem is the canonical dynamic discrete
choice example. A bus operator observes mileage and chooses whether to keep
the current engine or replace it.

SEES can be used through the dataframe wrapper for a quick package smoke test.

```python
from econirl.datasets import load_rust_bus
from econirl import SEES

df = load_rust_bus()

model = SEES(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    basis_type="fourier",
    basis_dim=8,
    penalty_weight=0.01,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.policy_[[0, 20, 40, 60, 80], 1])
```

## Interpretation

The Rust-style linear utility specification estimates operating-cost and
replacement-cost parameters. The fitted policy gives the replacement
probability by mileage bin. The fitted `alpha_` vector stores the sieve
coefficients used to approximate the value function.

## Lower-Level Variant

Use `econirl.estimation.sees.SEESEstimator` for package-native `Panel`
objects, custom reward features, supplied transition tensors, or exact control
over the basis and Bellman penalty.

```python
from econirl.estimation.sees import SEESEstimator

estimator = SEESEstimator(
    basis_type="bspline",
    basis_dim=21,
    penalty_weight=100.0,
)
summary = estimator.estimate(panel, utility, problem, transitions)

print(summary.parameters)
print(summary.metadata["bellman_violation"])
```

## Replication Boundary

This page is a package smoke example, not the full historical Rust
replication. The certified release evidence is the known-truth validation
artifact, because it has known reward, policy, value, Q, and counterfactual
oracle objects. See [Validation](validation.md) for that evidence path.
