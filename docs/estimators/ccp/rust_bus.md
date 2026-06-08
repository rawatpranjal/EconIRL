# Rust Bus Engine Example

The Rust bus-engine replacement problem is the canonical dynamic discrete
choice example. A bus operator observes mileage and chooses whether to keep the
current engine or replace it.

EconIRL ships a bundled Rust-style dataset that is suitable for a quick public
CCP smoke test.

```python
from econirl.datasets import load_rust_bus
from econirl import CCP

df = load_rust_bus()

model = CCP(
    n_states=90,
    discount=0.9999,
    utility="linear_cost",
    num_policy_iterations=10,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

## Interpretation

The `linear_cost` specification estimates two parameters. The first is the
operating cost slope over mileage states. The second is the replacement cost.
The fitted policy gives the replacement probability by mileage state.

```python
states = [0, 10, 50, 89]
print(model.predict_proba(states))
```

## Counterfactual Replacement Cost

```python
cf = model.counterfactual(RC=4.0)
print(cf.policy[50, 1])
```

Increasing replacement cost should reduce replacement probabilities in worn
states relative to a lower replacement-cost counterfactual.

## Replication Boundary

This page is a package smoke example, not the full historical Rust
replication. The certified release evidence is the known-truth validation
cell, because it has known reward, policy, value, Q, and counterfactual oracle
objects.
