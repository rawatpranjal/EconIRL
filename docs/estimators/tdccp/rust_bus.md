# Rust Bus Engine Example

The Rust bus-engine replacement problem is the standard dynamic discrete
choice example. A bus operator observes mileage and chooses whether to keep the
current engine or replace it.

EconIRL ships a small Rust-style dataset for a public TD-CCP smoke test.

```python
from econirl.datasets import load_rust_bus
from econirl import TDCCP

df = load_rust_bus()

model = TDCCP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    method="semigradient",
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

## Interpretation

The `linear_cost` specification estimates two parameters: the mileage cost
slope and the replacement cost. The fitted policy gives the replacement
probability at each mileage state.

```python
states = [0, 10, 50, 89]
print(model.predict_proba(states))
```

## Boundary

This page is a usage example, not the release certification. The certified
evidence is the known-truth validation case, where the true reward, policy,
value function, Q function, and counterfactual decisions are available for
comparison.

The wrapper may estimate transitions from the panel for fitted policy and
value outputs. The structural parameter step remains separate from
transition-density modeling.
