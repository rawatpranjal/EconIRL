# Rust Bus Engine Example

The Rust bus-engine replacement problem is the canonical dynamic discrete
choice example. A bus operator observes mileage and chooses whether to keep the
current engine or replace it.

EconIRL ships a bundled Rust-style dataset that is suitable for a quick public
TD-CCP smoke test.

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

The `linear_cost` specification estimates two parameters. The first is the
operating cost slope over mileage states. The second is the replacement cost.
The fitted policy gives the replacement probability by mileage state.

```python
states = [0, 10, 50, 89]
print(model.predict_proba(states))
```

## Transition Boundary

TD-CCP uses observed successor tuples to estimate the recursive CCP terms.
The wrapper can estimate transitions from the panel for final fitted policy
and value outputs, but the structural parameter step is not a transition-density
model.

## Replication Boundary

This page is a package smoke example, not the TD-CCP release certification.
The reported validation evidence is the hard finite-theta known-truth cell,
which has known reward, policy, value, Q, and counterfactual oracle objects.
For paper Algorithm 2 evidence, use the validation page rather than this Rust
bus wrapper example.
