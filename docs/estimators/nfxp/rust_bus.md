# Bus Engine Example

The bundled bus engine replacement dataset gives a quick end-to-end smoke
test: load the panel, fit, inspect, and run a counterfactual.

```python
from econirl.datasets import load_rust_bus
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.se_)
print(model.summary())

# What if replacement became 50 percent more expensive?
cf = model.counterfactual(RC=model.params_["RC"] * 1.5)
print(cf.policy)
```

## Interpretation

The `linear_cost` specification estimates two parameters: the operating cost
slope over mileage states (`theta_c`) and the flat replacement cost (`RC`).
The fitted policy gives the replacement probability by mileage state. You can
inspect the replacement probability at specific mileage bins:

```python
states = [0, 10, 50, 89]
print(model.predict_proba(states))
```

## Replication Boundary

This page is a package smoke test on the bundled dataset, not a full
historical replication of the original study. The estimator's recovery
properties are established on a synthetic cell whose data-generating process
is fully specified; see the [Simulation Study](validation.md) page. The
[bus engine simulation page](../../simulation_studies/rust_bus.md) compares
NFXP against the full estimator roster on a synthetic bus engine panel.
