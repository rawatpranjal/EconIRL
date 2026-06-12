# Bus Engine Example

The bundled bus engine replacement dataset gives a quick end-to-end smoke
test: load the panel, fit, inspect, and run a counterfactual.

```python
from econirl.datasets import load_rust_bus
from econirl import UFXP

df = load_rust_bus()

model = UFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.se_)
print(model.summary())

# What if replacement became 50 percent more expensive?
cf = model.counterfactual(RC=model.params_["RC"] * 1.5)
print(cf.policy)
```

## Interpretation

The linear-cost utility has two parameters: the operating cost slope
(`theta_c`, the per-bin cost of accumulating mileage) and the replacement cost
(`RC`, paid when the engine is replaced and the mileage resets). UFXP recovers
both from the inverted choice probabilities and prices their uncertainty with
the efficient moment variance.

## Replication Boundary

This page is a package smoke test on the bundled dataset, not a full
historical replication of the original study. The estimator's recovery
properties are established on a synthetic cell whose data-generating process
is fully specified; see the
[Simulation Study](validation.md) page. The
[bus engine simulation page](../../simulation_studies/rust_bus.md) compares
UFXP against the full estimator roster on a synthetic bus engine panel.
