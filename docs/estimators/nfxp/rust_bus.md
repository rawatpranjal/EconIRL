# Bus Engine Example

## Important Links

- [NFXP Overview](../nfxp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Counterfactuals](counterfactuals.md)
- [Rust Replication](../../replications.md)

The bundled bus engine replacement dataset shows the complete workflow: load
the panel, estimate the costs, inspect uncertainty, and run a counterfactual.

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.se_)
print(model.summary())

# What if replacement became 50 percent more expensive?
cf = model.counterfactual(replacement_cost=model.params_["replacement_cost"] * 1.5)
print(cf.summary())
```

## Estimation

The `rust_bus_reward_spec` specification estimates two parameters: the
operating cost slope over mileage states (`operating_cost`) and the flat
replacement cost (`replacement_cost`).

| Parameter | Estimate | Standard error | 95 percent interval |
| --- | ---: | ---: | ---: |
| Operating cost | 0.0010 | 0.0004 | [0.0002, 0.0018] |
| Replacement cost | 3.0723 | 0.0740 | [2.9273, 3.2172] |

The positive operating-cost estimate means keeping an engine becomes less
attractive as mileage rises. The fitted policy gives the replacement
probability at each mileage state:

```python
states = [0, 10, 50, 89]
print(model.predict_proba(states))
```

## Inference

The standard errors use the robust sandwich covariance estimate. On these data,
the operating-cost estimate has a p-value of 0.010 and the replacement-cost
estimate has a p-value below 0.001. The [Simulation Study](validation.md)
checks repeated-sample interval coverage on 1,000 independently simulated
panels.

## Counterfactual

Increasing the fitted replacement cost by 50 percent lowers the long-run
replacement rate from 5.3 percent to 2.2 percent. Mean long-run mileage rises
from 10.5 to 20.6 states. The model therefore translates the estimated costs
into a direct prediction about fleet behavior.

This worked example uses the bundled data. The
[Rust replication](../../replications.md) reproduces the published table,
while the
[bus engine simulation study](../../simulation_studies/rust_bus.md) compares
NFXP with other estimators.
