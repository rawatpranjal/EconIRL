# Bus Engine Example

## Important Links

- [NFXP Overview](../nfxp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Counterfactuals](counterfactuals.md)
- [Rust Replication](../../replications.md)

This example uses the bundled synthetic bus engine replacement panel. It
follows the Rust setting but is not the original Rust sample. The example
estimates costs, inspects uncertainty, and runs a counterfactual.

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

for name in model.params_:
    print(f"{name}: estimate={model.params_[name]:.6f}, se={model.se_[name]:.6f}")
```

**Result**

```text
operating_cost: estimate=0.001003, se=0.000389
replacement_cost: estimate=3.072264, se=0.073965
```

## Estimation

`rust_bus_reward_spec` estimates two parameters: the operating cost slope over
mileage states (`operating_cost`) and the flat replacement cost
(`replacement_cost`).

| Parameter | Estimate | Standard error | 95 percent interval |
| --- | ---: | ---: | ---: |
| Operating cost | 0.0010 | 0.0004 | [0.0002, 0.0018] |
| Replacement cost | 3.0723 | 0.0740 | [2.9273, 3.2172] |

The positive operating-cost estimate means keeping an engine becomes less
attractive as mileage rises. The fitted policy gives the replacement
probability at each mileage state:

```python
states = [0, 10, 50, 89]
for state, probabilities in zip(states, model.predict_proba(states)):
    print(
        f"{state:2d}: keep={probabilities[0]:.6f}, "
        f"replace={probabilities[1]:.6f}"
    )
```

**Result**

```text
 0: keep=0.955734, replace=0.044266
10: keep=0.947597, replace=0.052403
50: keep=0.913667, replace=0.086333
89: keep=0.884130, replace=0.115870
```

## Inference

The standard errors use the robust sandwich covariance estimate. On these data,
the operating-cost estimate has a p-value of 0.010 and the replacement-cost
estimate has a p-value below 0.001. The [Simulation Study](validation.md)
checks repeated-sample interval coverage on 1,000 independently simulated
panels.

## Counterfactual

Increasing the fitted replacement cost by 50 percent lowers the long-run
replacement rate from 5.3 percent to 2.2 percent. The stationary mean state
index rises from 10.5 to 20.6. Under this counterfactual, buses run longer
before engine replacement.

```python
cf = model.counterfactual(
    replacement_cost=model.params_["replacement_cost"] * 1.5
)
print(
    f"replacement_cost: {model.params_['replacement_cost']:.6f}"
    f" -> {cf.params['replacement_cost']:.6f}"
)
print(
    f"P(replace | state=50): {model.predict_proba([50])[0, 1]:.6f}"
    f" -> {cf.policy[50, 1]:.6f}"
)
```

**Result**

```text
replacement_cost: 3.072264 -> 4.608396
P(replace | state=50): 0.086333 -> 0.042539
```

This worked example uses the bundled data. The
[Rust replication](../../replications.md) reproduces the published table,
while the
[bus engine simulation study](../../simulation_studies/rust_bus.md) compares
NFXP with other estimators.
