# Quick Start

```python
from econirl.datasets import load_rust_bus
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)   # {"theta_c": ..., "RC": ...}
print(model.se_)
print(model.summary())
```

Fitted attributes follow the same convention as CCP and UFXP:

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for the structural parameters. |
| `coef_` | Coefficients as a numpy array. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated value function by state. |
| `log_likelihood_` | Maximized conditional choice log likelihood. |
| `converged_` | Whether the outer optimizer reported convergence. |

## Counterfactual Example

```python
cf = model.counterfactual(RC=4.0)

print(cf.params)
print(cf.policy[50, 1])
```

This solves the fitted model again with a higher replacement cost and returns
the new value function and policy.

## Lower-Level Control

Use `econirl.estimation.nfxp.NFXPEstimator` when you need direct control over
panel objects, utility objects, transition tensors, optimizer options, or
diagnostic metadata:

```python
from econirl.estimation import NFXPEstimator

result = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10).estimate(
    panel, utility, problem, transitions
)
```
