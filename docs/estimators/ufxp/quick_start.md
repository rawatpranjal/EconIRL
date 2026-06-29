# Quick Start

This is the smallest UFXP path: fit the unnested structural estimator and read
the same kind of reward, policy, and value objects as NFXP or CCP. The shortcut
depends on empirical choice-probability support.

```python
from econirl.datasets import load_rust_bus
from econirl import UFXP

df = load_rust_bus()

model = UFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)          # {"theta_c": ..., "RC": ...}
print(model.se_)              # standard errors (optimal weighting)
print(model.summary())
```

Fitted attributes follow the same convention as NFXP and CCP:

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated parameters by name. |
| `se_` | Standard errors from the efficient moment variance. |
| `coef_` | Coefficients as a numpy array. |
| `log_likelihood_` | Log-likelihood evaluated at the estimate. |
| `policy_` | Choice probabilities P(a given s), shape (n_states, n_actions). |
| `value_` | Value function V(s), shape (n_states,). |
| `converged_` | Whether the closed-form solve had full rank. |

## Weighting Modes

```python
model = UFXP(n_states=90, weights="optimal")   # default: efficient, with SEs
model = UFXP(n_states=90, weights="random",    # plain random projections
             num_projections=64)               # consistent, no SEs
```

The default optimal weighting is the one to use; the random-projection mode is
the paper's baseline construction and is kept for comparison.

## Full Estimator API

The underlying `UFXPEstimator` accepts a `Panel` and an explicit utility
specification, problem definition, and transition tensor, mirroring the other
low-level estimators:

```python
from econirl.estimation import UFXPEstimator

result = UFXPEstimator().estimate(panel, utility, problem, transitions)
```
