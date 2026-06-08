# Quick Start

The public API follows the same sklearn convention as NFXP. Create an
estimator, call `fit`, and read fitted attributes.

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
print(model.se_)
print(model.policy_.shape)
```

The fitted estimator exposes structural parameters, standard errors, a policy,
a value function, and a likelihood.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for the structural parameters. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated policy value function by state. |
| `log_likelihood_` | Maximized CCP pseudo log likelihood. |

## Hotz-Miller And NPL

```python
one_step = CCP(n_states=90, discount=0.9999, num_policy_iterations=1)
npl = CCP(n_states=90, discount=0.9999, num_policy_iterations=10)
```

The one-step model uses the empirical CCPs once. The NPL model updates the
policy after each pseudo-likelihood fit.

## Counterfactual Example

```python
cf = model.counterfactual(RC=4.0)

print(cf.params)
print(cf.policy[50, 1])
```

The counterfactual method solves the fitted structural model again under the
changed parameter vector. This mirrors the NFXP public API.

## Lower-Level Control

Use `econirl.estimation.ccp.CCPEstimator` when you need direct control over
panel objects, utility objects, transition tensors, CCP smoothing, NPL stopping
rules, or diagnostic metadata.
