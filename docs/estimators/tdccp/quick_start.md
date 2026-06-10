# Quick Start

The public API follows the same convention as the other structural estimators:
create an estimator, call `fit`, and inspect fitted attributes.

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
print(model.se_)
print(model.policy_.shape)
```

The fitted estimator exposes the estimated reward parameters, standard errors,
policy, value function, and likelihood diagnostics.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated reward parameters |
| `se_` | Standard errors for those parameters |
| `policy_` | Estimated action probabilities by state |
| `value_` | Estimated value function by state |
| `log_likelihood_` | Maximized CCP pseudo log likelihood |
| `ev_features_` | Continuation-value feature decomposition when available |

## Method Choices

```python
semigradient = TDCCP(method="semigradient", basis_type="polynomial")
encoded = TDCCP(method="semigradient", basis_type="encoded")
neural = TDCCP(method="neural", avi_iterations=20)
```

Use the semigradient path for the current validated workflow. Use the encoded
basis when states already have meaningful numeric encoders. The neural AVI path
is available for flexible approximation, but it is not the current release
certification.

## Lower-Level Control

Use `econirl.estimation.td_ccp.TDCCPEstimator` when you need direct control
over panels, utility objects, basis settings, cross-fitting, robust standard
errors, or supplied transition tensors for policy and value evaluation.
