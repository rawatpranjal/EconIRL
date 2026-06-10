# Quick Start

The public API follows the same sklearn convention as the other structural
estimators. Create an estimator, call `fit`, and read fitted attributes.

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

The fitted estimator exposes structural parameters, standard errors, a policy,
a value function, and likelihood diagnostics.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for the structural parameters. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated policy value function by state. |
| `log_likelihood_` | Maximized CCP pseudo log likelihood. |
| `ev_features_` | Continuation-value feature decomposition when exposed by the lower-level result. |

## Method Choices

```python
semigradient = TDCCP(method="semigradient", basis_type="polynomial")
encoded = TDCCP(method="semigradient", basis_type="encoded")
neural = TDCCP(method="neural", avi_iterations=20)
```

Use the semi-gradient path for the certified finite-theta workflow. Use the
encoded basis when the problem carries meaningful state encoders. The neural
AVI path is available for flexible approximation, but the public certified
artifact is the paper-faithful semi-gradient hard case.

## Lower-Level Control

Use `econirl.estimation.td_ccp.TDCCPEstimator` when you need direct control
over `Panel` objects, utility objects, problem specifications, basis settings,
cross-fitting, robust standard errors, or supplied transition tensors for final
policy/value evaluation.
