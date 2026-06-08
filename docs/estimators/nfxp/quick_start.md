# Quick Start

The public API follows the sklearn convention. Create an estimator, call
`fit`, and read fitted attributes.

```python
from econirl.datasets import load_rust_bus
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.se_)
print(model.policy_.shape)
```

Output

```text
{'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
{'theta_c': 0.0003951203834731132, 'RC': 0.0747256949543953}
(90, 2)
```

The fitted estimator exposes structural parameters, standard errors, a policy,
a value function, and a likelihood.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for the structural parameters. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated value function by state. |
| `log_likelihood_` | Maximized conditional choice log likelihood. |

## Counterfactual Example

```python
cf = model.counterfactual(RC=4.0)

print(cf.params)
print(cf.policy[50, 1])
```

Output

```text
{'theta_c': 0.0010028828858836278, 'RC': 4.0}
0.05519477716656161
```

This solves the fitted model again with a higher replacement cost and returns
the new value function and policy.

## Lower-Level Control

Use `econirl.estimation.nfxp.NFXPEstimator` when you need direct control over
panel objects, utility objects, transition tensors, optimizer options, or
diagnostic metadata.
