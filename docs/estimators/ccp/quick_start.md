# Quick Start

## Important Links

- [CCP Overview](../ccp.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Simulation Study](validation.md)
- [Counterfactuals](counterfactuals.md)
- [Bus Engine Example](rust_bus.md)

This page shows the public CCP wrapper. The snippet checks that the estimator
runs and returns the structural objects. The pre-estimation page covers support
and first-stage CCP quality.

The public API follows the same sklearn convention as NFXP. Create an
estimator, call `fit`, and read fitted attributes.

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import CCP

df = load_rust_bus()

model = CCP(
    n_states=90,
    discount=0.9999,
    utility=rust_bus_reward_spec(90),
    num_policy_iterations=3,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

for name in model.params_:
    print(f"{name}: estimate={model.params_[name]:.6f}, se={model.se_[name]:.6f}")
print(f"policy_shape={model.policy_.shape}")
print(f"transition_source={model.transition_source_}")
```

**Result**

```text
operating_cost: estimate=0.000995, se=0.000421
replacement_cost: estimate=3.072211, se=0.074237
policy_shape=(90, 2)
transition_source=estimated from fitted panel
```

The fitted estimator exposes structural parameters, standard errors, a policy,
a value function, and a likelihood.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for the structural parameters. |
| `coef_` | Coefficients as a numpy array. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated policy value function by state. |
| `transition_tensor_` | Transition probabilities used for each action. |
| `transition_source_` | Whether transitions came from the panel or were supplied. |
| `log_likelihood_` | CCP pseudo-log-likelihood at the fitted parameters. |
| `converged_` | Whether the requested CCP run completed successfully. |
| `npl_converged_` | Whether the policy sequence met the NPL stopping tolerance. |
| `termination_reason_` | Why the CCP run stopped. |

Set `num_policy_iterations=1` for a one-step Hotz-Miller estimate. Set it to a
larger positive integer for an exact number of NPL updates. Set it to `-1` to
iterate until the stopping tolerance or iteration cap.

## Counterfactual Example

```python
cf = model.counterfactual(replacement_cost=4.0)

print(f"replacement_cost={cf.params['replacement_cost']:.6f}")
print(f"P(replace | state=50)={cf.policy[50, 1]:.6f}")
```

**Result**

```text
replacement_cost=4.000000
P(replace | state=50)=0.054908
```

This solves the fitted model again with a higher replacement cost and returns
the new value function and policy.

## Advanced API

Use `econirl.estimation.ccp.CCPEstimator` when you need direct control over
panel objects, utility objects, transition tensors, CCP smoothing, NPL stopping
rules, or diagnostic metadata.
