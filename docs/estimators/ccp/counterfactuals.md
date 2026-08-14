# Counterfactuals

## Important Links

- [CCP Overview](../ccp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Bus Engine Example](rust_bus.md)

## Quick Start

Use the public `counterfactual` method to solve a fitted conditional choice
probability (CCP) model under new reward parameters or a replacement transition
tensor.

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

The method returns a parameter dictionary, value function, and policy. Pass
`transitions=new_tensor` to change the transition law. A single call can change
reward parameters or transitions, but not both. The tensor orientation is
`(n_actions, n_states, n_states)`. The method does not support changes to the
action space.

## Results

The counterfactual experiment fits a three-stage nested pseudo-likelihood (NPL)
model to 20 independently generated panels. Each panel contains 160,000 choices
across 100 states. Every fit uses the transition tensor that generated its panel.

| Change | Mean policy TV | Mean value loss |
| --- | ---: | ---: |
| Increase the first reward parameter by 1.0 | 0.0022 | 0.000184 |
| Slow deterioration | 0.0019 | 0.000083 |

Policy TV is the state-averaged total-variation distance between the fitted
counterfactual policy and the policy from the true parameters. Value loss is
the expected discounted loss from following the fitted counterfactual policy.
It is evaluated under the true counterfactual reward and transitions from the
study's initial state distribution. See the
[Simulation Study](validation.md) for the estimation and inference results.
