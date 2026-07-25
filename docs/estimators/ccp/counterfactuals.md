# Counterfactuals

## Important Links

- [CCP Overview](../ccp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Bus Engine Example](rust_bus.md)

Use the public `counterfactual` method to solve a fitted CCP model under new
reward parameters or a replacement transition tensor.

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
`transitions=new_tensor` to change the transition law. The tensor orientation
is `(n_actions, n_states, n_states)`. Removing an action requires a model with
a reduced action space.

## Results

The 100-state study evaluates reward and transition changes over 20
independent panels:

| Change | Mean policy error | Mean value loss |
| --- | ---: | ---: |
| Increase the first reward parameter by 1.0 | 0.0022 | 0.000184 |
| Slow deterioration | 0.0019 | 0.000083 |

Policy error compares the fitted counterfactual policy with the policy from
the true parameters. Value loss measures the cost of using the fitted policy
instead of the true-parameter policy. See the
[Simulation Study](validation.md) for the estimation and inference results.
