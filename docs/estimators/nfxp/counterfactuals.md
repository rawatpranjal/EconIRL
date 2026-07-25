# Counterfactuals

Read this page as structural re-solving. NFXP estimates reward parameters in a
model that can be solved again after a primitive changes.

NFXP recovers the utility parameters in the same parameterization as the
data-generating process, so counterfactual analysis is straightforward:
change a parameter, re-solve the dynamic program once, and read off the new
policy and value function.

```python
cf = model.counterfactual(replacement_cost=4.0)

print(cf.params)
print(cf.value_function)
print(cf.policy)
```

The environment can change as well. For example, a new maintenance technology
may alter how engines deteriorate:

```python
cf_transition = model.counterfactual(transitions=alternative_transitions)

print(cf_transition.value_function)
print(cf_transition.policy)
```

## Results

The 200-state study evaluates both kinds of change:

| Change | True shift | Policy error | Value loss |
| --- | ---: | ---: | ---: |
| Increase the first reward parameter by 1.0 | 0.0829 | 0.0064 | 0.0030 |
| Slow engine deterioration | 0.0454 | 0.0067 | 0.0018 |

Policy distance ranges from zero to one. Zero means the two policies choose
each action with the same probability in every state. The fitted model captures
both changes while remaining within 0.0067 of the true-parameter policy.

Expected-value loss compares the fitted counterfactual policy with the policy
computed from the true parameters. It is 0.0030 for the reward change and
0.0018 for the transition change. See the
[Simulation Study](validation.md) for the corresponding estimation and
inference results.
