# Counterfactuals

## Important Links

- [NFXP Overview](../nfxp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Bus Engine Example](rust_bus.md)

Read this page as structural re-solving. NFXP estimates reward parameters in a
model that can be solved again after a primitive changes.

NFXP recovers the utility parameters in the same parameterization as the
data-generating process, so counterfactual analysis is straightforward:
change a parameter, re-solve the dynamic program once, and read off the new
policy and value function.

```python
cf = model.counterfactual(replacement_cost=4.0)

print(
    f"replacement_cost: {model.params_['replacement_cost']:.6f}"
    f" -> {cf.params['replacement_cost']:.6f}"
)
print(
    f"P(replace | state=50): {model.predict_proba([50])[0, 1]:.6f}"
    f" -> {cf.policy[50, 1]:.6f}"
)
```

**Result**

```text
replacement_cost: 3.072264 -> 4.000000
P(replace | state=50): 0.086333 -> 0.055196
```

The environment can change as well. For example, a new maintenance technology
may alter how engines deteriorate:

```python
alternative_transitions = model.transition_tensor_.copy()
alternative_transitions[0] = 0.0

for state in range(model.n_states):
    for increment, probability in enumerate([0.80, 0.15, 0.05]):
        next_state = min(state + increment, model.n_states - 1)
        alternative_transitions[0, state, next_state] += probability

cf_transition = model.counterfactual(transitions=alternative_transitions)

print(
    f"P(replace | state=50): {model.predict_proba([50])[0, 1]:.6f}"
    f" -> {cf_transition.policy[50, 1]:.6f}"
)
```

**Result**

```text
P(replace | state=50): 0.086333 -> 0.088450
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
