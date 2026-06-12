# Counterfactuals

UFXP recovers the utility parameters in the same gauge as the data-generating
process, so counterfactual analysis works exactly as it does for NFXP: change
a parameter, re-solve the dynamic program once, and read off the new policy
and value function.

```python
result = model.counterfactual(RC=15.0)

print(result.params)   # parameters used for the counterfactual solve
print(result.policy)   # choice probabilities under the new world
print(result.value)    # value function under the new world
```

## Counterfactual Families

The validation cell checks recovered counterfactual behavior against exact
oracle objects for three intervention families:

| Family | Intervention | Checked against |
| --- | --- | --- |
| Type A | Reward shift (a payoff component changes). | Oracle policy, value, and welfare regret. |
| Type B | Transition change (the dynamics change). | Oracle policy, value, and welfare regret. |
| Type C | Action removal (one action is penalized away). | Oracle policy, value, and welfare regret. |

## Reported Results

On the canonical synthetic cell the welfare regret of UFXP's counterfactual
policies is below 0.0003 in every family (Type A 0.00016, Type B 0.00030,
Type C 0.00008; see the
[machine-readable results file](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ufxp.json)
and the [Simulation Study](validation.md) page). Because the recovered reward
lives in the linear feature gauge, it re-solves cleanly under all three
intervention types.
