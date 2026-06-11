# Counterfactuals

The public `counterfactual` method handles structural parameter changes. It
solves the fitted model again after changing one or more fitted parameters.

```python
cf = model.counterfactual(RC=4.0)

print(cf.params)
print(cf.value_function)
print(cf.policy)
```

The method returns a parameter dictionary, value function, and policy. It does
not currently expose a one-call public interface for changing transitions or
disabling actions.

## Counterfactual Families

The simulation harness evaluates three broader counterfactual
families against oracle solutions.

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-anchor action. | Action-set or design counterfactual. |

## Reported Results

These rows come from the same simulation results file used on the
[simulation study page](validation.md).

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.005109 | 7.56e-5 | 0.000238 | 0.000213 |
| Type B | 0.005457 | 8.20e-5 | 0.000363 | 0.000362 |
| Type C | 0.003548 | 3.56e-5 | 0.000114 | 0.000086 |

The regret values report how the policy from the recovered reward compares
with the oracle counterfactual policy.
