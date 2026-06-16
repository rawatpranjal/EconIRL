# Counterfactuals

NFXP recovers the utility parameters in the same parameterization as the
data-generating process, so counterfactual analysis is straightforward:
change a parameter, re-solve the dynamic program once, and read off the new
policy and value function.

```python
cf = model.counterfactual(RC=4.0)

print(cf.params)
print(cf.value_function)
print(cf.policy)
```

## Counterfactual Families

The simulation cell checks recovered counterfactual behavior against exact
oracle objects for three intervention families:

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-anchor action. | Action-set or design counterfactual. |

## Reported Results

These rows come from the same results file used on the
[Simulation Study](validation.md) page (see also the
[results file](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nfxp.json)).

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.005109 | 7.56e-5 | 0.000238 | 0.000213 |
| Type B | 0.005457 | 8.20e-5 | 0.000363 | 0.000362 |
| Type C | 0.003548 | 3.56e-5 | 0.000114 | 0.000086 |

The regret values are small because the recovered reward is close enough to
the true reward that re-solving the dynamic program under each intervention
produces nearly the same policy as the true-parameter solution.
