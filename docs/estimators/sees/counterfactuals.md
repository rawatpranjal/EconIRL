# Counterfactuals

SEES estimates structural reward parameters and a value approximation. The
counterfactual evidence comes from the simulation harness,
which re-solves oracle counterfactuals and compares them with policies implied
by the recovered SEES structural object.

See [Simulation Study](validation.md) for the generator, links to results files, and
hard gate definitions.

## Counterfactual Families

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-anchor action. | Action-set or design counterfactual. |

## Primary Results

These rows are the high-dimensional primary-cell results from
`sees_results.json`.

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.002096 | 2.78e-5 | 0.000234 | 0.000113 |
| Type B | 0.002421 | 3.67e-5 | 0.000309 | 0.000183 |
| Type C | 0.001231 | 6.91e-6 | 0.000027 | 0.000014 |

All three regrets are below the 0.01 reported gate. The table compares
intervened policies from the recovered SEES object with oracle counterfactual
policies.

## Low-Dimensional Sanity Check

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.004668 | 6.33e-5 | 0.000197 | 0.000179 |
| Type B | 0.004937 | 6.80e-5 | 0.000299 | 0.000299 |
| Type C | 0.003270 | 3.00e-5 | 0.000092 | 0.000071 |

The low-dimensional cell is not the primary SEES evidence. It reports the same
counterfactual families for the state-index basis.
