# Counterfactuals

NNES estimates structural reward parameters and a neural value approximation.
The counterfactual evidence comes from the known-truth validation harness,
which re-solves oracle counterfactuals and compares them with policies implied
by the recovered NNES structural object.

See [Validation](validation.md) for the generator, artifact links, and hard
gate definitions.

## Counterfactual Families

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-anchor action. | Action-set or design counterfactual. |

## Primary Results

These rows are the high-dimensional primary-cell results from
`nnes_results.json`.

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.021578 | 0.002165 | 0.007880 | 0.004865 |
| Type B | 0.021651 | 0.002128 | 0.007916 | 0.005559 |
| Type C | 0.013053 | 0.000920 | 0.002325 | 0.001314 |

All three regrets are below the 0.05 reported gate. The table compares
intervened policies from the recovered NNES object with oracle counterfactual
policies.

## Low-Dimensional Sanity Check

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.005080 | 0.000083 | 0.000271 | 0.000224 |
| Type B | 0.005505 | 0.000089 | 0.000335 | 0.000332 |
| Type C | 0.002695 | 0.000021 | 0.000076 | 0.000055 |

The public `NNES` wrapper exposes fitted rewards, policies, values, and choice
probabilities. It does not currently expose a wrapper-level `counterfactual()`
method, so this page reports the harness-level counterfactual evidence rather
than promising a public counterfactual API.
