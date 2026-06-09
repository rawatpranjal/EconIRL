# Counterfactuals

MPEC estimates structural reward parameters and a value function, but it does
not currently expose the same one-call dataframe wrapper `counterfactual`
method as NFXP and CCP. Counterfactual evidence therefore comes from the
known-truth validation harness.

The harness re-solves the structural model under intervention-specific oracle
objects and compares MPEC's recovered structural object to those oracle
solutions.

## Counterfactual Families

The known-truth validation harness evaluates three broader counterfactual
families against oracle solutions.

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-anchor action. | Action-set or design counterfactual. |

## Reported Results

These rows come from the same known-truth artifact used on the
[validation page](validation.md).

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.005109 | 7.56e-5 | 0.000238 | 0.000213 |
| Type B | 0.005457 | 8.20e-5 | 0.000363 | 0.000362 |
| Type C | 0.003549 | 3.56e-5 | 0.000114 | 0.000086 |

The regret values report how the policy from the recovered reward compares
with the oracle counterfactual policy.
