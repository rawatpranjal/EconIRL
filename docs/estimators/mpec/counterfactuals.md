# Counterfactuals

Read this page as simulation-harness counterfactual evidence for the MPEC
structural target. The estimator target supports re-solving, even though the
public wrapper does not expose the same shortcut method as NFXP.

MPEC estimates structural reward parameters and a value function, but it does
not currently expose the same one-call dataframe wrapper `counterfactual`
method as NFXP and CCP. Counterfactual evidence therefore comes from the
simulation harness.

The harness re-solves the structural model under intervention-specific oracle
objects and compares MPEC's recovered structural object to those oracle
solutions.

## Counterfactual Families

The simulation harness evaluates three counterfactual families against oracle
solutions.

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
| Type C | 0.003549 | 3.56e-5 | 0.000114 | 0.000086 |

The regret values report how the policy from the recovered reward compares
with the oracle counterfactual policy.
