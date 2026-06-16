# Counterfactuals

TD-CCP estimates reward parameters without fitting a transition-density model
for the original data-generating process. Counterfactual analysis is a separate
step. Once `theta` is estimated, policy and value evaluation still need a
transition environment for the counterfactual being studied.

The public wrapper does not yet expose the same one-call parameter
counterfactual interface as NFXP and CCP. The simulation harness evaluates
counterfactuals directly after fitting the lower-level estimator.

## Counterfactual Families

The simulation harness checks three common counterfactual types.

| Type | Intervention | What it tests |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed | Payoff counterfactuals |
| Type B | Change transitions and hold rewards fixed | State-dynamics counterfactuals |
| Type C | Disable one non-baseline action | Action-set counterfactuals |

## Reported Results

These rows come from the
[results file](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp.json)
and the [Simulation Study](validation.md) page.

| Counterfactual | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A | 0.004643 | 0.001885 | 0.001883 |
| Type B | 0.004840 | 0.001867 | 0.001864 |
| Type C | 0.006953 | 0.003208 | 0.003200 |

Regret measures how much value is lost when the policy implied by the
estimated reward is used instead of the oracle counterfactual policy.

## Boundary

TD-CCP separates reward-parameter estimation from transition modeling. That is
the computational advantage. It does not remove the need for transition
information in counterfactual evaluation. If the counterfactual changes the
transition process, that new process must still be supplied or estimated for
the evaluation step.
