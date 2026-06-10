# Counterfactuals

TD-CCP estimates structural parameters without fitting a transition-density
model. Counterfactual analysis is separate: after `theta` is estimated, policy
and value evaluation may still require supplied or estimated transitions for
the counterfactual environment.

The public wrapper does not currently expose the same one-call parameter
counterfactual interface as NFXP and CCP. The validation harness evaluates
counterfactuals directly after fitting the lower-level estimator.

## Counterfactual Families

The known-truth validation harness evaluates three counterfactual families
against oracle solutions.

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-baseline action. | Action-set or design counterfactual. |

## Reported Results

These rows come from the certified hard-case artifact used on the
[validation page](validation.md).

| Counterfactual | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A | 0.004643 | 0.001885 | 0.001883 |
| Type B | 0.004840 | 0.001867 | 0.001864 |
| Type C | 0.006953 | 0.003208 | 0.003200 |

The regret values report how the policy from the recovered structural reward
compares with the oracle counterfactual policy.

## Boundary

The TD-CCP paper's computational advantage is about structural parameter
estimation: it avoids estimating the original transition density to estimate
`theta`. Counterfactuals can involve new transition processes, so the
counterfactual step must still define the transition environment being
evaluated. In other words, TD-CCP separates structural-parameter estimation
from counterfactual transition modeling; it does not make counterfactual
evaluation transition-free.
