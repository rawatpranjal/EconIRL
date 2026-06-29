# Counterfactuals

Read this page as behavioral counterfactual evidence. The reported regret is
about the policy induced by the learned reward, not a structural parameter
being re-solved under new primitives.

f-IRL recovers a tabular state reward rather than structural utility parameters,
so counterfactual analysis works differently from structural estimators. The
recovered reward is not in the same parameterization as the data-generating
utility. It cannot be re-solved under a reward shift in the structural sense.
Instead, the post-fit policy is used directly for behavioral prediction.

The validation checks whether the recovered policy still performs well in each
counterfactual world, using welfare regret as the measure: the loss relative to
the policy that would be optimal in that world.

## Counterfactual Families

The primary cell checks recovered counterfactual behavior against exact oracle
objects for three intervention families:

| Family | Intervention | Checked against |
| --- | --- | --- |
| Type A | Reward shift (a payoff component changes). | Oracle policy, value, and welfare regret. |
| Type B | Transition change (the dynamics change). | Oracle policy, value, and welfare regret. |
| Type C | Action removal (one action is penalized away). | Oracle policy, value, and welfare regret. |

## Reported Results (Primary Cell)

| Counterfactual | Policy TV | Regret | Value RMSE |
| --- | --- | --- | --- |
| Type A | 0.0102 | 0.00708 | 0.00710 |
| Type B | 0.0151 | 0.01236 | 0.01237 |
| Type C | 0.00770 | 0.00273 | 0.00277 |

These results hold in the primary source-paper cell (state marginal matching,
state-only reward). They reflect the quality of the recovered policy rather than
the ability to re-solve a structural reward under each intervention.

The full record is in
[f_irl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/f_irl.json).
The diagnostic action-dependent cell is not included in counterfactual evidence,
because that cell fails the reward-range check and the resulting policy is not a
reliable starting point.

## Interpretation

Low regret in the primary cell means the soft-optimal policy recovered under
the state-marginal objective happens to behave well under the three intervention
types tested. This is not the same as structural re-solving: the reward was not
updated to reflect the new world, and the policy was not re-derived from first
principles. For interventions that substantially change the optimal action
distribution, a structural estimator (NFXP, CCP, UFXP) with a linear utility
specification will produce more reliable counterfactual predictions.
