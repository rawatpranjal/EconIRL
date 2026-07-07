# Pre-Estimation Checks

Read this page before fitting GLADIUS. The anchor and action-difference
projection are what turn learned Q objects into reward evidence.

GLADIUS uses neural networks and an action-difference projection, so it carries
the usual high-dimensional estimation risks plus a few specific to the anchor
Bellman strategy.

| Check | Why it matters for GLADIUS |
| --- | --- |
| Feature rank | A rank-deficient action-difference feature matrix leaves a direction of theta undetermined in the projection. |
| Feature condition number | Ill-conditioning inflates the variance of the action-difference OLS. |
| State coverage | The projection is scored at all states; very thin coverage leaves the reward surface underidentified in those regions. |
| Action support per state | A state where one action is never taken makes the action-difference construction degenerate at that state. |
| Anchor validity | The anchor action must appear in every state for the anchor Bellman loss to pin Q everywhere. |
| Transition row sums | Transition tensors must be row-stochastic in the `(n_actions, n_states, n_states)` orientation; row errors above `1e-6` indicate a construction bug. |
| Network scaling | With a high discount factor ($\beta$ near 1), Q-values are large; the default `value_scale = 1/(1-beta)` rescales the MLP output to a numerically stable range. |

## Canonical Simulation Checks

Values from the primary synthetic cell (see [Simulation Study](validation.md)):

| Check | Value | Status |
| --- | --- |  --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.376 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Single-action states | 0 | pass |
| Anchor valid | true | pass |
| Max transition row error | 2.4e-08 | pass |

## Common Risk Patterns

The anchor Bellman loss is the main lever for structural reward recovery. When
anchor rewards are not supplied, Q is identified only up to a state-dependent
constant and the action-difference projection can still recover reward slopes,
but level parameters require the anchor to pin the scale. Check
`anchor_bellman_loss=True` and verify that `anchor_rewards` is a vector of
length `n_states` containing the correct known rewards for the anchor action.

Policy TV can be good even when raw Bellman reward or value recovery fails.
Inspect the `projected_reward_normalized_rmse` and `raw_bellman_reward_normalized_rmse`
fields in the results file alongside policy TV before drawing
any structural conclusions. A good imitation policy is a necessary but not
sufficient condition for structural counterfactual validity.
