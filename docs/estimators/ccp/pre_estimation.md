# Pre-Estimation Checks

## Important Links

- [CCP Overview](../ccp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Counterfactuals](counterfactuals.md)

Read this page before fitting CCP. These checks focus on support and
conditioning because the estimator leans on first-stage choice probabilities.

Some CCP estimation risks are visible before optimization starts. Run these
checks before treating a result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Reward parameters are not identified when action-dependent features are collinear. |
| Feature condition number | Near-collinearity inflates standard errors. |
| Transition row sums | Each transition row must be a valid probability distribution. |
| State coverage | Unvisited states require extrapolated CCPs. |
| State-action coverage | One-action states make counterfactual action values weakly supported. |
| Minimum positive CCP | Very small probabilities make the log correction unstable. |
| Reward normalization | Additive reward shifts do not change choice probabilities, and the shock scale must be normalized. |
| Transition orientation | CCP expects transition tensors in action, state, next-state order. |

## Simulation Checks

The first 20-state inference panel in the
[Simulation Study](validation.md) reports the following checks.

| Check | Value | Status |
| --- | ---: | --- |
| Reward features | 3 | pass |
| Feature rank | 3 / 3 | pass |
| Action-contrast rank | 3 / 3 | pass |
| Feature condition number | 1.000 | pass |
| Action-contrast condition number | 1.000 | pass |
| Observed states | 20 / 20 | pass |
| State-action coverage | 1.000 | pass |
| Single-action states | 0 | pass |
| States with fewer than five observations | 0 | pass |
| Action counts | 5,605 and 4,395 | pass |

The panel has full support, so the reported fit does not rely on uniform CCP
fallbacks for unvisited states.

## Common Risk Patterns

Data with many unvisited states force CCP to extrapolate the first-stage
policy. States with only one observed action make counterfactual action values
weakly supported. Too little smoothing can make log corrections unstable,
while too much smoothing biases the empirical policy toward uniform choice.
Transition matrices with wrong orientation can produce plausible arrays and
wrong economics.
