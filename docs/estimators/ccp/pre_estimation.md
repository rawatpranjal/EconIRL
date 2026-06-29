# Pre-Estimation Checks

CCP can show estimation risk for reasons that are visible before optimization
starts. Run these checks before treating a result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Reward parameters are not identified when action-dependent features are collinear. |
| Feature condition number | Near-collinearity inflates standard errors. |
| Transition row sums | Each transition row must be a valid probability distribution. |
| State coverage | Unvisited states require extrapolated CCPs. |
| State-action coverage | One-action states make counterfactual action values weakly supported. |
| Minimum positive CCP | Very small probabilities make the log correction unstable. |
| Reward normalization | Reward level and scale need a valid anchor. |
| Transition orientation | CCP expects transition tensors in action, state, next-state order. |

## Canonical Simulation Checks

The CCP results file records these pre-estimation checks.
See the [simulation study page](validation.md) for the generator script,
JSON results file.

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.512 | pass |
| Transition row error | 2.42e-8 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Action shares | 0.345, 0.330, 0.325 | pass |
| Minimum action share | 0.325 | pass |
| Minimum positive CCP | 0.153 | pass |
| Exit and absorbing anchor | true | pass |

The canonical cell has full support, so the CCP evidence is about the
estimator rather than a sparse first-stage policy.

## Common Risk Patterns

Data with many unvisited states force CCP to extrapolate the first-stage
policy. States with only one observed action make counterfactual action values
weakly supported. Too little smoothing can make log corrections unstable,
while too much smoothing biases the empirical policy toward uniform choice.
Transition matrices with wrong orientation can produce plausible arrays and
wrong economics.
