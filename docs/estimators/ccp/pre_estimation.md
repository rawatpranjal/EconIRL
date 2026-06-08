# Pre Estimation

Run support checks before treating CCP as a structural estimate. The estimator
can run on sparse data, but the interpretation depends on first-stage support.

## Required Checks

| Check | Why It Matters |
| --- | --- |
| Feature rank | Reward parameters are not identified when action-dependent features are collinear. |
| Feature condition number | Near-collinearity inflates standard errors. |
| State coverage | Unvisited states require extrapolated CCPs. |
| State-action coverage | One-action states make counterfactual action values weakly supported. |
| Minimum positive CCP | Very small probabilities make the log correction unstable. |
| Transition row sums | CCP inversion assumes valid transition probabilities. |

## Canonical Validation Diagnostics

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 4 of 4 | pass |
| Feature condition number | 4.512 | pass |
| Transition row error | 2.42e-8 | pass |
| Observed states | 21 of 21 | pass |
| State-action coverage | 1.000 | pass |
| Minimum action share | 0.325 | pass |
| Minimum positive CCP | 0.153 | pass |
| Exit and absorbing anchor | true | pass |

The canonical cell has full support, so the CCP validation claim is about the
estimator rather than a sparse first-stage policy.

## Practical Rules

If many states are unvisited, pool states or use a coarser state grid. If many
states have only one action, report CCP as a weak-support estimate and compare
against NFXP. If the feature matrix is rank deficient, change the reward
specification before fitting any structural estimator.
