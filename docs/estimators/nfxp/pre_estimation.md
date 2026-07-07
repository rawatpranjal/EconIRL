# Pre-Estimation Checks

Read this page before fitting NFXP. These checks are gates for structural
interpretation: passing them does not prove the model, but failing them makes
the reward estimate hard to read.

NFXP maximizes the conditional log likelihood over the reward parameters, so
identification and numerical stability need to hold before optimization starts.
Run these checks before treating a result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Rank below the number of parameters means theta is not identified. |
| Feature condition number | A high condition number signals unstable estimates. |
| Transition row sums | Each transition row must be a valid probability distribution. |
| Transition orientation | NFXP expects transitions in the $(A, S, S)$ orientation. |
| State coverage | Unobserved states produce weak or degenerate likelihood regions. |
| Action support | Rare actions leave their payoff weakly identified. |
| Reward normalization | Reward level and scale need a valid anchor. |

## Canonical Simulation Checks

Values from the canonical synthetic run (see [Simulation Study](validation.md)):

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.512 | pass |
| Transition row error | 2.42e-8 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Action shares | 0.345, 0.330, 0.325 | pass |
| Minimum action share | 0.325 | pass |
| Exit and absorbing anchor | true | pass |

## Common Risk Patterns

Feature matrices with state-only features copied identically across actions
collapse the action-specific payoff differences, leaving them unidentified.
Data with almost no replacement choices can fit in-sample behavior while
leaving the replacement cost weakly identified. Transition matrices with the
wrong orientation produce plausible arrays but wrong economics. When state
coverage is thin, UFXP's optimal weighting handles missing states more
gracefully; NFXP pools all observations through the likelihood and degrades
more slowly in coverage.
