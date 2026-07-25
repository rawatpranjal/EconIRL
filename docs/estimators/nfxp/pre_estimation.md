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

## Example Diagnostics

These values come from the 200-state example in the
[Simulation Study](validation.md):

| Diagnostic | Value |
| --- | ---: |
| Reward features | 3 |
| Design rank | 3 / 3 |
| Action-contrast rank | 3 / 3 |
| Observed states | 186 / 200 |
| Actions | 2 |
| Observations | 7,500 |

## Common Risk Patterns

Feature matrices with state-only features copied identically across actions
collapse the action-specific payoff differences, leaving them unidentified.
Data with almost no replacement choices can fit in-sample behavior while
leaving the replacement cost weakly identified. Transition matrices with the
wrong orientation produce plausible arrays but wrong economics. When state
coverage is thin, UFXP's optimal weighting handles missing states more
gracefully; NFXP pools all observations through the likelihood and degrades
more slowly in coverage.
