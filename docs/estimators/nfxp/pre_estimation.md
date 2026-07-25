# Pre-Estimation Checks

## Important Links

- [NFXP Overview](../nfxp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Bus Engine Example](rust_bus.md)

Read this page before fitting NFXP. These checks support structural
interpretation. A satisfactory result does not prove the model. A poor result
makes the reward estimate hard to interpret.

NFXP maximizes the conditional log likelihood over the reward parameters, so
the model must be identified and numerically stable before optimization starts.
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

Feature matrices that copy state-only features identically across actions
collapse action-specific payoff differences and leave those differences
unidentified.
Data with almost no replacement choices can fit in-sample behavior while
leaving the replacement cost weakly identified. Transition matrices with the
wrong orientation produce plausible arrays but wrong economics. When state
coverage is thin, UFXP uses optimal weighting for missing states. NFXP instead
pools all observations through the likelihood.
