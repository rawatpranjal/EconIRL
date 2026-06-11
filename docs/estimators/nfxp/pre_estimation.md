# Pre-Estimation Checks

NFXP can show estimation risk for reasons that are visible before optimization
starts. Run these checks before treating a result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Rank below the number of parameters means theta is not identified. |
| Feature condition number | A high condition number signals unstable estimates. |
| Transition row sums | Each transition row must be a valid probability distribution. |
| State coverage | Unobserved states produce weak or degenerate likelihood regions. |
| Action support | Rare actions make their payoff weakly identified. |
| Reward normalization | Reward level and scale need a valid anchor. |
| Transition orientation | NFXP expects action, state, next-state transition tensors internally. |

## Canonical Simulation Checks

The NFXP results file records these pre-estimation checks.
See the [simulation study page](validation.md) for the generator script,
rendered table source, and JSON results file.

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

Feature matrices with copied state-only features across actions can collapse
the action-specific payoff differences. Data with almost no replacement
choices can fit in-sample behavior while leaving replacement cost weakly
identified. Transition matrices with wrong orientation can produce plausible
arrays and wrong economics.
