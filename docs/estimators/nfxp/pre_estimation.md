# Pre-Estimation Checks

## Important Links

- [NFXP Overview](../nfxp.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Bus Engine Example](rust_bus.md)

Read this page before fitting NFXP. These checks support structural
interpretation. A satisfactory result does not prove the model. A poor result
makes the reward estimate hard to interpret.

NFXP maximizes the conditional log likelihood over the reward parameters. Check
the reward design and numerical inputs before optimization starts. Run these
checks before treating a result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Raw rank below the number of free coefficients means the reward design is redundant. |
| Action-contrast rank | NFXP stops before optimization unless this rank equals the number of free coefficients. |
| Feature condition number | A high condition number signals unstable estimates. |
| Transition row sums | Each transition row must be a valid probability distribution. |
| Transition shape | NFXP accepts a keep kernel with shape $(S, S)$ or a full tensor with shape $(A, S, S)$. |
| State coverage | Thin coverage leaves behavior in some states weakly informed by the panel. |
| Action support | Rare actions leave their payoff weakly identified. |
| Reward normalization | The shock scale is fixed at 1.0. Omit coefficients that do not change action contrasts. |

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

NFXP currently requires full action-contrast rank. A feature copied identically
across actions fails this pre-fit check. Confirm that the action-contrast rank
equals the number of free coefficients before fitting.

Data with almost no replacement choices can fit in-sample behavior while
leaving the replacement cost weakly identified. Transition matrices with the
wrong orientation produce plausible arrays but wrong economics. Thin state
coverage leaves some regions weakly informed by observed choices. Evaluating
one panel likelihood does not create support in unobserved states.
