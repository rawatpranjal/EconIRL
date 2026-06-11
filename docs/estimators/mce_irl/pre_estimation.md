# Pre-Estimation Checks

Most MCE-IRL failures are visible before optimization starts. Check feature
variation, support, transition validity, and normalization before treating a
fit as reward recovery.

| Check | Why it matters |
| --- | --- |
| Feature rank | Rank below the number of reward features means some reward directions are unidentified. |
| Feature condition number | A high condition number signals unstable feature matching. |
| Action dependence | Multi-action reward recovery needs features that vary across actions. |
| Transition row sums | Occupancy measures require valid probability rows. |
| State coverage | Unobserved states weaken the occupancy comparison. |
| Action support | Rare actions make action-specific rewards weakly pinned down. |
| Reward anchor | MCE rewards need a normalization for level and scale. |

## Primary Simulation Checks

The primary `mce_low_high_reward` artifact records these checks. The
[simulation study page](validation.md) links the generator script, rendered table
source, and JSON artifact.

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 8 / 8 | pass |
| Feature condition number | 1.373 | pass |
| Action-dependent features | true | pass |
| Transition row error | 1.86e-8 | pass |
| Observed states | 25 / 25 | pass |
| State-action coverage | 1.000 | pass |
| Action shares | 0.334, 0.336, 0.330 | pass |
| Minimum positive CCP | 0.130 | pass |
| Reward anchor | true | pass |

## Common Risk Patterns

A state-only feature matrix can produce a plausible reward vector while still
failing to identify action-specific payoffs. Sparse demonstrations can match
the dominant actions while leaving rare-action rewards weak. Transition tensors
with the wrong orientation can have valid row sums and still produce the wrong
occupancy measure.
