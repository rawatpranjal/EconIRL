# Pre-Estimation Checks

Most RHIP failures are visible before optimization starts. Check feature
variation, support, transition validity, and normalization before treating a
fit as reward recovery.

| Check | Why it matters |
| --- | --- |
| Feature rank | Rank below the number of reward features means some reward directions are unidentified. |
| Feature condition number | A high condition number signals unstable feature matching. |
| Action dependence | Multi-action reward recovery needs features that vary across actions. |
| Transition row sums | Occupancy measures require valid probability rows. |
| Transition orientation | The tensor must be `(n_actions, n_states, n_states)`. Wrong orientation can have valid row sums and still produce the wrong occupancy measure. |
| State coverage | Unobserved states weaken the occupancy comparison. |
| Action support | Rare actions make action-specific rewards weakly pinned down. |
| Reward anchor | RHIP rewards need a normalization for level and scale. |
| Planning horizon | The horizon `H` sets the behavioral model the estimator matches. Choose it for the demonstrator-noise regime. It is a planner hyperparameter, not an identifying assumption. |

## Canonical Simulation Checks

Values from the primary high-dimensional route-choice synthetic run
(see [Simulation Study](validation.md)), 150-node random geometric road
network, features `[edge_cost, amenity, goal]`, true parameters `[1.0, 0.5, 1.0]`:

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 3 / 3 | pass |
| Feature condition number | 2.46e+01 | pass |
| Action-contrast rank | 3 / 3 | pass |
| Transition row error | < 1e-10 | pass |
| Observed states | 150 / 150 | pass |
| Transition orientation | (4, 150, 150) | pass |
| Action support | 4 actions, 400 agents | pass |
| Reward anchor | true | pass |

## Common Risk Patterns

A state-only feature matrix can produce a plausible reward vector while still
failing to identify action-specific payoffs. Sparse demonstrations can match
the dominant actions while leaving rare-action rewards weak. Transition tensors
with the wrong orientation can have valid row sums and still produce the wrong
occupancy measure. Choosing a horizon much higher than the demonstrator's
actual lookahead does not fail silently, but the extra soft backups add cost
without improving recovery.
