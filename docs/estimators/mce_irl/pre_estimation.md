# Pre-Estimation Checks

## Important Links

- [MCE-IRL overview](../mce_irl.md)
- [Quick start](quick_start.md)
- [Simulation evidence](validation.md)
- [Counterfactuals](counterfactuals.md)

Read this page before fitting MCE-IRL. The checks tell you whether the supplied
features, transitions, and normalization can support reward recovery from
demonstrations.

Most MCE-IRL failures are visible before optimization starts. Check feature
variation, support, transition validity, and normalization before treating a
fit as reward recovery.

| Check | Why it matters |
| --- | --- |
| Action-contrast rank | Rank below the number of reward features means some reward directions are unidentified from choices. |
| Feature condition number | A high condition number signals unstable feature matching. |
| Feature occupancy variation | Reward recovery needs feasible policies to induce different expected feature counts. State-only features can be informative through the dynamics. |
| Transition row sums | Occupancy measures require valid probability rows. |
| State coverage | Unobserved states weaken the occupancy comparison. |
| Action support | Rare actions make action-specific rewards weakly pinned down. |
| Reward normalization | Fix an additive reward reference. The entropy temperature, fixed at one by this estimator, determines the reward scale. |
| Task membership | Each trajectory must belong to one supplied task. |
| Deterministic successor | Every observed next state must match `next_state[s, a]`. |
| Horizon | No demonstration can exceed the task horizon. |
| Terminal features | Terminal states contribute zero reward features. |

For `DeterministicTransitions`, inspect `next_state.shape` and
`valid_action.shape`. Both must be `(S, A)`. Every nonterminal state needs at
least one legal action. A terminal state uses an absorbing action in the
compiled task.

For destination-specific route data, transform the observations into task
views over one fixed global transition system. `active_states` defines the
candidate path set. The reward features and parameter names remain shared.
Do not create a separate estimator or reward vector for each destination.

## Canonical Simulation Checks

Values from the primary `mce_low_high_reward` synthetic run (see
[Simulation Study](validation.md)):

| Check | Value | Status |
| --- | ---: | --- |
| Raw feature rank | 8 / 8 | pass |
| Raw feature condition number | 1.373 | pass |
| Action-contrast rank | 8 / 8 | pass |
| Action-contrast condition number | 1.766 | pass |
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

For action-dependent linear features, the wrapper checks the action-contrast
rank at fit time and warns when it is deficient. Support, conditioning, and
reward-normalization checks remain the user's responsibility.
