# Pre-Estimation Checks

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Your Own Data](../../user_guide/your_own_data.md)

Check the transition model, normalization, and support before fitting.

| Check | Requirement |
| --- | --- |
| Transition orientation | `(n_actions, n_states, n_states)` |
| Transition rows | Finite, nonnegative, and row-stochastic |
| State indices | Integers in `[0, n_states)` |
| Action indices | Integers in `[0, n_actions)` |
| Action availability | If actions differ by state, supply `action_mask`; every state must retain at least one action |
| Demonstrations | When `action_mask` is supplied, no observed action may be unavailable |
| State-action reward | A valid anchored action |
| State-only reward | A valid reference state for the global additive normalization |
| Projection basis | Full column rank when `features=` is supplied |

State and action columns must use finite integer-coded values. The estimator
checks their coding and ranges. It checks transitions, action availability,
demonstrations, and anchors before training. Projection shape and rank are
checked after the reward fit. Detected violations raise a specific `ValueError`.

Thin support is still a scientific limitation when the input passes. The
fitted `diagnostics_` mapping records observed state coverage, observed
state-action coverage, the smallest observed action share, and the number of
states with only one available action under `action_mask`. A neural reward
surface outside the observed support is an extrapolation.

Projected coefficients describe one fitted reward surface. Interpret them only
when the projection rank and condition number are acceptable. They do not
become structural parameters because the projection has a high R-squared.
