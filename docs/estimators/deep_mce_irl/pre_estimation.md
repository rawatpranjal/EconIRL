# Pre-Estimation Checks

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Wulfmeier-Shaped Study](wulfmeier_objectworld.md)
- [Your Own Data](../../user_guide/your_own_data.md)

Check the transition model, normalization, and support before fitting.

| Check | Requirement |
| --- | --- |
| Transition orientation | `(n_actions, n_states, n_states)` |
| Transition rows | Finite, nonnegative, and row-stochastic |
| State indices | Integers in `[0, n_states)` |
| Action indices | Integers in `[0, n_actions)` |
| Action availability | If actions differ by state, pass `action_mask`. Retain at least one action per state |
| Demonstrations | When `action_mask` is supplied, every observed action must be available |
| State-action reward | A valid anchored action |
| State-only reward | A valid reference state for the global additive normalization |
| Projection basis | Full column rank when `features=` is supplied |

State and action columns must use finite integer-coded values. The estimator
checks their coding and ranges. It checks transitions, action availability,
demonstrations, and anchors before training. Projection shape and rank are
checked after the reward fit. Detected violations raise a specific `ValueError`.

Thin support is still a scientific limitation when the input passes. The
fitted `diagnostics_` mapping has `data`, `identification`, `transitions`, and
`optimization` blocks. The data block records observed state coverage,
observed state-action coverage, the smallest marginal action share across the
full panel, and the number of states with only one available action under
`action_mask`. A neural reward surface outside the observed support is an
extrapolation.

Projected coefficients describe one fitted reward surface. Interpret them only
when the projection rank and condition number are acceptable. They do not
become structural parameters because the projection has a high R-squared.

## Worked Diagnostics

The fitted object from the [Quick Start](quick_start.md) reports:

| Diagnostic | Value |
| --- | ---: |
| Transition orientation | `(n_actions, n_states, n_states)` |
| Largest transition row error | 0.000 |
| Observed states | 3 / 3 |
| State coverage | 1.000 |
| State-action coverage | 1.000 |
| States with one available action | 0 |
| Smallest global action share | 0.398 |
| Normalization | `anchor_action=0` |

These checks describe the supplied panel and transition model. They do not
establish reward identification outside the observed support.

## Common Risk Patterns

Sparse state-action coverage leaves parts of a neural reward map weakly
supported. More training epochs do not add information to an unvisited state.

A tensor can have the expected dimensions while using the wrong axis order.
Confirm that each row of `transitions[action, state]` is a next-state
distribution.

An action mask must agree with every demonstrated action. If the anchored
reward will be interpreted, the anchor action should be available on the
relevant state support. The estimator validates the anchor index. It does not
require the anchor action to be available in every state.
