# Pre-Estimation Checks

Read this page before fitting Neural MCE-IRL. The main risks are anchor
normalization, state encoding, and thin occupancy support for the neural reward
map.

Deep MCE-IRL inherits all MCE-IRL data requirements and adds the neural
reward normalization problem. Check these before fitting:

| Check | Why it matters for Deep MCE-IRL |
| --- | --- |
| Transition orientation | Tensors must be row-stochastic in the `(n_actions, n_states, n_states)` orientation. A wrongly oriented tensor produces silent wrong results. |
| State encoding alignment | The state encoder output dimension must match the network input dimension. If no encoder is supplied, the default normalizes the state index to [0, 1]. |
| State coverage | The soft Bellman solve covers all states, but occupancy matching uses only observed state-action pairs. Very thin coverage starves the gradient in those regions. |
| Action support per state | A state where one action is never taken contributes no gradient for that action. The anchor action is exempt - it has no trainable reward - so its absence from a state is safe. |
| Anchor action validity | The `anchor_action` index must fall within `[0, n_actions)` and should correspond to an action with reasonable support. An action that is never taken is a poor anchor. |
| Anchor consistency | The anchor action or absorbing state must be set consistently between the training run and any downstream comparison. Changing the anchor invalidates reward-map comparisons. |
| Feature rank (projection only) | If `features=` is supplied, the feature matrix for the projected cell should be full rank with a low condition number. A rank-deficient projection cannot identify the projected parameters. |

## Primary Cell Checks

Values from the primary synthetic run (see [Simulation Study](validation.md)):

| Check | Value | Status |
| --- | --- | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 39.86 | pass |
| Observed states | 32 / 32 | pass |
| State-action coverage | 1.000 | pass |
| Minimum action share | 0.189 | pass |
| Anchor action valid | true | pass |

## Common Risk Patterns

**Anchor mismatch** is the most common mistake. If two runs use different anchor
actions, or one uses an anchor action and the other uses an absorbing state,
the reward matrices use different normalizations and cannot be compared
directly.
Set the same anchor before any cross-run comparison.

**Projection overconfidence** is the next most common mistake. The neural
reward is projected onto a linear feature basis as a convenience. When the
projection R-squared is low or the projected condition number is high (as in
the `deep_mce_neural_features` support cell, where the condition number reaches
478), the projected theta is not a reliable substitute for the reward matrix.
Report the reward matrix and behavioral metrics first.

**Thin tail coverage** compounds with early stopping. If demonstrations cluster
in a narrow part of the state space, the occupancy gradient is nearly zero at
unvisited states, the network receives no signal there, and early stopping
halts before any signal arrives. Use `max_epochs` generously when coverage is
thin.
