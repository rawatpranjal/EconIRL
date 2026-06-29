# Pre-Estimation Checks

Read this page before fitting f-IRL. The primary question is whether the expert
state marginal is well covered enough for behavioral matching to be meaningful.

f-IRL estimates a tabular reward by gradient ascent on a divergence objective,
so its failure modes differ from structural estimators. Check these before fitting:

| Check | Why it matters for f-IRL |
| --- | --- |
| State coverage | The expert state marginal is empirical; unobserved states carry zero expert density and will drive the reward toward negative infinity under forward KL. |
| Reward range after fitting | A near-zero reward range signals a flat-reward solution; stop and investigate before reading other metrics. |
| Action support per state | A state where only one action is ever taken may still be usable, but sparse action support degrades log-likelihood tracking. |
| Transition row sums | Transition tensors must be row-stochastic in the `(n_actions, n_states, n_states)` orientation. |
| Horizon length | The forward propagation runs for `horizon` steps; a horizon that is too short underestimates long-run state visitation. |
| Learning rate and clip bound | Too large a learning rate causes oscillation; too small a clip bound truncates the reward signal. |

## Canonical Simulation Checks

Values from the primary source-paper benchmark (see [Simulation Study](validation.md)):

| Check | Value | Status |
| --- | --- | --- |
| Observed states | 8 / 8 | pass |
| State-action coverage | 1.000 | pass |
| Feature rank | 3 / 3 | pass |
| Feature condition number | 18.471 | pass |
| Minimum action share | 0.320 | pass |
| Anchor valid | true | pass |
| Reward range (post-fit) | 1.292 | pass |

## Common Risk Patterns

**Flat reward.** Forward KL is the most common divergence to collapse to a flat
solution when the expert marginal has near-zero mass at some states, because the
gradient log-ratio becomes numerically unstable. Check `reward_range` first; if
it is near zero, try reducing the learning rate or switching to Jensen-Shannon.

**State coverage gaps.** If the expert panel does not visit all states, those
states have zero expert marginal. Forward KL assigns infinite cost to the model
visiting them; in practice this is clipped, but the resulting reward at
unvisited states is not trustworthy. If coverage is thin, prefer a feature-based
IRL method such as MaxEnt-IRL.

**Marginal scope mismatch.** Using `marginal_space="state_action"` with
`reward_scope="state"` is not supported and raises a configuration error.
Action-dependent marginal matching paired with an action-dependent reward
(`reward_scope="state_action"`) is supported but is a diagnostic exercise under
current evidence; see the Identification and Applicability sections of the
[main f-IRL page](../f_irl.md).
