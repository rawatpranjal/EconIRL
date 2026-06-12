# Pre-Estimation Checks

IQ-Learn has the same general data-quality checks as other estimators plus
coverage checks that are specific to its Q-based reward recovery.

| Check | Why it matters for IQ-Learn |
| --- | --- |
| Expert state coverage | The objective only scores expert (s, a) pairs; states not in the panel receive no direct Q signal. |
| Expert state-action coverage | Off-support state-action pairs get no gradient; their implied reward is extrapolated, not fitted. |
| Q parameterization and divergence | Tabular Q with simple divergence has no upper bound; chi-squared is required for bounded optimization. |
| Feature rank (linear head) | A rank-deficient feature matrix leaves directions of theta undetermined. |
| Feature condition number | Ill-conditioning inflates the variance of the linear Q solve. |
| Transition row sums | Transitions must be row-stochastic in the (n_actions, n_states, n_states) orientation for the inverse Bellman reward to be valid. |
| Discount and scale | Misspecified beta or sigma shift the implied reward by a constant factor. |

## Coverage Gates

IQ-Learn output is suitable for reward and counterfactual diagnostics only
when:

- `expert_state_coverage == 1.0` (every state in the MDP was visited),
- `expert_state_action_coverage >= 0.95` (at least 95 percent of
  state-action pairs were visited).

Below these thresholds the Q table and implied reward are valid only on
support; off-support values are extrapolation.

## Canonical Simulation Checks

Values from the primary synthetic cell run (see [Simulation Study](validation.md)):

| Check | Value | Status |
| --- | --- | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.51 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Minimum action share | 0.325 | pass |

## Common Risk Patterns

Sparse expert panels are the main risk. When the expert panel covers only a
subset of states, the Q table on unvisited states is unconstrained and the
inverse Bellman reward at those cells is unreliable. The linear Q head
mitigates this by constraining Q to propagate through features, but it
requires a well-specified feature matrix. Always check both coverage figures
from `summary.metadata` before interpreting the reward output.

The sparse-support guard in
[`sparse_support_guard.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/iq_learn/sparse_support_guard.py)
demonstrates that small policy TV and low counterfactual regret numbers are
not sufficient evidence when coverage is below the gates; support must pass
first.
