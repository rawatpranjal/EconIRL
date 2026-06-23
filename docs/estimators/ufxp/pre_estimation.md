# Pre-Estimation Checks

UFXP inverts empirical choice probabilities state by state, so its
pre-estimation risks are CCP's plus the usual linear-utility ones. Check these
before fitting:

| Check | Why it matters for UFXP |
| --- | --- |
| Feature rank | A rank-deficient design leaves a direction of theta undetermined; the closed-form solve flags it through `converged_`. |
| Feature condition number | Ill-conditioning inflates the variance of the closed-form solve. |
| State coverage | Conditions are scored only at visited states; unvisited states drop out, and very thin coverage leaves few usable conditions. |
| Action support per state | A state where one action is never taken makes the log-odds inversion degenerate at that state. |
| Transition row sums | Transition tensors must be row-stochastic in the `(n_actions, n_states, n_states)` orientation. |
| Reward normalization | One action's utility should anchor the normalization (the reference action). |

## Canonical Simulation Checks

Values from the canonical synthetic run (see [Simulation Study](validation.md)):

| Check | Value | Status |
| --- | --- | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.51 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Minimum action share | 0.325 | pass |

## Common Risk Patterns

Concentrated panels are the pattern to watch. When most trajectories visit a
narrow corridor of states (a gridworld walked corner to corner, a mileage
process that rarely runs high), the choice-probability estimates at the edges
are noisy or absent. The optimal weighting handles this gracefully. Thin
states are downweighted by their sample share rather than trusted. But no
weighting can recover information the data never carried. If coverage is thin
everywhere, prefer NFXP, which pools all observations through the likelihood.
