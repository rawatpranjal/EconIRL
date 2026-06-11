# Behavioral Cloning Paper Context

Behavioral cloning is a supervised imitation baseline rather than a target
paper method in the current structural-estimator release. Broader paper routing
lives in `../../papers/index.md`.

## Package Translation

| Concept | Package concept | Notes |
| --- | --- | --- |
| Supervised action prediction | Empirical CCP table | Counts actions by state. |
| Model-free imitation | No transition use | Transitions are ignored. |
| No reward recovery | Zero value function, policy parameters | Structural objects are absent. |
| Support issue | Unvisited state warning or smoothing fallback | Must be surfaced in comparisons. |
| Baseline role | Lower-bound policy imitation | Not a release estimator. |

## Documentation Rule

When behavioral cloning appears in docs, explain what it rules out: if a
supposedly structural estimator cannot outperform BC on behavior, its dynamic
structure is not paying rent. Do not let BC appear as evidence for rewards,
values, or counterfactuals.
