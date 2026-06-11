# GCL Paper Context

Primary source: Finn, Levine, and Abbeel (2016) for guided cost learning.
Broader paper routing lives in `../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Cost function | `NeuralCostFunction` | Reward is negative cost. |
| Trajectory samples | Package `Trajectory` objects | Used for demonstrations and policy rollouts. |
| Importance sampling | Trajectory weight computation | Sensitive to policy mismatch. |
| Partition-function approximation | Sampled trajectory weighting | No exact partition function in release artifact. |
| Policy update | Soft value iteration | Tabular package adaptation. |
| Neural reward/cost | Cost matrix and reward matrix | Not an interpretable structural theta. |

## Derivation Checklist

1. Define cost and reward sign convention.
2. Define demonstration and sampled trajectory distributions.
3. Derive or explain the importance weights.
4. State the cost-network update.
5. State the policy update under negative cost.
6. Explain why GCL is not the same as Deep MCE-IRL.
7. Tie any public claims to a tracked validation JSON.

## Release Gap

The current tracked result is a small gridworld comparison against MCE-IRL. It
does not include the release gate bundle used for structural estimators. Keep
GCL in contrib unless that artifact is expanded.
