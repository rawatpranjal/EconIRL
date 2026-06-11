# Max Margin Planning Paper Context

Primary source: Ratliff, Bagnell, and Zinkevich (2006) for maximum margin
planning. Abbeel and Ng (2004) remains relevant for apprenticeship-learning
context. Broader paper routing lives in `../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Structured margin objective | Loss-augmented planning loss | Implemented in contrib. |
| Demonstrated policy | Empirical expert policy from panel | Used for loss and features. |
| Loss-augmented inference | Inner value/hybrid iteration | Adds task loss to reward. |
| Subgradient update | Learning-rate schedule and L2 penalty | Replaces QP constraint solve. |
| Task loss | `policy_kl` or `trajectory_hamming` | Configurable. |
| Release evidence | Missing | Needs known-truth validation JSON. |

## Derivation Checklist

1. Define expert and candidate policy features.
2. Define the task loss.
3. Write the structured hinge objective.
4. Derive the loss-augmented MDP reward.
5. Derive the subgradient update.
6. State release validation requirements.

## Release Gap

The estimator is tested but not release-certified. Keep it as contrib unless a
validation runner and result JSON are added.
