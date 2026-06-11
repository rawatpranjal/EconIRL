# Deep MaxEnt IRL Internal Notes

Deep MaxEnt IRL is the contributed neural maximum-entropy estimator inspired by
Wulfmeier, Ondruska, and Posner. It parameterizes `R(s, a)` with a neural
network and optimizes feature-matching behavior through soft planning. It is
related to Deep MCE-IRL but should remain separate unless objective, occupancy
calculation, gauge handling, and validation artifacts are aligned.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Deep MCE comparison: `../deep_mce_irl/index.md`.
- Public RTD page: none currently.
- Contrib implementation: `../../../src/econirl/contrib/deep_maxent_irl.py`.
- Tests: `../../../tests/test_deep_maxent_irl.py`.
- Validation result: none currently.

## Algorithm

The implementation uses learned state and action embeddings and an MLP reward
network:

```text
r_eta(s, a) = MLP([embed_state(s), embed_action(a)]).
```

Training alternates:

1. compute the full reward matrix from the network;
2. solve soft value iteration;
3. compute expected state-action visitation under the induced policy;
4. compare expert and model visitation;
5. backpropagate the feature-matching loss through the reward network;
6. retain the best policy and reward matrix by log-likelihood.

The returned parameters are the flattened learned reward matrix for downstream
compatibility, not interpretable neural weights.

## Difference From Deep MCE-IRL

| Dimension | Deep MaxEnt IRL | Deep MCE-IRL |
| --- | --- | --- |
| Package state | Contrib only | Public estimator page exists |
| Validation JSON | None currently | Tracked validation artifact |
| Reward target | Neural reward matrix | Anchored neural reward map under MCE contract |
| Public claims | None | Limited validated claims |
| Merge status | Keep separate | Do not merge without matching evidence |

## Current Release State

Current evidence:

- contrib implementation exists;
- unit tests exist;
- no tracked validation JSON exists;
- no public RTD estimator page exists.

Before public exposure, add a validation runner with reward-map recovery,
policy TV, value/Q metrics, counterfactual checks, and a clear comparison
against Deep MCE-IRL.

## Debugging Order

1. Check state/action embedding dimensions.
2. Verify reward matrix gauge and scale.
3. Confirm soft value iteration convergence.
4. Inspect visitation mismatch before policy quality.
5. Compare learned reward matrix under a fixed normalization.
6. Do not compare raw neural weights as structural parameters.

## Implementation Paths

- Contrib estimator: `../../../src/econirl/contrib/deep_maxent_irl.py`.
- Tests: `../../../tests/test_deep_maxent_irl.py`.
- Deep MCE internal page: `../deep_mce_irl/index.md`.

## Public Documentation Boundary

Do not fold this page into Deep MCE-IRL or public RTD until matching validation
evidence exists.
