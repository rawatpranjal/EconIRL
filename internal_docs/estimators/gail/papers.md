# GAIL Paper Context

Primary source: Ho and Ermon (2016) for generative adversarial imitation
learning. Public citations live in `../../../docs/references.md`. Broader paper
routing lives in `../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Occupancy matching | State-action occupancy comparison | Main validation target. |
| Discriminator | Tabular or linear discriminator | Package does not require neural discriminator. |
| Policy optimizer | Soft value iteration | Tabular DDC adaptation of policy update. |
| Discriminator reward | Pseudo-reward for policy update | Not a primitive reward. |
| Behavior cloning comparison | Baseline policy frequencies | Runner compares both. |
| Reward transfer | Not certified | Use AIRL/AIRL-Het for that question. |

## Derivation Checklist

1. Define expert and model occupancy measures.
2. State the discriminator objective.
3. Connect the discriminator-derived reward to policy update.
4. Explain why the learned reward is a training signal rather than a structural
   primitive.
5. Compare occupancy divergence and policy match against behavioral cloning.
6. State the absence of a tracked release JSON until one is added.

## Release Gap

GAIL has unit/integration tests and a validation runner, but no tracked release
JSON in `validation/results/`. Do not add public RTD claims until the artifact
contract is completed.
