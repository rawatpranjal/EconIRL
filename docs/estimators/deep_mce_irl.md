# Deep MCE-IRL

Deep MCE-IRL extends maximum causal entropy IRL by replacing a linear reward
table with a neural reward map. It uses the same occupancy-matching objective
and soft Bellman planning as MCE-IRL, but the reward function is a small
feedforward network rather than a dot product with fixed features. The
validated object is the anchored reward matrix and the behavior it induces,
not the raw network weights.

Start here when the reward structure in your problem is nonlinear in the
available state encodings and you want the full MCE behavioral guarantee -
policy, value, Q function, and counterfactuals - without committing to a
hand-chosen linear feature set.

## Source Papers

This page draws on {ref}`Ziebart (2010) <ziebart-2010>` for the maximum causal
entropy IRL framework and on {ref}`Wulfmeier, Ondruska, and Posner (2015)
<wulfmeier-2015>` for neural maximum-entropy reward learning.

## Quick Decision

| Use Deep MCE-IRL when | Prefer another estimator when |
| --- | --- |
| Transitions are known or supplied. | Transitions must be estimated jointly. |
| The reward is nonlinear in the available state encodings. | A linear reward table is adequate (use MCE-IRL). |
| Behavioral fit - policy, value, Q - matters more than a structural parameter vector. | You need identified structural parameters (use the structural family). |
| You can impose an anchor action or absorbing state to normalize the reward. | The reward normalization cannot be fixed before estimation. |
| Counterfactual re-solving under the learned reward is the goal. | Policy-only imitation is enough (use BC). |

## Quick Start

```python
from econirl.estimators import MCEIRLNeural

model = MCEIRLNeural(
    n_states=32, n_actions=3, discount=0.95,
    reward_type="state_action", anchor_action=0,
)
model.fit(
    data=df, state="state", action="action", id="agent_id",
    transitions=transitions,
)

print(model.reward_.shape)   # (32, 3)
print(model.policy_.shape)   # (32, 3)
print(model.summary())
```

## Evidence

Deep MCE-IRL is reported on a synthetic cell with a fixed nonlinear neural
reward, known stochastic transitions, and an anchor action that normalizes
the reward. The cell has 32 states, 3 actions, 160,000 observations, and
known policy, value, Q, and counterfactual oracle objects, so every recovery
claim is checked against the truth. The machine-readable results file records
the reported results. Deep MCE-IRL also runs on the bus engine and gridworld
pages of the [simulation studies](../simulation_studies/index.md) alongside
the rest of the IRL roster.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic tabular simulation with fixed neural reward. |
| Primary cell | `deep_mce_neural_reward`. |
| Machine-readable results file | [deep_mce_irl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `MCEIRLNeural` with `reward_type="state_action"`. |

## Deep MCE-IRL Guide

- [Context](deep_mce_irl/context.md)
- [Quick Start](deep_mce_irl/quick_start.md)
- [Under the Hood](deep_mce_irl/under_the_hood.md)
- [Pre-Estimation Checks](deep_mce_irl/pre_estimation.md)
- [Simulation Study](deep_mce_irl/validation.md)
- [Counterfactuals](deep_mce_irl/counterfactuals.md)
- [Abstract MDP Example](deep_mce_irl/abstract_mdp.md)

```{toctree}
:hidden:

deep_mce_irl/context
deep_mce_irl/quick_start
deep_mce_irl/under_the_hood
deep_mce_irl/pre_estimation
deep_mce_irl/validation
deep_mce_irl/counterfactuals
deep_mce_irl/abstract_mdp
```
