# Deep MCE-IRL

## Overview

Deep MCE-IRL uses the maximum causal entropy occupancy-matching objective with
a neural reward map. The current simulation-study scope is nonlinear
reward-map recovery over supplied state encodings and known transitions.

The study evaluates the recovered reward matrix, not the raw neural network
weights.

## Source Papers

This page draws on {ref}`Ziebart (2010) <ziebart-2010>` for maximum causal
entropy IRL and {ref}`Wulfmeier, Ondruska, and Posner (2015)
<wulfmeier-2015>` for neural maximum-entropy reward learning.

## When to Use

Use Deep MCE-IRL when:

- transitions are known or supplied;
- demonstrations come from a discrete dynamic decision problem;
- the reward can be represented by a neural map over supplied encodings;
- you can impose a reward gauge, such as an anchor action or absorbing state;
- reward-map and counterfactual recovery matter more than finite-theta
  interpretation.

Avoid Deep MCE-IRL when you need identified finite structural parameters, or
when the input is raw spatial data that would require a convolutional reward
network outside the current simulation-study scope.

## Basic Usage

```python
import pandas as pd

from econirl.estimators import MCEIRLNeural

data = pd.read_csv("dynamic_choices.csv")

model = MCEIRLNeural(
    n_states=32,
    n_actions=3,
    discount=0.95,
    reward_type="state_action",
    state_encoder=state_encoder,
    state_dim=state_dim,
)
model.fit(
    data=data,
    state="state",
    action="action",
    id="agent_id",
    transitions=transitions,
    features=reward_features,
)

print(model.summary())
```

Pass `features=` only when you want an interpretable projection of the learned
reward map onto supplied reward features.

## Evidence

The reported simulation evidence covers the anchored neural reward-map
Shapeshifter cell. Finite-theta projections are reported only when the
projection basis is numerically identified.

Anchored neural reward-map DGP means a known-transition synthetic environment
with supplied state encodings, an action anchor for the reward gauge, and known
policy, value, reward-table, and counterfactual truth.

## Further Reading

- Machine-readable results file: [deep_mce_irl_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/deep_mce_irl/deep_mce_irl_results.json)
