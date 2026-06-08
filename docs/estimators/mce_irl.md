# MCE-IRL

## Overview

MCE-IRL learns reward parameters by matching expert feature expectations under
the maximum causal entropy policy. In econirl, the validated path uses known
transitions and explicit reward features.

This is the finite-dimensional IRL baseline for structural reward-feature
recovery.

## When to Use

Use MCE-IRL when:

- demonstrations come from a discrete dynamic decision problem;
- transitions are known or supplied;
- reward features are explicit and finite-dimensional;
- you want reward, policy, value, Q, and counterfactual recovery;
- maximum causal entropy is the right behavioral regularization.

Avoid MCE-IRL when reward features are unknown, transitions are unavailable, or
you need a neural reward map. For the neural variant, use Deep MCE-IRL.

## Basic Usage

```python
import pandas as pd

from econirl.estimators import MCEIRL

data = pd.read_csv("dynamic_choices.csv")

model = MCEIRL(
    n_states=25,
    n_actions=3,
    discount=0.95,
    feature_matrix=reward_features,
)
model.fit(data, state="state", action="action", id="agent_id")

print(model.params_)
print(model.summary())
```

For multi-action structural recovery, pass an explicit reward specification to
`fit()` or provide `feature_matrix` at construction time.

## Validation Status

MCE-IRL passes the package known-truth gates on the validated
action-dependent reward-feature cells. The wrapper no longer treats
`feature_matrix=None` as a validated structural default for multi-action
models.

The validation DGPs are finite-state dynamic choice benchmarks with known
transitions, known action-specific reward features, and known policy, value, and
counterfactual truth.

## Further Reading

- Machine-readable artifact: [mce_irl_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mce_irl/mce_irl_results.json)
