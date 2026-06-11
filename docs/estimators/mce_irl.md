# MCE-IRL

Maximum causal entropy IRL estimates reward parameters by matching the feature
counts in demonstrations to the feature counts implied by a soft dynamic
policy. In EconIRL, the tabular simulation uses known transitions and
action-dependent reward features supplied by the user.

Use this page for finite-dimensional tabular IRL. Use Deep MCE-IRL when the
target is a neural reward map instead of a fixed feature vector.

## Quick Decision

| Use MCE-IRL when | Prefer another estimator when |
| --- | --- |
| Demonstrations come from a discrete sequential decision problem. | You need likelihood-based structural standard errors. |
| Transitions are known or can be supplied. | Transition estimation is the main difficulty. |
| Reward features are supplied and action-dependent. | Reward features are unknown or purely neural. |
| The behavioral model is maximum causal entropy. | The target is deterministic optimal control without entropy regularization. |
| You want reward, policy, value, Q, and counterfactual recovery checks. | You only need fitted conditional choice probabilities. |

## Minimal Fit

```python
import numpy as np

from econirl.datasets import load_rust_bus
from econirl.estimators import MCEIRL

n_states = 90
n_actions = 2
features = np.zeros((n_states, n_actions, 2))
features[:, 0, 0] = -np.arange(n_states) / 100.0
features[:, 1, 1] = -1.0

df = load_rust_bus()

model = MCEIRL(
    n_states=n_states,
    n_actions=n_actions,
    discount=0.99,
    feature_matrix=features,
    feature_names=["keep_mileage_cost", "replace_cost"],
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.policy_.shape)
```

Multi-action MCE-IRL needs an explicit reward specification. Pass a
`RewardSpec` to `fit()` or provide `feature_matrix` at construction time. The
wrapper no longer treats `feature_matrix=None` as a structural default.

## Simulation Study

The simulation uses two action-dependent synthetic cells. The primary cell
has 25 states, 3 actions, 8 reward features, known transitions, known rewards,
known policies, known value and Q functions, and Type A, Type B, and Type C
counterfactual oracles.

| Item | Current state |
| --- | --- |
| Question | Recover reward and counterfactual behavior in a synthetic MCE data-generating process. |
| Primary cell | `mce_low_high_reward`. |
| Machine-readable results file | [mce_irl_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mce_irl/mce_irl_results.json). |
| Recovery outputs | Reward, policy, value, Q, feature-moment, and counterfactual metrics. |
| Public example | Uses `MCEIRL` with explicit action-dependent features. |

## MCE-IRL Guide

```{toctree}
:maxdepth: 2

mce_irl/context
mce_irl/quick_start
mce_irl/under_the_hood
mce_irl/pre_estimation
mce_irl/validation
mce_irl/counterfactuals
mce_irl/rust_bus
```
