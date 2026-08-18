# Quick Start

`GLADIUS` is the public paper-reference estimator. It defaults to the shared-
trunk minimax objective from Kang, Yoganarasimhan, and Jain (2025). Supply a
known reward for one anchor action in every declared state when reward levels,
bootstrap intervals, or structural counterfactuals are the target.

## Public API

```python
from econirl import GLADIUS

model = GLADIUS(
    n_actions=3,
    discount=0.95,
    anchor_action=2,
    anchor_rewards=known_anchor_rewards,  # length n_states
    q_hidden_dim=128,
    q_num_layers=3,
    ev_hidden_dim=128,
    ev_num_layers=3,
    max_epochs=500,
    patience=50,
    compute_se=True,
    n_bootstrap=99,
    seed=42,
    se_seed=43,
)
model.fit(
    df,
    state="state_bin",
    action="action",
    id="individual_id",
    features=reward_spec,
    transitions=transition_tensor,
)

print(model.summary())
print(model.diagnostics_)
print(model.params_)
print(model.conf_int())
```

GLADIUS does not use `transition_tensor` to fit Q, continuation value, reward,
or policy. It validates and stores the tensor for post-estimation planning.

Important fitted attributes:

| Attribute | Meaning |
| --- | --- |
| `q_` | Learned action-value table, shape `(n_states, n_actions)`. |
| `continuation_value_` | Learned conditional continuation table. |
| `reward_` | Implied anchored reward `q_ - discount * continuation_value_`. |
| `policy_` | Softmax policy implied by Q. |
| `params_`, `coef_` | Descriptive action-contrast projection onto supplied reward features. |
| `se_`, `pvalues_` | Descriptive projection diagnostics, not sampling uncertainty. |
| `bootstrap_` | Whole-trajectory reward and policy draws when `compute_se=True`. |
| `diagnostics_` | Coverage, rank, anchor, and optimization diagnostics. |
| `termination_reason_` | Why fitting stopped; exhausting `max_epochs` alone is not convergence. |

Use `conf_int()` for percentile intervals from the trajectory bootstrap. It
refuses to turn the descriptive projection standard errors into confidence
intervals.

## Counterfactual

```python
import numpy as np

reward_delta = np.zeros_like(model.reward_)
reward_delta[:, 1] = 0.25
result = model.counterfactual(reward_delta=reward_delta)

print(result.counterfactual_policy)
print(result.welfare_change)
```

A structural counterfactual requires both the fitted anchor and stored
planning transitions. An unanchored fit may still support policy prediction,
but GLADIUS refuses to label its reward levels or re-solved welfare structural.

## Lower-Level API

`GLADIUSEstimator` remains available for research protocols. Its configuration
surface exposes both objectives. Set both fields explicitly when reproducing
the public paper path:

```python
from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

config = GLADIUSConfig(
    anchor_action=2,
    anchor_rewards=known_anchor_rewards,
    anchor_bellman_mode="paper_minimax",
    network_mode="shared_trunk",
    gradient_clip_mode="value",
    output_bias_init=0.0,
    seed=42,
)
summary = GLADIUSEstimator(config=config).estimate(
    panel, utility, problem, transitions
)
```

The lower-level `anchor_moment` mode is a separate fitted-Q diagnostic used by
the known-truth structural stress test. It is not the public `GLADIUS` default.

See the
[applied notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/gladius/gladius_applied_workflow.ipynb)
for the complete fitted workflow.
