# Quick Start

## Sklearn-Style API

`GLADIUS` accepts a pandas DataFrame directly.

```python
from econirl import GLADIUS
from econirl.core.reward_spec import RewardSpec

model = GLADIUS(
    n_actions=3,
    discount=0.95,
    anchor_action=2,
    q_hidden_dim=128,
    q_num_layers=3,
    ev_hidden_dim=128,
    ev_num_layers=3,
    max_epochs=300,
    patience=50,
    bellman_weight=1.0,
    verbose=False,
)
model.fit(
    df,
    state="state_bin",
    action="action",
    id="individual_id",
    features=reward_spec,      # RewardSpec or feature matrix; omit for policy only
)

print(model.params_)           # projected structural parameters (dict)
print(model.se_)               # projection standard errors
print(model.policy_)           # imitation policy, shape (n_states, n_actions)
print(model.value_)            # soft value function, shape (n_states,)
print(model.projection_r2_)   # R-squared of the action-difference projection
```

Fitted attributes:

| Attribute | Meaning |
| --- | --- |
| `params_` | Projected structural parameters by name. |
| `se_` | Standard errors from the projection OLS. |
| `coef_` | Parameter vector as a numpy array. |
| `policy_` | Imitation policy P(a given s), shape (n_states, n_actions). |
| `value_` | Soft value function V(s), shape (n_states,). |
| `projection_r2_` | R-squared of the action-difference regression. |
| `converged_` | Whether early stopping triggered before `max_epochs`. |
| `n_epochs_` | Number of training epochs completed. |

## Lower-Level API

`GLADIUSEstimator` accepts a `Panel` and an explicit utility, problem, and
transition tensor, mirroring the other lower-level estimators.

```python
from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

config = GLADIUSConfig(
    anchor_action=2,
    anchor_rewards=anchor_rewards,     # known reward for anchor action, shape (n_states,)
    anchor_bellman_loss=True,
    anchor_bellman_mode="anchor_moment",
    q_hidden_dim=128,
    q_num_layers=3,
    v_hidden_dim=128,
    v_num_layers=3,
    max_epochs=300,
    patience=50,
    bellman_penalty_weight=1.0,
)
estimator = GLADIUSEstimator(config=config)
summary = estimator.estimate(panel, utility, problem, transitions)

print(summary.parameters)      # projected structural vector
print(summary.policy)          # imitation policy
print(summary.value_function)  # soft value function
```

## Anchor Bellman Modes

```python
# Default: anchor moment pins Q to the continuation-value estimate
config = GLADIUSConfig(anchor_bellman_mode="anchor_moment")

# Paper's literal bi-conjugate minimax term (retained for method research)
config = GLADIUSConfig(anchor_bellman_mode="paper_minimax")
```

Use `anchor_moment` for normal estimation. The `paper_minimax` mode is kept for
diagnosing the original paper objective and is not recommended for production.
