# Quick Start

```python
from econirl.estimators import MCEIRLNeural
import numpy as np

# transitions: (n_actions, n_states, n_states), row-stochastic
model = MCEIRLNeural(
    n_states=32,
    n_actions=3,
    discount=0.95,
    reward_type="state_action",   # R(s,a) -- default, more general
    anchor_action=0,              # fix action 0 reward column to zero
    reward_hidden_dim=32,
    reward_num_layers=2,
    max_epochs=200,
    lr=3e-3,
)
model.fit(
    data=df,
    state="state",
    action="action",
    id="agent_id",
    transitions=transitions,
)

print(model.reward_.shape)   # (32, 3)  -- anchored reward matrix R(s,a)
print(model.policy_.shape)   # (32, 3)  -- choice probabilities pi(a|s)
print(model.value_.shape)    # (32,)    -- value function V(s)
print(model.summary())
```

Fitted attributes follow the same behavioral convention as GLADIUS and AIRL:

| Attribute | Meaning |
| --- | --- |
| `reward_` | Neural reward map. Shape (n_states,) for `reward_type="state"`, (n_states, n_actions) for `"state_action"`. |
| `policy_` | Choice probabilities pi(a given s), shape (n_states, n_actions). |
| `value_` | Value function V(s), shape (n_states,). |
| `converged_` | Whether early stopping triggered (training loss plateau). |
| `n_epochs_` | Number of training epochs completed. |
| `feature_difference_` | L2 norm of the occupancy residual at the final epoch. |
| `occupancy_moment_residual_` | Max absolute occupancy residual. |
| `params_` | Projected parameters (dict), or None if no features were supplied. |
| `coef_` | Projected coefficients as a numpy array, or None. |
| `projection_r2_` | R-squared of the feature projection, or None. |

## State-Only Reward

```python
model = MCEIRLNeural(
    n_states=21, n_actions=3, discount=0.95,
    reward_type="state",   # R(s) broadcast to all actions
)
model.fit(data=df, state="state", action="action", id="agent_id",
          transitions=transitions)

print(model.reward_.shape)   # (21,)
```

Use `reward_type="state"` only when the data-generating process is known to
have no action-dependent reward component. For most environments
`reward_type="state_action"` is safer.

## Feature Projection

Pass `features=` to project the learned neural reward onto a linear basis
and extract an interpretable coefficient vector. The projection is a
least-squares regression of the reward matrix onto the feature matrix.

```python
model.fit(
    data=df, state="state", action="action", id="agent_id",
    transitions=transitions,
    features=np.asarray(env.feature_matrix),   # (n_states, n_features) or (n_states, n_actions, n_features)
)

print(model.params_)        # projected theta (meaningful only if projection is identified)
print(model.projection_r2_) # R-squared of the projection
```

Do not interpret projected parameters as structural estimates unless the
projection is well-identified. See the [Pre-Estimation Checks](pre_estimation.md)
page for the conditions.

## Advanced Estimator API

```python
# Predict choice probabilities at specific states
probs = model.predict_proba(states=np.array([0, 5, 10]))
# shape (3, n_actions)

# Access the full reward matrix
R = model.reward_matrix_   # always (n_states, n_actions)
```
