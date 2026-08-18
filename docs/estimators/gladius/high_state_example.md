# High-State Example

Read this page as the concrete GLADIUS setup: high-dimensional state features,
low-dimensional reward projection, and an anchor action. The anchor is what
turns the learned Q objects into reward evidence.

The primary evaluation cell for GLADIUS uses 21 discrete states encoded into
64-dimensional feature vectors, a 4-parameter linear reward, and an anchor
action with known rewards. This setup follows the motivating design from
{ref}`Kang, Yoganarasimhan, and Jain (2025) <kang-2025>`: high-dimensional
state representation, low-dimensional reward structure, and an observable
anchor to pin the reward level.

```python
from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

config = GLADIUSConfig(
    anchor_action=2,
    anchor_rewards=anchor_rewards,   # shape (n_states,): known reward for action 2
    anchor_bellman_loss=True,
    anchor_bellman_mode="paper_minimax",
    network_mode="shared_trunk",
    q_hidden_dim=128,
    q_num_layers=3,
    v_hidden_dim=128,
    v_num_layers=3,
    max_epochs=500,
    batch_size=512,
    bellman_penalty_weight=1.0,
    patience=50,
    alternating_updates=True,
    verbose=True,
)

estimator = GLADIUSEstimator(config=config)
summary = estimator.estimate(panel, utility, problem, transitions)

print(summary.parameters)        # projected structural vector (4 parameters)
print(summary.policy[:5, :])     # imitation policy for first 5 states
```

## What the Networks Learn

The Q-network maps `(state_features, action_onehot)` - a vector of length
`state_dim + n_actions` - to a scalar action-value. The zeta network maps the
same input to a scalar approximation of $\mathbb{E}[V(s') \mid s, a]$.
During training, the zeta network is updated on even mini-batches against the
soft value of next states from the Q-network held fixed during that step; the Q-network is updated on
odd mini-batches against the NLL of observed choices and the anchor Bellman
residual.

After training, implied rewards are extracted as
$\hat{r}(s,a) = Q(s,a) - \beta\,\zeta(s,a)$ and projected onto the action-
difference feature matrix to recover the 4-parameter structural vector.

## Interpretation

The primary known-truth cell passes all 12 frozen gates: parameter cosine is
0.9773, projected reward NRMSE is 0.1311, raw Bellman reward NRMSE is 0.2989,
policy TV is 0.0187, Q NRMSE is 0.1237, and value NRMSE is 0.2194. The three
counterfactual regrets are all below 0.009. That cell uses the lower-level
`anchor_moment` structural diagnostic; the paper-reference shared-trunk minimax
path is qualified separately against Table 2.

## Replication Boundary

This page illustrates the high-state evaluation setup. The estimator's
recovery properties and the separate paper-replication boundary are established
in the [Validation and Qualification](validation.md) page. The results file
[`gladius.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius.json)
contains all numeric results.
