# Quick Start

The wrapper follows the sklearn-style pattern: build an estimator, call `fit`,
then read fitted attributes. RHIP requires a pre-built `Panel`, an
action-dependent feature array, and a transition tensor.

```python
import numpy as np

from econirl import RHIP
from econirl.core.types import Panel

model = RHIP(
    horizon=3,
    discount=0.95,
    scale=1.0,
    feature_names=["edge_cost", "amenity", "goal"],
)
model.fit(panel, features=features, transitions=transitions)

print(model.params_)
print(model.policy_.shape)
```

`features` must be a 3D array of shape `(n_states, n_actions, n_features)` and
`transitions` a 3D tensor of shape `(n_actions, n_states, n_states)`. The
fitted estimator exposes reward parameters, standard errors when requested,
the recovered reward matrix, the policy, and the value function.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated reward parameters. |
| `se_` | Standard errors for the reward parameters when `compute_se=True`. |
| `reward_matrix_` | Reward matrix by state and action, shape `(n_states, n_actions)`. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated value function by state. |
| `converged_` | Whether the gradient norm or occupancy moment crossed tolerance. |

## The Horizon Knob

The `horizon` parameter spans a family of behavioral models:

| `horizon` | Behavioral model |
| --- | --- |
| `float("inf")` or `None` | Max Causal Entropy IRL (stochastic, the MCE-IRL endpoint) |
| `0` | Max-Margin-Planning-like (deterministic, cheapest) |
| `1`, `3`, ... | Receding-horizon interpolation between the two ends |

The `H = inf` endpoint delegates to `MCEIRLEstimator` so the result is
identical to MCE-IRL by construction. Finite horizons run `H` soft-Bellman
backups on top of a deterministic continuation value.

```python
from econirl import RHIP

mce_end = RHIP(horizon=float("inf"))          # the MCE-IRL endpoint
mce_end.fit(panel, features=features, transitions=transitions)

det_end = RHIP(horizon=0)                     # the deterministic end
det_end.fit(panel, features=features, transitions=transitions)
```

## Lower-Level Interface

Use `econirl.estimators.rhip.RHIPEstimator` when you need direct control over
`Panel` objects, `DDCProblem`, `RHIPConfig`, or transition tensors as JAX
arrays. `RHIPEstimator` accepts a `RHIPConfig` that exposes Adam optimizer
settings, deterministic tail parameters, and bootstrap SE controls.
