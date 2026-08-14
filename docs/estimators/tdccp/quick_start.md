# Quick Start

## Important Links

- [TD-CCP overview](../tdccp.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Counterfactuals](counterfactuals.md)
- [Applied workflow notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/tdccp/tdccp_applied_workflow.ipynb)

The public estimator accepts a `RewardSpec`, panel data, and an optional
action-specific transition tensor. A supplied tensor must use `(actions,
states, next_states)` orientation. When it is omitted, TD-CCP estimates one
from the fitted panel. The stored tensor supports policy and value
calculations. The structural parameter stage learns from observed successor
tuples.

```python
import numpy as np

from econirl import TDCCP
from econirl.core.reward_spec import RewardSpec
from econirl.environments import ArrayMDP
from econirl.simulation.synthetic import simulate_panel

n_states = 8
transitions = np.zeros((2, n_states, n_states))
for state in range(n_states):
    transitions[0, state, state] = 0.6
    transitions[0, state, min(state + 1, n_states - 1)] += 0.4
transitions[1, :, 0] = 1.0

condition = np.linspace(0.0, 1.0, n_states)
features = np.zeros((n_states, 2, 2))
features[:, 0, 0] = -condition
features[:, 1, 1] = -1.0
names = ["condition_cost", "replacement_cost"]
reward = RewardSpec(features, names=names)
environment = ArrayMDP(
    transitions,
    features,
    theta=np.array([1.5, 2.0]),
    discount_factor=0.9,
    parameter_names=names,
    seed=21,
)
panel = simulate_panel(environment, n_individuals=80, n_periods=25, seed=21)

model = TDCCP(
    n_states=n_states,
    discount=0.9,
    utility=reward,
    seed=21,
    basis_dim=3,
    ccp_method="logit",
    ccp_poly_degree=2,
)
model.fit(panel, transitions=transitions)

print(np.round(model.coef_, 3).tolist())
print(model.policy_.shape)
print(np.all(np.isfinite(list(model.se_.values()))))
```

**Result**

```text
[1.449, 1.888]
(8, 2)
True
```

The fitted estimator exposes the shared structural attributes.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated reward parameters |
| `se_` | Standard errors |
| `policy_` | Action probabilities by state |
| `value_` | Integrated value function |
| `log_likelihood_` | CCP pseudo log likelihood |
| `ev_features_` | Recursive reward-feature components |
| `diagnostics_` | Data, identification, transition, and fit checks |
| `bootstrap_` | Bootstrap draws and intervals when requested |

## Method Choices

The default is the linear semigradient with Algorithm 2 inference.

| Approximation | Configuration |
| --- | --- |
| Polynomial semigradient | Set `method="semigradient"` and `basis_type="polynomial"`. |
| Encoded semigradient | Set `basis_type="encoded"` and pass an `(n_states, n_features)` array through `state_features`. |
| Neural AVI | Set `method="neural"`. The default uses 20 AVI iterations. |
| Gradient-boosting AVI | Set `method="neural"` and `avi_functional_class="gbm"`. |

The gradient-boosting path requires scikit-learn. The default neural path uses
the package's installed JAX stack.

Use encoded state features when a scalar state index hides meaningful
coordinates. Use neural or boosting AVI when the recursive terms are nonlinear
in those coordinates.

## Bootstrap

Set `se_method="bootstrap"` to resample whole individual trajectories. The
wrapper keeps its stored transition tensor fixed across draws. `bootstrap_`
records all successful draws, failures, standard errors, and percentile
intervals.

## Low-Level API

`econirl.estimation.td_ccp.TDCCPEstimator` exposes the paper-level configuration
object and accepts `Panel`, utility, and `DDCProblem` objects directly. The
public `TDCCP` wrapper is the recommended applied interface.
