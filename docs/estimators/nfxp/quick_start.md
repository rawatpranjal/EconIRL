# Quick Start

This is the smallest NFXP path: fit the reference structural estimator, then
inspect parameters, standard errors, policy, and value objects. It is a usage
example, not a substitute for the pre-estimation checks.

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)   # {"operating_cost": ..., "replacement_cost": ...}
print(model.se_)
print(model.summary())
```

Fitted attributes follow the same convention as CCP and UFXP:

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for the structural parameters. |
| `coef_` | Coefficients as a numpy array. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated value function by state. |
| `transition_tensor_` | Transition probabilities used for each action. |
| `transition_source_` | Whether those probabilities came from the panel or were supplied. |
| `log_likelihood_` | Maximized conditional choice log likelihood. |
| `converged_` | Whether the outer optimizer reported convergence. |

## Custom Reward Features

`rust_bus_reward_spec(n_states)` builds the Rust bus features for you. To use
your own features, build a `RewardSpec` directly and pass it to `fit`. The
feature array has shape `(n_states, n_actions, n_features)`, and `reward=`
overrides the constructor default.

```python
import numpy as np
from econirl import NFXP, RewardSpec
from econirl.datasets import load_rust_bus

df = load_rust_bus()
n_states, n_actions = 90, 2
mileage = np.arange(n_states) / n_states

features = np.zeros((n_states, n_actions, 2))
features[:, 0, 0] = -mileage   # operating cost on "keep", rising with mileage
features[:, 1, 1] = -1.0       # flat replacement cost on "replace"

spec = RewardSpec(features, names=["operating_cost", "replacement_cost"])

model = NFXP(n_states=n_states, discount=0.9999)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id", reward=spec)

print(model.params_)   # {"operating_cost": ..., "replacement_cost": ...}
```

A `RewardSpec` whose state or action dimension does not match `n_states` /
`n_actions` raises a clear `ValueError` at `fit` time, not a broadcasting error
later.

## Counterfactual Example

```python
cf = model.counterfactual(replacement_cost=4.0)

print(cf.params)
print(cf.policy[50, 1])
print(cf.summary())
```

This solves the fitted model again with a higher replacement cost and returns
the new value function and policy. For an environment counterfactual, pass a
complete action-specific tensor instead:

```python
cf_transition = model.counterfactual(transitions=alternative_transitions)
print(cf_transition.summary())
```

`summary(alpha=0.10)` reports 90 percent confidence intervals. The report also
shows whether estimation converged, how many iterations it took, elapsed time,
and where the transition probabilities came from.

## Advanced API

Use `econirl.estimation.nfxp.NFXPEstimator` when you need direct control over
the model inputs and optimizer:

```python
from econirl.estimation import NFXPEstimator

result = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10).estimate(
    panel, utility, problem, transitions
)
```
