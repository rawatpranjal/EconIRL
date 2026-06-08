# CCP

## Overview

CCP estimates structural dynamic discrete choice models with conditional choice
probabilities. It avoids repeated NFXP Bellman solves by using policy inversion
and, in NPL mode, iterating the pseudo-likelihood mapping.

This is the natural starting point when the state-action space is tabular and
empirical choice probabilities are well supported.

## When to Use

Use CCP when:

- choices are discrete and forward-looking;
- states and actions are tabular;
- observed action support is strong enough to estimate CCPs;
- transitions are known or can be estimated in a first stage;
- you want a structural estimator that is cheaper than NFXP.

Avoid CCP when important states or actions have sparse support, or when the
Hotz-Miller inversion is numerically fragile for the data at hand.

## Basic Usage

```python
import pandas as pd

from econirl.estimators import CCP

data = pd.read_csv("zurcher_bus.csv")

model = CCP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    num_policy_iterations=10,
)
model.fit(data, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

Use `num_policy_iterations=1` for a one-step Hotz-Miller style run. Use more
iterations for NPL.

## Validation Status

CCP passes the package known-truth gates on the low-dimensional
action-dependent DGP.

Here, low-dimensional action-dependent DGP means a compact finite-state dynamic
choice benchmark with action-specific rewards, known transitions, and known
reward, policy, value, and counterfactual truth.

## Further Reading

- Machine-readable artifact: [ccp_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/ccp/ccp_results.json)
