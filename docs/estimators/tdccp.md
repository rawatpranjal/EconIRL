# TD-CCP

## Overview

TD-CCP estimates dynamic discrete choice models by combining the CCP likelihood
with temporal-difference estimates of the recursive value terms. It learns the
recursive terms from observed state-action-next-state tuples, then optimizes the
CCP pseudo-likelihood.

Known transitions are only needed after estimation if you want final policy,
value, or counterfactual evaluation.

## When to Use

Use TD-CCP when:

- choices are discrete and forward-looking;
- rewards are finite-dimensional and linear in known features;
- state features may be flexible, including neural state encodings;
- transition-density modeling is the bottleneck;
- you have panel trajectories with current and next state-action pairs;
- you want structural parameters from a CCP-style estimator.

Avoid TD-CCP when observed policy support is sparse, reward features are weakly
identified, or you need raw nonparametric neural reward recovery from choices
alone.

## Basic Usage

```python
import pandas as pd

from econirl.estimators import TDCCP

data = pd.read_csv("zurcher_bus.csv")

model = TDCCP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    method="semigradient",
    basis_type="encoded",
)
model.fit(data, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

For custom reward features or lower-level control over the dynamic discrete
choice problem, use `econirl.estimation.TDCCPEstimator`.

## Evidence

The reported known-truth evidence covers the low-dimensional action-dependent
DGP.

The evidence also includes the paper-faithful hard flexible DGP with
stochastic flexible transitions, frozen neural state features, and a finite
linear structural reward with an action-0 normalization.

The high-dimensional action-dependent encoded-state stress DGP and the raw
neural flexible DGP are comparison artifacts. The public validation scope is
finite-parameter reward recovery.

Here, low-dimensional action-dependent DGP means the compact known-truth dynamic
choice benchmark. The hard flexible DGP tests finite-theta recovery with neural
state features, while the raw neural diagnostic has no finite true theta.

## Further Reading

- Machine-readable artifact: [tdccp_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp_results.json)
