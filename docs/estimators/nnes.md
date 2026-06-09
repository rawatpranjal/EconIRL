# NNES

## Overview

NNES estimates structural dynamic discrete choice models with a neural
value-function approximation inside an NPL-style policy iteration. It keeps the
reward finite-dimensional while using a flexible value approximation to avoid
exact dynamic programming in larger state spaces.

## When to Use

Use NNES when:

- choices are discrete and forward-looking;
- rewards are parametric and structural;
- transitions are known or estimated up front;
- the state space is large enough that exact value iteration is unattractive;
- you still need reward, policy, value, and counterfactual recovery.

Avoid NNES when the reward itself is nonparametric, or when the state-action
space is small enough that NFXP or CCP gives a simpler benchmark.

## Basic Usage

```python
import pandas as pd

from econirl.estimators import NNES

data = pd.read_csv("zurcher_bus.csv")

model = NNES(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    bellman="npl",
)
model.fit(data, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

Use `econirl.estimation.NNESEstimator` for the lower-level NPL implementation.

## Evidence

The reported known-truth evidence covers both the low-dimensional structural
DGP and the high-dimensional structural DGP. The high-dimensional cell
exercises encoded state features and a larger reward-feature basis.

The low-dimensional cell is the compact finite-state action-dependent benchmark.
The high-dimensional cell keeps known reward, policy, value, and counterfactual
truth while testing the neural value path on encoded states.

## Further Reading

- Machine-readable artifact: [nnes_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes_results.json)
