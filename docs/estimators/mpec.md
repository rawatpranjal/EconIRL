# MPEC

## Overview

MPEC estimates the same structural dynamic discrete choice likelihood as NFXP,
but it treats the Bellman fixed point as an explicit equality constraint. The
optimizer solves jointly for reward parameters and value functions.

MPEC is useful as a constrained-optimization counterpart to NFXP when the
Bellman constraint dimension is moderate.

## When to Use

Use MPEC when:

- choices are discrete and forward-looking;
- the state-action space is tabular and moderate in size;
- transitions are known or estimated before fitting;
- you want to inspect Bellman constraint violations directly;
- a constrained optimizer is a useful check on NFXP or CCP estimates.

Avoid MPEC when the value-function constraint is too large for the optimizer or
when a faster CCP-style estimator is already sufficient.

## Basic Usage

```python
from econirl.estimation import MPEC, MPECConfig

config = MPECConfig(solver="sqp")
estimator = MPEC(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
```

The lower-level API expects an `econirl.core.Panel`, a utility object, a
`DDCProblem`, and transition matrices.

## Validation Status

MPEC passes the package known-truth gates on the low-dimensional
action-dependent DGP.

Here, low-dimensional action-dependent DGP means a compact finite-state dynamic
choice benchmark with action-specific rewards, known transitions, and known
reward, policy, value, and counterfactual truth.

## Further Reading

- Primer artifact: [mpec_results.tex](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mpec/mpec_results.tex)
