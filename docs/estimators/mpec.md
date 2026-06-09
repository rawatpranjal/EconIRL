# MPEC

Mathematical programming with equilibrium constraints estimates the same
structural dynamic discrete choice likelihood as NFXP, but it treats the
Bellman fixed point as an explicit equality constraint. The optimizer solves
jointly for reward parameters and value functions.

Use MPEC as a constrained-optimization counterpart to NFXP when the state space
is moderate and you want to inspect Bellman constraint violations directly.

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

## What Is Certified

MPEC is certified on the low-dimensional action-dependent known-truth DGP. The
validation cell has known rewards, transitions, policies, values, Q functions,
and Type A, Type B, and Type C counterfactual oracles. The machine-readable
artifact and generated primer results are the release source of truth.

| Evidence | Current state |
| --- | --- |
| Release status | Certified. |
| Primary cell | `canonical_low_action`. |
| Machine-readable artifact | [mpec_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mpec/mpec_results.json). |
| Bellman constraint gate | Passes with final violation `7.72e-12`. |
| Counterfactual gates | Type A, Type B, and Type C all pass. |
| Public example | Uses `MPEC` with the tabular DDC lower-level API. |

## MPEC Guide

```{toctree}
:maxdepth: 2

mpec/validation
```
