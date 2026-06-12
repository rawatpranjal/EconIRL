# UFXP

Unnested fixed point estimates the same structural parameters as NFXP without
ever solving a dynamic program inside the parameter search. The value-function
dependence of Bellman's first-order conditions is eliminated by dual fixed
points computed once, up front, so for a linear utility the whole estimator is
a single closed-form solve. With its optimal weighting it is as asymptotically
efficient as maximum likelihood.

Start here when you want structural estimates and standard errors at a small
fraction of NFXP's cost, or when the state space makes repeated Bellman solves
painful.

## Source Papers

This page draws on {ref}`Bray <bray-2019>` for the unnested fixed-point
construction and on {ref}`Oguz and Bray (2026) <oguz-bray-2026>` for the
optimally weighted form and its efficiency result.

## Quick Decision

| Use UFXP when | Prefer another estimator when |
| --- | --- |
| States and actions are discrete and transitions are known or pre-estimated. | Transition estimation is the main modeling challenge. |
| You want MLE-grade structural estimates much faster than NFXP. | You need an exact finite-sample MLE benchmark (use NFXP). |
| The likelihood search cost of NFXP or MPEC is binding. | The reward must be high-dimensional or neural (use the IRL family). |
| Most states are visited in the data. | State coverage is very thin everywhere (CCP-style inversion degrades). |
| Counterfactual policy analysis is central. | You only need fitted choice probabilities. |

## Minimal Fit

```python
from econirl.datasets import load_rust_bus
from econirl import UFXP

df = load_rust_bus()

model = UFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

## Evidence

UFXP is reported on the low-dimensional action-dependent synthetic
data-generating process. The simulation cell has known rewards, transitions,
policies, values, Q functions, and Type A, Type B, and Type C counterfactual
oracles. The machine-readable results file records the reported results. UFXP
also runs on every page of the [simulation studies](../simulation_studies/index.md)
alongside the rest of the structural family.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic tabular simulation. |
| Primary cell | `canonical_low_action`. |
| Machine-readable results file | [ufxp_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ufxp.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `UFXP` with `utility="linear_cost"`. |

## UFXP Guide

- [Context](ufxp/context.md)
- [Quick Start](ufxp/quick_start.md)
- [Under the Hood](ufxp/under_the_hood.md)
- [Pre-Estimation Checks](ufxp/pre_estimation.md)
- [Simulation Study](ufxp/validation.md)
- [Counterfactuals](ufxp/counterfactuals.md)
- [Bus Engine Example](ufxp/rust_bus.md)

```{toctree}
:hidden:

ufxp/context
ufxp/quick_start
ufxp/under_the_hood
ufxp/pre_estimation
ufxp/validation
ufxp/counterfactuals
ufxp/rust_bus
```
