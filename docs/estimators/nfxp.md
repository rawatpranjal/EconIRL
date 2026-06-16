# NFXP

Nested fixed point is the reference estimator for tabular structural dynamic
discrete choice. It estimates primitive reward parameters by solving the
agent's dynamic program inside each likelihood evaluation.

Start here for a small or moderate tabular dynamic discrete choice problem
where exact nested Bellman solves are feasible and a structural MLE benchmark
is the goal.

## Source Papers

This page draws on {ref}`Rust (1987) <rust-1987>` for the canonical dynamic
replacement model and nested fixed-point estimation, and on
{ref}`Iskhakov et al. (2016) <iskhakov-2016>` for the computational comparison
with constrained optimization.

## Quick Decision

| Use NFXP when | Prefer another estimator when |
| --- | --- |
| States and actions are discrete. | The state space is too large for repeated Bellman solves. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling challenge. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| You need a structural reference estimate. | You only need a fast imitation baseline. |
| Counterfactual policy analysis is central. | You only need fitted choice probabilities. |

## Quick Start

```python
from econirl.datasets import load_rust_bus
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

## Evidence

NFXP is reported on the low-dimensional action-dependent synthetic
data-generating process. The simulation cell has known rewards, transitions,
policies, values, Q functions, and Type A, Type B, and Type C counterfactual
oracles. The results file records the reported results. NFXP
also runs on every page of the [simulation studies](../simulation_studies/index.md)
alongside the rest of the structural family.

| Evidence | Current state |
| --- | --- |
| Scope | Synthetic tabular simulation. |
| Primary cell | `canonical_low_action`. |
| Results file | [nfxp_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nfxp.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `NFXP` with `utility="linear_cost"`. |

## NFXP Guide

- [Context](nfxp/context.md)
- [Quick Start](nfxp/quick_start.md)
- [Under the Hood](nfxp/under_the_hood.md)
- [Pre-Estimation Checks](nfxp/pre_estimation.md)
- [Simulation Study](nfxp/validation.md)
- [Counterfactuals](nfxp/counterfactuals.md)
- [Rust Bus Engine Example](nfxp/rust_bus.md)

```{toctree}
:hidden:

nfxp/context
nfxp/quick_start
nfxp/under_the_hood
nfxp/pre_estimation
nfxp/validation
nfxp/counterfactuals
nfxp/rust_bus
```
