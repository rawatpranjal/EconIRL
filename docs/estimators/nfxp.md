# NFXP

Nested fixed point is the reference estimator for tabular structural dynamic
discrete choice. It estimates primitive reward parameters by solving the
agent's dynamic program inside each likelihood evaluation.

Start here for a small or moderate tabular dynamic discrete choice problem.
The page tree below replaces the old primer document as the public
documentation surface.

## Quick Decision

| Use NFXP when | Prefer another estimator when |
| --- | --- |
| States and actions are discrete. | The state space is too large for repeated Bellman solves. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling challenge. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| You need a structural reference estimate. | You only need a fast imitation baseline. |
| Counterfactual policy analysis is central. | You only need fitted choice probabilities. |

## Minimal Fit

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

NFXP is reported on the low-dimensional action-dependent synthetic data-generating process. The
simulation cell has known rewards, transitions, policies, values, Q functions,
and Type A, Type B, and Type C counterfactual oracles. The machine-readable
results file records the reported results.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic tabular simulation. |
| Primary cell | `canonical_low_action`. |
| Machine-readable results file | [nfxp_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nfxp/nfxp_results.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `NFXP` with `utility="linear_cost"`. |

## NFXP Guide

```{toctree}
:maxdepth: 2

nfxp/context
nfxp/quick_start
nfxp/under_the_hood
nfxp/pre_estimation
nfxp/validation
nfxp/counterfactuals
nfxp/rust_bus
```
