# CCP

Conditional choice probability estimation is the fast tabular counterpart to
NFXP. It estimates first-stage action probabilities, uses Hotz-Miller inversion
to recover continuation values, and optionally iterates the
Aguirregabiria-Mira NPL mapping.

Use CCP when the state-action space is tabular, transitions are known or
estimated first, and empirical action support is strong. Use NFXP when you need
the most direct likelihood reference or when the first-stage policy is too
sparse for stable inversion.

## Source Papers

This page draws on {ref}`Hotz and Miller (1993) <hotz-miller-1993>` for CCP
inversion and {ref}`Aguirregabiria and Mira (2002)
<aguirregabiria-mira-2002>` for nested pseudo-likelihood iteration.

## Quick Decision

| Use CCP when | Prefer another estimator when |
| --- | --- |
| States and actions are discrete. | Many states have weak or one-action support. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling problem. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| NFXP is too expensive for repeated comparison runs. | You need the reference nested fixed-point likelihood. |
| You want a Hotz-Miller or NPL structural estimate. | You only need a behavioral cloning baseline. |

## Minimal Fit

```python
from econirl.datasets import load_rust_bus
from econirl import CCP

df = load_rust_bus()

model = CCP(
    n_states=90,
    discount=0.9999,
    utility="linear_cost",
    num_policy_iterations=10,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

Set `num_policy_iterations=1` for the one-step Hotz-Miller estimator. Set a
larger value for fixed-iteration NPL.

## Evidence

CCP is reported on the low-dimensional action-dependent synthetic data-generating process. The
simulation cell has fully specified rewards, transitions, policies, values, Q functions,
and Type A, Type B, and Type C counterfactual oracles. The machine-readable
results file records the reported results.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic tabular simulation with support conditions. |
| Primary cell | `canonical_low_action`. |
| Machine-readable results file | [ccp.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ccp.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `CCP` with `utility="linear_cost"`. |

## CCP Guide

- [Context](ccp/context.md)
- [Quick Start](ccp/quick_start.md)
- [Under the Hood](ccp/under_the_hood.md)
- [Pre-Estimation Checks](ccp/pre_estimation.md)
- [Simulation Study](ccp/validation.md)
- [Counterfactuals](ccp/counterfactuals.md)
- [Rust Bus Engine Example](ccp/rust_bus.md)

```{toctree}
:hidden:

ccp/context
ccp/quick_start
ccp/under_the_hood
ccp/pre_estimation
ccp/validation
ccp/counterfactuals
ccp/rust_bus
```
