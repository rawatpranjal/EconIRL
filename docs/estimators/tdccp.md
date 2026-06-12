# TD-CCP

TD-CCP estimates structural reward parameters in dynamic discrete choice models
without modeling the transition density. It learns the continuation terms that
enter the CCP likelihood directly from observed current and next state-action
pairs, using temporal-difference recursions. The flow utility is linear in known
features; the estimator recovers the finite-dimensional weight vector with
cross-fitted, locally robust standard errors.

Start here when the reward target is finite-dimensional, the panel contains
current and successor state-action information, and building a transition-density
model is the difficult part of the problem.

## Source Papers

This page draws on {ref}`Adusumilli and Eckardt (2025)
<adusumilli-eckardt-2025>` for the TD fixed-point construction and locally
robust inference and on {ref}`Hotz and Miller (1993) <hotz-miller-1993>` for
the CCP foundation.

## Quick Decision

| Use TD-CCP when | Prefer another estimator when |
| --- | --- |
| Choices are discrete and agents are forward-looking. | The state space is small and tabular likelihood methods are easy to run. |
| The panel contains current and next state-action pairs. | The panel does not record successor states or actions. |
| Transition-density modeling is the difficult part. | Transitions are known or easily estimated. |
| The reward is a finite linear function of known features. | The target is an unrestricted neural reward map (use the IRL family). |
| Valid inference with locally robust standard errors is required. | Observed action support is very sparse at key states. |

## Minimal Fit

```python
from econirl.datasets import load_rust_bus
from econirl import TDCCP

df = load_rust_bus()

model = TDCCP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    method="semigradient",
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

For custom reward features, panel objects, basis settings, cross-fitting, robust
standard errors, or supplied transition tensors, use
`econirl.estimation.TDCCPEstimator`.

## Evidence

TD-CCP is reported on the `shapeshifter_encoded_state_locally_robust` synthetic
cell: 81 states, 3 actions, two encoded state coordinates, and 6 reward
parameters. The data-generating process is fully specified, so recovered
parameters, policy, value function, Q function, and Type A, Type B, and Type C
counterfactual outcomes are all compared against oracle objects. The
machine-readable results file records the reported results. TD-CCP also
appears on the [bus engine](../simulation_studies/rust_bus.md) and
[abstract MDP sanity](../simulation_studies/abstract_mdp_1_sanity.md) pages of
the simulation studies.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic tabular simulation with encoded state features. |
| Primary cell | `shapeshifter_encoded_state_locally_robust`. |
| Machine-readable results file | [tdccp.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `TDCCP` with `utility="linear_cost"` on the bundled bus dataset. |

## TD-CCP Guide

- [Context](tdccp/context.md)
- [Quick Start](tdccp/quick_start.md)
- [Under the Hood](tdccp/under_the_hood.md)
- [Pre-Estimation Checks](tdccp/pre_estimation.md)
- [Simulation Study](tdccp/validation.md)
- [Counterfactuals](tdccp/counterfactuals.md)
- [Rust Bus Engine Example](tdccp/rust_bus.md)

```{toctree}
:hidden:

tdccp/context
tdccp/quick_start
tdccp/under_the_hood
tdccp/pre_estimation
tdccp/validation
tdccp/counterfactuals
tdccp/rust_bus
```
