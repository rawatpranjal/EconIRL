# MPEC

Mathematical programming with equilibrium constraints estimates the same
structural dynamic discrete choice likelihood as NFXP, but it treats the
Bellman fixed point as an explicit equality constraint. The optimizer solves
jointly for reward parameters and value functions.

Use MPEC as a constrained-optimization counterpart to NFXP when the state
space is moderate and you want to inspect Bellman constraint violations
directly. It targets the same tabular structural object as NFXP and CCP, but
uses a different numerical representation of the fixed point.

## Source Papers

This page draws on {ref}`Su and Judd (2012) <su-judd-2012>` for the MPEC
formulation and {ref}`Iskhakov et al. (2016) <iskhakov-2016>` for the
computational comparison with NFXP.

## Quick Decision

| Use MPEC when | Prefer another estimator when |
| --- | --- |
| States and actions are discrete. | The Bellman constraint is too large for a constrained optimizer. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling problem. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| You need a constrained-likelihood check on NFXP. | You need the fastest repeated comparison run. |
| Bellman constraint diagnostics are central. | You only need a behavioral cloning baseline. |

## Minimal Fit

```python
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation import simulate_panel

env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.99)
panel = simulate_panel(env, n_individuals=100, n_periods=50)
utility = LinearUtility.from_environment(env)

model = MPECEstimator(config=MPECConfig(solver="sqp"))
summary = model.estimate(
    panel=panel,
    utility=utility,
    problem=env.problem_spec,
    transitions=env.transition_matrices,
)

print(summary.parameters)
print(summary.metadata["final_constraint_violation"])
```

The MPEC public surface is currently the lower-level estimator API. It expects
an `econirl.core.Panel`, a utility object, a `DDCProblem`, and transition
matrices.

## Evidence

MPEC is reported on the low-dimensional action-dependent synthetic data-generating process. The
simulation cell has known rewards, transitions, policies, values, Q functions,
and Type A, Type B, and Type C counterfactual oracles. The machine-readable
results file and generated primer results record the reported results.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic constrained-likelihood simulation. |
| Primary cell | `canonical_low_action`. |
| Machine-readable results file | [mpec_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mpec/mpec_results.json). |
| Bellman constraint check | Reported final violation `7.72e-12`. |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `MPECEstimator` with the tabular DDC lower-level API. |

## MPEC Guide

```{toctree}
:maxdepth: 2

mpec/context
mpec/quick_start
mpec/under_the_hood
mpec/pre_estimation
mpec/validation
mpec/counterfactuals
mpec/rust_bus
```
