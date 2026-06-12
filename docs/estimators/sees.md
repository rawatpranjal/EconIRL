# SEES

Sieve estimation for economic structural models estimates dynamic discrete
choice rewards while approximating the value function with a deterministic
basis. It keeps the structural likelihood target, but avoids solving a full
nested fixed point inside every likelihood evaluation.

Use SEES when the model is structural, transitions are known or estimated
first, and the value function can be represented well by a sieve. It is the
deterministic-basis counterpart to MPEC and the nearest stepping stone before
NNES.

## Source Papers

This page draws on {ref}`Luo and Sang (2024) <luo-sang-2024>` for sieve-based
structural estimation.

## Quick Decision

| Use SEES when | Prefer another estimator when |
| --- | --- |
| Choices are discrete and forward-looking. | The state-action space is small enough for exact NFXP. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling problem. |
| Rewards are finite-dimensional and parametric. | The reward itself needs a neural or nonparametric form. |
| A deterministic value basis is credible. | The value basis cannot approximate the Bellman solution. |
| You want a scalable structural check after MPEC. | You need a pure CCP or behavioral-cloning baseline. |

## Quick Start

```python
from econirl.datasets import load_rust_bus
from econirl import SEES

df = load_rust_bus()

model = SEES(
    n_states=90,
    discount=0.9999,
    utility="linear_cost",
    basis_type="fourier",
    basis_dim=8,
    penalty_weight=10.0,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

The dataframe wrapper is a package smoke path. The simulation-study evidence below
uses the lower-level estimator with explicit finite-state penalty weights.

Use `econirl.estimation.sees.SEESEstimator` when you need direct control over
the `Panel`, utility object, `DDCProblem`, transition tensor, basis choice, or
Bellman penalty.

## Evidence

SEES is reported on the high-dimensional action-dependent synthetic data-generating process.
The low-dimensional cell is retained as a sanity check, while the primary
cell uses encoded states and a richer reward-feature basis. Both cells have
known rewards, transitions, policies, values, Q functions, and Type A, Type B,
and Type C counterfactual oracles.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic encoded-state simulation. |
| Primary cell | `canonical_high_action`. |
| Machine-readable results file | [sees_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/sees.json). |
| Primary Bellman gate | Reported violation `3.08e-6`. |
| Primary recovery gates | Parameter, reward, policy, value, Q, and counterfactual gates are reported in the results file. |
| Public example | Uses `SEES` with `utility="linear_cost"` and the package default penalty; the simulation study uses `SEESEstimator` with explicit finite-state penalties. |

## SEES Guide

- [Context](sees/context.md)
- [Quick Start](sees/quick_start.md)
- [Under the Hood](sees/under_the_hood.md)
- [Pre-Estimation Checks](sees/pre_estimation.md)
- [Simulation Study](sees/validation.md)
- [Counterfactuals](sees/counterfactuals.md)
- [Rust Bus Engine Example](sees/rust_bus.md)

```{toctree}
:hidden:

sees/context
sees/quick_start
sees/under_the_hood
sees/pre_estimation
sees/validation
sees/counterfactuals
sees/rust_bus
```
