# NNES

NNES estimates structural dynamic discrete choice models with a neural
value-function approximation inside an NPL-style policy iteration. It keeps the
reward finite-dimensional while using a flexible value approximation to avoid
exact dynamic programming in larger state spaces.

Use NNES when the reward is still structural and finite-dimensional, but the
state representation is large, encoded, smooth, or multi-dimensional enough
that repeated exact Bellman solves are no longer the best default.

## Source Papers

This page draws on {ref}`Nguyen (2025) <nguyen-2025>` for NNES and uses the
CCP/NPL logic of {ref}`Hotz and Miller (1993) <hotz-miller-1993>` and
{ref}`Aguirregabiria and Mira (2002) <aguirregabiria-mira-2002>`.

## Quick Decision

| Use NNES when | Prefer another estimator when |
| --- | --- |
| The value object is too large, smooth, or encoded for repeated exact DP. | The state-action space is small enough for exact NFXP or tabular CCP. |
| Rewards are parametric and structural. | The reward itself must be an unrestricted neural function. |
| Transitions are known or can be estimated before estimation. | Transition estimation is the main modeling problem. |
| You want counterfactuals from a recovered structural object. | You only need fitted choice probabilities. |
| Neural value approximation is the point of the exercise. | You need the cleanest exact likelihood reference. |
| Use `bellman="npl"` for the reported path. | `bellman="nfxp"` is a diagnostic variant and does not carry the NPL orthogonality claim. |

## Quick Start

```python
from econirl.datasets import load_rust_bus
from econirl import NNES

df = load_rust_bus()

model = NNES(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    bellman="npl",
    hidden_dim=32,
    num_layers=2,
    v_epochs=500,
    n_outer_iterations=3,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.policy_.shape)
print(model.v_network_.shape)
```

Output from the package smoke run:

```text
{'theta_c': 0.001034, 'RC': 3.073617}
(90, 2)
(90,)
```

Set `bellman="npl"` for the reported NNES path. Set `bellman="nfxp"` for the
neural soft-Bellman diagnostic variant, which does not carry the same
orthogonality claim.

## Evidence

NNES is reported on low-dimensional and high-dimensional action-dependent
synthetic data-generating processes. The high-dimensional cell is the primary study because
it uses encoded states and a richer reward-feature basis.

| Evidence | Current state |
| --- | --- |
| Question | Recover finite-dimensional structural reward and counterfactual behavior with a neural value approximation. |
| Study scope | Synthetic low- and high-dimensional structural DDC simulations. |
| Primary cell | `canonical_high_action`. |
| Machine-readable results file | [nnes_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nnes.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `NNES` with `utility="linear_cost"` and `bellman="npl"`. |

The caveat is the approximation boundary. The study reports recovery within
the numerical checks for the finite-dimensional structural reward
and the NNES value-approximation path. It is paper-consistent evidence for the
neural value route, not a claim that arbitrary neural reward models are
identified or that the EconIRL harness literally replicates Nguyen's Monte
Carlo design.

## NNES Guide

- [Context](nnes/context.md)
- [Quick Start](nnes/quick_start.md)
- [Under the Hood](nnes/under_the_hood.md)
- [Pre-Estimation Checks](nnes/pre_estimation.md)
- [Simulation Study](nnes/validation.md)
- [Counterfactuals](nnes/counterfactuals.md)
- [Rust Bus Engine Example](nnes/rust_bus.md)

```{toctree}
:hidden:

nnes/context
nnes/quick_start
nnes/under_the_hood
nnes/pre_estimation
nnes/validation
nnes/counterfactuals
nnes/rust_bus
```
