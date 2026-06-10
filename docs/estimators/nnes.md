# NNES

NNES estimates structural dynamic discrete choice models with a neural
value-function approximation inside an NPL-style policy iteration. It keeps the
reward finite-dimensional while using a flexible value approximation to avoid
exact dynamic programming in larger state spaces.

Use NNES when the reward is still structural and finite-dimensional, but the
state representation is large enough that repeated exact Bellman solves are no
longer the best default.

## Quick Decision

| Use NNES when | Prefer another estimator when |
| --- | --- |
| Choices are discrete and forward-looking. | The state-action space is small enough for NFXP or CCP. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling problem. |
| Rewards are parametric and structural. | The reward itself must be a black-box neural function. |
| The value function needs a flexible approximation. | You need a fully exact tabular likelihood reference. |
| Counterfactual policy analysis is central. | You only need fitted choice probabilities. |

## Minimal Fit

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

Set `bellman="npl"` for the certified NNES path. Set `bellman="nfxp"` for the
neural soft-Bellman diagnostic variant, which does not carry the same
orthogonality claim.

## What Is Certified

NNES is reported on low-dimensional and high-dimensional action-dependent
known-truth DGPs. The high-dimensional cell is the primary validation because
it uses encoded states and a richer reward-feature basis.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Known-truth low- and high-dimensional structural DDC cells. |
| Primary cell | `canonical_high_action`. |
| Machine-readable artifact | [nnes_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes_results.json). |
| Release status | Certified with caveat. |
| Counterfactual gates | Type A, Type B, and Type C are reported in the artifact. |
| Public example | Uses `NNES` with `utility="linear_cost"` and `bellman="npl"`. |

The caveat is the approximation boundary. The validation certifies recovery
within the known-truth gates for the finite-dimensional structural reward and
the NNES value-approximation path; it is not a claim that arbitrary neural
reward models are identified.

## NNES Guide

```{toctree}
:maxdepth: 2

nnes/context
nnes/quick_start
nnes/under_the_hood
nnes/pre_estimation
nnes/validation
nnes/counterfactuals
nnes/rust_bus
```
