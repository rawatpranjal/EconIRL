# NNES

NNES estimates structural dynamic discrete choice models with a neural
value-function approximation inside an NPL-style policy iteration. It keeps the
reward finite-dimensional while using a flexible value approximation to avoid
exact dynamic programming in larger state spaces.

Use NNES when the reward is still structural and finite-dimensional, but the
state representation is large, encoded, smooth, or multi-dimensional enough
that repeated exact Bellman solves are no longer the best default.

## Questions NNES Answers

**What can NNES do that NFXP cannot do comfortably?** NFXP is the exact
tabular reference when the state space is manageable. NNES keeps the same
structural target but replaces the full tabular value object with a trained
neural approximation, which makes larger or encoded state representations a
natural validation target.

**What is a concrete example?** In the Nguyen paper, the Monte Carlo design is
a Rust-style replacement model with multiple bus modules. The true value
function is additively separable across modules, but NNES is not told that
shortcut. It trains a single value network and is compared with an oracle NFXP
benchmark that knows the separability.

**How does NNES work operationally?** It supplies or estimates transitions,
trains a value network, combines structural rewards with continuation values,
updates the likelihood through an NPL-style path, and reports structural
parameters, standard errors, policies, values, and recovery metadata.

## Core Contract

| Contract | Meaning |
| --- | --- |
| Validated path | Use `bellman="npl"`. This is the NNES path tied to the paper's NPL orthogonality argument and the known-truth artifact. |
| Diagnostic path | `bellman="nfxp"` trains a neural soft-Bellman approximation, but it does not carry the same zero-Jacobian or standard-error claim. |
| Transition model | NNES is model-based. Transitions must be known or estimated before policy evaluation. |
| Neural object | The neural network approximates the value or continuation object. The structural reward remains finite-dimensional. |
| Inference claim | The paper's efficient-inference claim relies on NPL orthogonality and a sufficiently accurate first-stage value or policy approximation. |

## Quick Decision

| Use NNES when | Prefer another estimator when |
| --- | --- |
| The value object is too large, smooth, or encoded for repeated exact DP. | The state-action space is small enough for exact NFXP or tabular CCP. |
| Rewards are parametric and structural. | The reward itself must be an unrestricted neural function. |
| Transitions are known or can be estimated before estimation. | Transition estimation is the main modeling problem. |
| You want counterfactuals from a recovered structural object. | You only need fitted choice probabilities. |
| Neural value approximation is the point of the exercise. | You need the cleanest exact likelihood reference. |

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

Output from the package smoke run:

```text
{'theta_c': 0.001034, 'RC': 3.073617}
(90, 2)
(90,)
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
the NNES value-approximation path. It is paper-consistent evidence for the
neural value route, not a claim that arbitrary neural reward models are
identified or that the EconIRL harness literally replicates Nguyen's Monte
Carlo design.

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
