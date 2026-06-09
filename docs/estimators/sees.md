# SEES

Sieve estimation for economic structural models estimates dynamic discrete
choice rewards while approximating the value function with a deterministic
basis. It keeps the structural likelihood target, but avoids solving a full
nested fixed point inside every likelihood evaluation.

Use SEES when the model is structural, transitions are known or estimated
first, and the value function can be represented well by a sieve. It is the
deterministic-basis counterpart to MPEC and the nearest stepping stone before
NNES.

## Quick Decision

| Use SEES when | Prefer another estimator when |
| --- | --- |
| Choices are discrete and forward-looking. | The state-action space is small enough for exact NFXP. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling problem. |
| Rewards are finite-dimensional and parametric. | The reward itself needs a neural or nonparametric form. |
| A deterministic value basis is credible. | The value basis cannot approximate the Bellman solution. |
| You want a scalable structural check after MPEC. | You need a pure CCP or behavioral-cloning baseline. |

## Minimal Fit

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
    penalty_weight=0.01,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

Output

```text
{'theta_c': -0.006116096477516279, 'RC': 3.1677777161163605}

Method:                    SEES (fourier, Luo & Sang 2024)
No. Observations:          9,410
Log-Likelihood:            -1,899.51
theta_c                 -0.0061     0.0034    -1.79    0.074
RC                       3.1678     0.1370    23.13    0.000
Prediction Accuracy:         94.9%
```

Use `econirl.estimation.sees.SEESEstimator` when you need direct control over
the `Panel`, utility object, `DDCProblem`, transition tensor, basis choice, or
Bellman penalty.

## What Is Certified

SEES is certified on the high-dimensional action-dependent known-truth DGP.
The low-dimensional cell is retained as a sanity check, while the primary
cell uses encoded states and a richer reward-feature basis. Both cells have
known rewards, transitions, policies, values, Q functions, and Type A, Type B,
and Type C counterfactual oracles.

| Evidence | Current state |
| --- | --- |
| Release status | Certified. |
| Primary cell | `canonical_high_action`. |
| Machine-readable artifact | [sees_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees_results.json). |
| Primary Bellman gate | Passes with violation `3.08e-6`. |
| Primary recovery gates | Parameter, reward, policy, value, Q, and counterfactual gates pass. |
| Public example | Uses `SEES` with `utility="linear_cost"`; validation uses `SEESEstimator`. |

## SEES Guide

```{toctree}
:maxdepth: 2

sees/context
sees/quick_start
sees/under_the_hood
sees/pre_estimation
sees/validation
sees/counterfactuals
sees/rust_bus
```
