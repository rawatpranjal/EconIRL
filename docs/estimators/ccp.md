# CCP

Conditional choice probability estimation is the fast tabular counterpart to
NFXP. It estimates first-stage action probabilities, uses Hotz-Miller inversion
to recover continuation values, and optionally iterates the
Aguirregabiria-Mira NPL mapping.

Use CCP when the state-action space is tabular, transitions are known or
estimated first, and empirical action support is strong. Use NFXP when you need
the most direct likelihood reference or when the first-stage policy is too
sparse for stable inversion.

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

CCP is reported on the low-dimensional action-dependent known-truth DGP. The
simulation cell has known rewards, transitions, policies, values, Q functions,
and Type A, Type B, and Type C counterfactual oracles. The machine-readable
artifact and generated primer results record the reported results.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Known-truth tabular cell with support conditions. |
| Primary cell | `canonical_low_action`. |
| Machine-readable artifact | [ccp_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/ccp/ccp_results.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the artifact. |
| Public example | Uses `CCP` with `utility="linear_cost"`. |

## CCP Guide

```{toctree}
:maxdepth: 2

ccp/context
ccp/quick_start
ccp/under_the_hood
ccp/pre_estimation
ccp/validation
ccp/counterfactuals
ccp/rust_bus
```
