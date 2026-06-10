# TD-CCP

Temporal-difference CCP estimates structural dynamic discrete choice
parameters by learning the CCP recursive terms from observed successor tuples.
It keeps the CCP pseudo-likelihood target from the paper, but avoids
transition-density modeling while estimating the reward parameter.

Use TD-CCP when the reward is finite-dimensional and linear in known features,
the state features may be flexible, and transition-density modeling is the
bottleneck. Do not use it as evidence for unrestricted raw neural reward
recovery.

The certified RTD claim is the paper's Algorithm 2 locally robust finite-theta
path. Flexible encoders and neural AVI are supported as approximation tools,
but the release artifact certifies a semigradient structural-parameter case
with formula-level zeta, lambda, covariance, and Monte Carlo SE checks.

## Quick Decision

| Use TD-CCP when | Prefer another estimator when |
| --- | --- |
| Choices are discrete and forward-looking. | The reward is not finite-dimensional in known features. |
| Transition-density modeling is the bottleneck. | You need the exact tabular likelihood reference. |
| State features are flexible or high-dimensional. | Observed policy support is sparse. |
| You have current and next state-action tuples. | You need raw neural reward recovery from choices alone. |
| You want CCP-style structural parameters without repeated transition integration. | You only need a fitted behavioral policy. |

## Paper Alignment

| Paper object | EconIRL surface |
| --- | --- |
| Finite structural reward `u_theta(a, x) = z(a, x)' theta` | Action 0 is normalized; non-baseline actions have finite linear reward features. |
| Observed panel tuples `(a, x, a', x')` | Panels provide current choices, encoded states, next choices, and next states. |
| No parametric transition density for `theta` | Structural estimation uses observed successor tuples, not a fitted transition-density model. |
| Semigradient TD recursion for `h` and `g` | Certified path uses encoded basis functions and projected TD normal equations. |
| AVI with flexible learners | Available as a flexible approximation path, but not the certified release artifact. |
| Algorithm 2 locally robust PMLE | Cross-fitting, preliminary plug-in `theta`, lambda recursion, held-out zeta solve, and fold covariance are recorded. |
| Theorem 5 inference target | Artifact reports individual-clustered locally robust SEs and repeated-seed coverage diagnostics. |

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

For custom reward features or direct control over panel objects, utility
objects, and transition tensors, use `econirl.estimation.TDCCPEstimator`.

## What Is Certified

TD-CCP is certified for the paper-faithful finite-theta hard case. The reward
is linear in two encoded state coordinates interacted with non-baseline
actions, action 0 is normalized to zero, and the estimator uses logit CCPs,
Algorithm 2 cross-fitting, and locally robust standard errors. It recovers
structural parameters, rewards, policies, values, Q functions, and Type A/B/C
counterfactual decisions within the reported gates.

| Evidence | Current state |
| --- | --- |
| Release status | Certified with caveat. |
| Primary cell | `shapeshifter_encoded_state_locally_robust`. |
| Machine-readable artifact | [tdccp_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp_results.json). |
| Certified scope | Algorithm 2 finite-theta semigradient path with locally robust zeta moment and fold sandwich SEs. |
| Optimizer evidence | Preliminary folds 2/2 converged; final robust zeta folds 2/2 converged; max preliminary projected gradient is 2.00e-07. |
| Formula evidence | Artifact records zeta moment, lambda recursion residuals, finite covariance, and moment stationarity diagnostics. |
| SE coverage evidence | 25 repeated-seed replications with 0.900 overall 95% CI coverage; run 100 reps for a paper-final CPU-budget audit. |
| Diagnostic boundary | Raw neural reward has no finite true `theta` and is not a success claim. |
| Transition boundary | Estimation avoids transition-density modeling; evaluation can use supplied transitions. |

The `canonical_low_action` cell is retained as a sanity check. The
`canonical_high_action` and raw neural-reward cases are diagnostic records, not
release certification.

## TD-CCP Guide

```{toctree}
:maxdepth: 2

tdccp/context
tdccp/quick_start
tdccp/under_the_hood
tdccp/pre_estimation
tdccp/validation
tdccp/counterfactuals
tdccp/rust_bus
```
