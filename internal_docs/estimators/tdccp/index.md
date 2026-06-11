# TD-CCP Internal Notes

TD-CCP estimates finite-dimensional structural reward parameters in dynamic
discrete choice models using conditional choice probabilities and
temporal-difference recursions. The parameter-estimation stage avoids modeling
the transition density directly. It uses observed current and next
state-action pairs to estimate recursive objects.

This page is the maintainer and AI reference for the paper-faithful
semigradient path, Algorithm 2 locally robust standard errors, cross-fitting,
and the boundary between transition-free parameter estimation and
transition-based validation.

## Source Boundary

The public RTD page should explain that TD-CCP estimates finite reward
parameters from panel transitions without a transition-density model. It should
not imply transition-free counterfactual evaluation or neural reward recovery.

Durable source context:

- Paper context: `papers.md`
- Vertical links: `links.md`
- Public RTD source: `../../../docs/estimators/tdccp.md`
- Validation runner: `../../../validation/estimators/tdccp/run.py`
- Validation results: `../../../validation/results/tdccp.json`

## Purpose

Use TD-CCP when choices are discrete, agents are forward-looking, panel data
contain current and next state-action information, and the hard part is
transition-density modeling rather than finite reward specification.

The package treats TD-CCP as a finite-theta estimator. It is not a raw neural
reward recovery method.

## Paper Logic

For feature vector `z(a,x)` and parameter vector `theta`, the paper writes the
CCP likelihood using:

```math
\pi(a\mid x;\theta,h,g)
=
\frac{\exp\{h(a,x)^\top\theta + g(a,x)\}}
{\sum_{a'}\exp\{h(a',x)^\top\theta + g(a',x)\}}.
```

The recursive terms solve:

```math
h(a,x)=z(a,x)+\beta E[h(a',x')\mid a,x],
```

and:

```math
g(a,x)=\beta E[e(a',x')+g(a',x')\mid a,x],
\quad
e(a,x)=\gamma_E-\log P(a\mid x).
```

The semigradient estimator replaces expectations with sample averages over
observed current and next state-action pairs. In linear form:

```math
\hat\omega
=
\left[
E_n\{\phi(a,x)(\phi(a,x)-\beta\phi(a',x'))^\top\}
\right]^{-1}
E_n\{\phi(a,x)z(a,x)\}.
```

There is an analogous equation for `g` using target `beta e(a',x')`.

## Implementation Mechanics

The implementation lives in `src/econirl/estimation/td_ccp.py`. The
semigradient path supports scalar polynomial bases, encoded-state bases, and
tabular bases. The known-truth validation uses encoded two-dimensional state
features with a logit first-stage CCP and Algorithm 2 locally robust standard
errors.

Known transitions are passed to the estimator result because the package
reports final policy and value objects. After `theta` is estimated, the
validation harness solves the dynamic program under the known synthetic
transition law to compare recovered policies, values, Q values, and
counterfactuals. That final evaluation does not feed back into PMLE parameter
estimation.

## Validation Design

The current result uses the hard case
`shapeshifter_encoded_state_locally_robust`.

| Quantity | Value |
| --- | ---: |
| States | 9 |
| State dimension | 2 |
| Actions | 3 |
| Reward parameters | 6 |
| Discount factor | 0.95 |
| Transition branching | 4 |
| Simulated observations | 120,000 |
| Basis type | encoded |
| Basis dimension | 2 |
| First-stage CCP | logit, degree 2 |
| Cross-fitting | true |
| Robust SE path | Algorithm 2 locally robust |

The validation notes state: paper-faithful hard case, encoded
two-dimensional state features, logit first-stage CCPs, Algorithm 2
cross-fitting, locally robust standard errors, and action-0 reward
normalization.

## Current Validation Results

The current result JSON records release status `Certified with caveat` for the
target "Temporal-difference CCP recovery of finite-dimensional structural
reward parameters with Algorithm 2 locally robust standard errors."

Run summary:

| Quantity | Value |
| --- | ---: |
| Converged | true |
| Optimizer iterations | 121 |
| Log likelihood | -99831.3015 |
| Estimation time | 10.71 seconds |
| Observations | 120,000 |
| Method | semigradient |
| First-stage CCP method | logit |
| CCP smoothing | 0.01 |
| Cross-fitting | true |
| Robust SE detail | `tdccp_algorithm2_locally_robust` |

Parameter recovery:

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_1_intercept` | 1.471089 | 1.487760 | 0.020127 | 0.016671 |
| `action_1_x0` | -1.174561 | -1.220465 | 0.025817 | -0.045904 |
| `action_1_x1` | 0.776064 | 0.792379 | 0.022374 | 0.016315 |
| `action_2_intercept` | -0.392696 | -0.468409 | 0.027026 | -0.075713 |
| `action_2_x0` | 0.290661 | 0.339360 | 0.034212 | 0.048699 |
| `action_2_x1` | -0.285572 | -0.215958 | 0.031091 | 0.069614 |

Recovery metrics:

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.050993 |
| Parameter relative RMSE | 0.059104 |
| Parameter cosine similarity | 0.998658 |
| Reward RMSE | 0.020629 |
| Reward normalized RMSE | 0.025307 |
| Value RMSE | 0.020109 |
| Value normalized RMSE | 0.000600 |
| Q RMSE | 0.023569 |
| Q normalized RMSE | 0.000733 |
| Policy total variation | 0.004658 |

Algorithm 2 and hard recovery gates:

| Gate | Threshold | Value | Status |
| --- | ---: | ---: | --- |
| Converged | true | true | pass |
| Algorithm 2 locally robust path | true | true | pass |
| Finite positive standard errors | true | true | pass |
| Zeta moment norm | <= 0.000100 | 7.91e-06 | pass |
| Covariance min eigenvalue | >= -1e-10 | 2.27e-05 | pass |
| Parameter cosine | >= 0.990000 | 0.998658 | pass |
| Parameter relative RMSE | <= 0.150000 | 0.059104 | pass |
| Reward normalized RMSE | <= 0.080000 | 0.025307 | pass |
| Policy total variation | <= 0.030000 | 0.004658 | pass |
| Value normalized RMSE | <= 0.100000 | 0.000600 | pass |
| Q normalized RMSE | <= 0.100000 | 0.000733 | pass |
| Type A regret | <= 0.050000 | 0.001883 | pass |
| Type B regret | <= 0.050000 | 0.001864 | pass |
| Type C regret | <= 0.050000 | 0.003200 | pass |

## Counterfactual Types

| Type | Intervention | Policy TV | Value RMSE | Regret |
| --- | --- | ---: | ---: | ---: |
| Type A | State-progress reward shift with baseline transitions | 0.004643 | 0.001885 | 0.001883 |
| Type B | Action-0 transition skip with baseline reward | 0.004840 | 0.001867 | 0.001864 |
| Type C | Disable action 1 with a large reward penalty | 0.006953 | 0.003208 | 0.003200 |

## Caveats

- Parameter estimation does not use a transition-density model, but validation
  and counterfactual evaluation use supplied transitions.
- The current result is a finite-theta hard case, not unrestricted neural
  reward recovery.
- The result is synthetic known-truth evidence, not a real-data replication.

## Debugging Order

If TD-CCP fails to recover known truth, debug in this order:

1. Confirm the semigradient path and Algorithm 2 locally robust SE path are
   active.
2. Check first-stage CCP quality and smoothing before changing reward
   features.
3. Check the basis type and basis dimension for the recursive terms.
4. Inspect zeta moment norm and covariance eigenvalues.
5. Confirm standard errors are finite and positive.
6. Separate parameter estimation from post-fit transition-based evaluation.
7. Do not treat neural or encoded state features as neural reward recovery.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/tdccp.py`
- Lower-level estimator: `../../../src/econirl/estimation/td_ccp.py`
- Validation runner: `../../../validation/estimators/tdccp/run.py`
- Validation results: `../../../validation/results/tdccp.json`
- Public RTD source: `../../../docs/estimators/tdccp.md`
