# MPEC Internal Notes

MPEC estimates the same structural dynamic discrete choice likelihood as NFXP,
but changes the numerical problem. NFXP eliminates the value function by
solving the Bellman fixed point inside every likelihood evaluation. MPEC keeps
the value function as an optimizer variable and imposes the Bellman fixed
point as an equality constraint.

This page is the maintainer and AI reference for the constrained optimization
setup, Bellman residual checks, standard-error path, high-discount caveats,
and validation evidence.

## Source Boundary

The public RTD page should present MPEC as a constrained-optimization
counterpart to NFXP and show how to inspect Bellman constraint diagnostics. It
should not carry the full constraint derivation or solver-debugging matrix.

Durable source context:

- Paper context: `papers.md`
- Vertical links: `links.md`
- Public RTD source: `../../../docs/estimators/mpec.md`
- Validation runner: `../../../validation/estimators/mpec/run.py`
- High-beta guard: `../../../validation/estimators/mpec/high_beta_smoke.py`
- Validation results: `../../../validation/results/mpec.json`

## Purpose

Use MPEC when the state space is tabular, transitions are known or first-stage
estimated, the Bellman constraint dimension is moderate, and a direct
constrained-optimization formulation is useful. It is also a useful audit path
for NFXP because both estimators target the same MLE through different
numerical geometry.

Do not use MPEC as the main recovery tool for large state spaces, nearly
singular Bellman Jacobians, or high-discount problems unless the constraint
solver has been stress-tested. The critical failure mode is not an inner
Bellman loop failing; it is an optimizer reporting success while constraint
violation, stationarity, or standard errors are not credible.

## Model

The observed panel contains state-action-next-state trajectories:

```text
(s_it, a_it, s_i,t+1), i = 1,...,N, t = 1,...,T.
```

Flow payoff is linear in known features:

```math
u_\theta(s,a) = \phi(s,a)^\top \theta.
```

The transition kernel is:

```math
P_a(s,s') = \Pr(s_{t+1}=s' \mid s_t=s, a_t=a).
```

Given `V`, the choice-specific value is:

```math
Q_\theta(s,a;V)
=
u_\theta(s,a)
+ \beta\sum_{s'}P_a(s,s')V(s').
```

The logit policy implied by `(theta, V)` is:

```math
\pi_{\theta,V}(a \mid s)
=
\frac{\exp(Q_\theta(s,a;V)/\sigma)}
{\sum_b \exp(Q_\theta(s,b;V)/\sigma)}.
```

The soft Bellman operator is:

```math
T_\theta V(s)
=
\sigma \log \sum_a
\exp\left(
\frac{u_\theta(s,a)+\beta\sum_{s'}P_a(s,s')V(s')}{\sigma}
\right).
```

## Constrained Estimator

NFXP solves:

```math
\hat\theta_{\mathrm{NFXP}}
=
\arg\max_\theta
\sum_{i,t}\log\pi_\theta(a_{it}\mid s_{it}),
\quad
\text{where } V_\theta = T_\theta V_\theta.
```

MPEC solves the constrained problem:

```math
(\hat\theta,\hat V)
=
\arg\max_{\theta,V}
\sum_{i,t}\log\pi_{\theta,V}(a_{it}\mid s_{it})
\quad
\text{subject to}\quad
c(\theta,V)=V-T_\theta V=0.
```

The implementation uses SLSQP with JAX-supplied objective gradients and
constraint Jacobians. It initializes `V` at the Bellman fixed point of the
initial `theta`, giving SLSQP a feasible or near-feasible start. Each SQP step
works with the joint variable `x = (theta, V)` and the equality constraint
`c(x)` with one row per state. There is no nested Bellman solve inside the
likelihood objective.

At convergence, standard errors use the same implicit score logic as NFXP:

```math
(I-\beta P_\pi)\frac{\partial V}{\partial\theta}
=
\sum_a \pi(a\mid s)\phi(s,a).
```

The known-truth validation uses robust standard errors from per-observation
scores and gates on finite standard errors.

## Identification And Numerical Requirements

MPEC and NFXP share the same structural identification requirements, with an
additional constrained-optimization requirement:

- Reward rank. The action-dependent feature matrix must have full column rank.
- Action support. Each action must appear in the finite panel.
- State coverage. The canonical tabular validation observes every state.
- Normalization. The logit scale is fixed and the synthetic DGP uses an exit
  action and absorbing state as a zero-reward anchor.
- Constraint qualification. The Bellman constraint Jacobian must be
  numerically well behaved at the solution. The run reports final Bellman
  residual and gates on it.

## Known-Truth Validation Design

The current MPEC validation is synthetic known-truth evidence, not a real-data
replication. It uses the same low-dimensional action-dependent structural
benchmark as NFXP and CCP.

Current validation cell:

| Quantity | Value |
| --- | ---: |
| Cell ID | `canonical_low_action` |
| Regular states | 20 |
| Absorbing states | 1 |
| Total states | 21 |
| Actions | 3 |
| Exit action | 2 |
| Discount factor | 0.95 |
| Shock scale | 1.0 |
| Reward mode | Action-dependent |
| Reward dimension | Low |
| Simulated individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |

Feature map:

```math
\phi(s,a)
=
\begin{cases}
(1, x_s, 0, 0) & a = 0,\\
(0, 0, 1, x_s) & a = 1,\\
(0, 0, 0, 0) & a = 2.
\end{cases}
```

True reward parameters:

| Parameter | Truth |
| --- | ---: |
| `action_0_intercept` | 0.10 |
| `action_0_progress` | 0.50 |
| `action_1_intercept` | 0.00 |
| `action_1_progress` | -0.20 |

## Pre-Estimation Diagnostics

For MPEC, the estimator is structural, so rank and normalization checks are
not optional.

Current diagnostics from `validation/results/mpec.json`:

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.512 | pass |
| Transition row error | 2.42e-08 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Action shares | 0.345, 0.330, 0.325 | pass |
| Minimum action share | 0.325 | pass |
| Single-action states | 0 | pass |
| Exit/absorbing anchor | true | pass |

## Current Validation Results

The current result JSON records release status `Certified` for the target
"Su-Judd constrained maximum likelihood for tabular DDC." In internal docs,
that means the committed synthetic result passes the structural recovery gates
and Bellman constraint gate for this DGP.

Run summary:

| Quantity | Value |
| --- | ---: |
| Converged | true |
| SQP iterations | 19 |
| Log likelihood | -174875.7719 |
| Estimation time | 1.92 seconds |
| Observations | 160,000 |
| Solver method | slsqp |
| Solver message | Optimization terminated successfully |
| Final constraint violation | 7.72e-12 |

Parameter recovery:

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083894 | 0.029336 | -0.016106 |
| `action_0_progress` | 0.500000 | 0.528522 | 0.035890 | 0.028522 |
| `action_1_intercept` | 0.000000 | -0.014461 | 0.036733 | -0.014461 |
| `action_1_progress` | -0.200000 | -0.200511 | 0.052502 | -0.000511 |

Recovery metrics:

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017905 |
| Parameter relative RMSE | 0.065378 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019445 |
| Q RMSE | 0.022437 |
| Policy KL | 9.21e-05 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018905 |

Hard recovery gates:

| Gate | Threshold | Value | Status |
| --- | ---: | ---: | --- |
| Converged | true | true | pass |
| Constraint violation | <= 1e-06 | 7.72e-12 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | >= 0.980000 | 0.998867 | pass |
| Parameter relative RMSE | <= 0.150000 | 0.065378 | pass |
| Policy total variation | <= 0.030000 | 0.005697 | pass |
| Value RMSE | <= 0.100000 | 0.019445 | pass |
| Q RMSE | <= 0.100000 | 0.022437 | pass |
| Type A regret | <= 0.050000 | 0.000213 | pass |
| Type B regret | <= 0.050000 | 0.000362 | pass |
| Type C regret | <= 0.050000 | 0.000086 | pass |

## Counterfactual Types

The known-truth harness evaluates the same three oracle counterfactual classes
used by NFXP:

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift reward, hold transitions fixed | Payoff counterfactual |
| Type B | Change transitions, hold reward fixed | State-dynamics counterfactual |
| Type C | Disable one non-anchor action | Action-set or design counterfactual |

Current counterfactual recovery:

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.005109 | 7.56e-05 | 0.000238 | 0.000213 |
| Type B | 0.005457 | 8.20e-05 | 0.000363 | 0.000362 |
| Type C | 0.003549 | 3.56e-05 | 0.000114 | 0.000086 |

## Recommended Non-Smoke Settings

Use these settings for structural validation:

```python
MPECEstimator(
    config=MPECConfig(
        solver="sqp",
        outer_max_iter=200,
        tol=1e-8,
        constraint_tol=1e-6,
    ),
    se_method="robust",
    compute_hessian=True,
)
```

Smoke settings only show that the constrained path runs. They are not evidence
that the Bellman residual, stationarity, standard errors, or counterfactual
objects are credible.

## Debugging Order

If MPEC fails to recover known truth, debug in this order:

1. Constraint violation. Check `metadata["final_constraint_violation"]`. A good
   likelihood value is not enough if `V != T_theta V`.
2. Solver success. Check both `summary.converged` and the underlying SLSQP
   success flag.
3. Feature rank. If rank is below the number of reward parameters, theta is
   not identified.
4. Action support. If one action is rare or missing, the likelihood has weak
   information about payoff differences.
5. Standard errors. Non-finite standard errors mean the score outer product or
   implied information matrix is singular.
6. Initialization. MPEC is sensitive to feasible starts. The implementation
   initializes `V` at the Bellman fixed point of the initial `theta`.
7. Scale and anchor. If the logit scale or exit-action anchor is changed,
   reward levels can drift.
8. Hard gates. Do not accept a non-smoke run that only warns. Known-truth
   validation should pass hard gates or raise.

## Implementation Paths

- Lower-level estimator: `../../../src/econirl/estimation/mpec.py`
- Known-truth harness: `../../../validation/known_truth.py`
- Validation runner: `../../../validation/estimators/mpec/run.py`
- High-beta smoke guard: `../../../validation/estimators/mpec/high_beta_smoke.py`
- Validation results: `../../../validation/results/mpec.json`
- Public RTD source: `../../../docs/estimators/mpec.md`
