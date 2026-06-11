# NFXP Internal Notes

NFXP is the exact nested fixed-point maximum-likelihood reference estimator for
tabular dynamic discrete choice models. It estimates primitive reward
parameters by solving the agent's dynamic program inside each likelihood
evaluation, then optimizing the structural parameters outside that fixed point.

This internal page is intentionally denser than the public RTD page. It is the
agent and maintainer reference for derivations, validation interpretation,
debugging order, and paper-to-package distinctions.

## Source Boundary

The public RTD page should present NFXP as the exact structural reference
estimator and link to validation evidence. It should not carry the full
derivation, debugging matrix, or old primer context.

Durable source context:

- Paper context: `papers.md`
- Vertical links: `links.md`
- Public RTD source: `../../../docs/estimators/nfxp.md`
- Validation runner: `../../../validation/estimators/nfxp/run.py`
- Validation results: `../../../validation/results/nfxp.json`

## Purpose

NFXP is the baseline against which CCP, MPEC, NNES, TD-CCP, SEES, and many
IRL-style estimators should be compared. It is appropriate when the state space
is tabular, the transition kernel is known or estimated in a first stage, the
reward is low-dimensional, and exact dynamic programming is feasible.

It is usually the wrong default for high-dimensional or continuous state
spaces. For those cases, compare against NNES, SEES, TD-CCP, or GLADIUS
depending on whether the project needs structural parameter recovery,
value-function approximation, transition-free parameter estimation, or a
Q-function method.

## Model

The observed panel contains state-action-next-state trajectories:

```text
(s_it, a_it, s_i,t+1), i = 1,...,N, t = 1,...,T.
```

The flow payoff is linear in known features:

```math
u_\theta(s,a) = \phi(s,a)^\top \theta.
```

The transition kernel is separated from payoff estimation:

```math
P_a(s,s') = \Pr(s_{t+1}=s' \mid s_t=s, a_t=a).
```

With discount factor `beta` and Type-I extreme-value shock scale `sigma`, the
integrated value function solves the soft Bellman fixed point:

```math
V_\theta(s)
= \sigma \log \sum_a
\exp\left(
\frac{u_\theta(s,a) + \beta \sum_{s'} P_a(s,s')V_\theta(s')}{\sigma}
\right).
```

The choice-specific value is:

```math
Q_\theta(s,a)
= u_\theta(s,a) + \beta \sum_{s'} P_a(s,s')V_\theta(s').
```

The implied conditional choice probability is:

```math
\pi_\theta(a \mid s)
=
\frac{\exp(Q_\theta(s,a)/\sigma)}
{\sum_b \exp(Q_\theta(s,b)/\sigma)}.
```

NFXP maximizes the conditional log likelihood:

```math
\hat{\theta}
=
\arg\max_\theta
\sum_{i,t} \log \pi_\theta(a_{it} \mid s_{it}).
```

## Estimator Mechanics

For each candidate `theta`, the inner loop solves `V = T V`. EconIRL supports
successive approximation, Newton-Kantorovich, and a hybrid solver. The
known-truth validation uses the hybrid solver: safe successive approximation
far from the fixed point, then Newton-Kantorovich updates near the solution.

The analytical score differentiates through the fixed point. For observation
`i`, the score has the logit form:

```math
\psi_i(\theta)
=
\frac{1}{\sigma}
\left[
\nabla_\theta Q(s_i,a_i)
-
\sum_a \pi_\theta(a \mid s_i)\nabla_\theta Q(s_i,a)
\right].
```

The Q-gradient is:

```math
\nabla_\theta Q(s,a)
=
\phi(s,a)
+
\beta \sum_{s'} P_a(s,s')\frac{\partial V(s')}{\partial \theta}.
```

The total derivative solves:

```math
(I-\beta P_\pi)\frac{\partial V}{\partial \theta}
=
\sum_a \pi_\theta(a \mid s)\phi(s,a).
```

The outer loop uses BHHH by default. It builds the positive semi-definite
outer-product matrix from per-observation scores, applies damping when needed,
and uses line search to maintain likelihood improvement.

## Identification Requirements

NFXP validation assumes three structural restrictions:

- Conditional independence. The state transition is Markov in observed state
  and action and does not depend on the current logit shock.
- Transition separation. The transition process is supplied or estimated
  outside the payoff likelihood.
- Reward normalization. The logit shock scale is fixed and payoff location is
  pinned down through an anchor. In the canonical known-truth cell, the exit
  action and absorbing state provide the zero-reward anchor.

The practical failures are feature rank, weak action support, invalid
normalization, state-only reward variation copied across actions, wrong
transition tensor orientation, and loose inner fixed-point tolerances.

## Known-Truth Validation Design

The current NFXP validation is a synthetic known-truth exercise, not a Rust
data replication. The DGP supplies the true reward, transition law, value
function, policy, Q function, and Type A/B/C counterfactual oracle objects.

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

Run diagnostics before fitting. If these fail, the issue is usually the DGP or
feature design, not the optimizer.

Current diagnostics from `validation/results/nfxp.json`:

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.512 | pass |
| Transition row error | 2.42e-08 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Action shares | 0.345, 0.330, 0.325 | pass |
| Minimum action share | 0.325 | pass |
| Exit/absorbing anchor | true | pass |

## Current Validation Results

The current result JSON records release status `Certified` for the synthetic
validation target "Exact nested fixed-point maximum likelihood for tabular
DDC." In internal docs, that means the committed result file passes the
known-truth gates for this specific DGP. It should not be generalized to every
empirical dataset.

Run summary:

| Quantity | Value |
| --- | ---: |
| Converged | true |
| Outer iterations | 12 |
| Log likelihood | -174875.7719 |
| Estimation time | 5.90 seconds |
| Function evaluations | 80 |
| Total inner iterations | 11,259 |
| Final inner iterations | 139 |
| Inner solver | hybrid |
| Outer optimizer | BHHH |

Parameter recovery:

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083894 | 0.029335 | -0.016106 |
| `action_0_progress` | 0.500000 | 0.528522 | 0.035889 | 0.028522 |
| `action_1_intercept` | 0.000000 | -0.014461 | 0.036733 | -0.014461 |
| `action_1_progress` | -0.200000 | -0.200511 | 0.052502 | -0.000511 |

Recovery metrics:

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017904 |
| Parameter relative RMSE | 0.065378 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019445 |
| Q RMSE | 0.022438 |
| Policy KL | 9.21e-05 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018905 |

Hard recovery gates:

| Gate | Threshold | Value | Status |
| --- | ---: | ---: | --- |
| converged | true | true | pass |
| parameter cosine | >= 0.980000 | 0.998867 | pass |
| parameter relative RMSE | <= 0.150000 | 0.065378 | pass |
| policy total variation | <= 0.030000 | 0.005697 | pass |
| value RMSE | <= 0.100000 | 0.019445 | pass |
| Type A regret | <= 0.050000 | 0.000213 | pass |
| Type B regret | <= 0.050000 | 0.000362 | pass |
| Type C regret | <= 0.050000 | 0.000086 | pass |

## Counterfactual Types

The known-truth harness evaluates three oracle counterfactual classes:

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
| Type C | 0.003548 | 3.56e-05 | 0.000114 | 0.000086 |

These regrets are small because the recovered reward is close enough to the
true reward that re-solving the intervened model produces almost the same
policy as the oracle.

## Recommended Non-Smoke Settings

Use these settings for structural validation, not for a quick unit smoke:

```python
NFXPEstimator(
    optimizer="BHHH",
    inner_solver="hybrid",
    inner_tol=1e-12,
    inner_max_iter=100_000,
    switch_tol=1e-3,
    outer_tol=1e-6,
    outer_max_iter=500,
    compute_hessian=True,
)
```

Smoke settings only show that the estimator runs and returns finite objects.
They are not evidence of parameter, reward, policy, or counterfactual recovery.

## Debugging Order

If NFXP fails to recover known truth, debug in this order:

1. Feature rank. If rank is below the number of reward parameters, theta is not
   identified.
2. Action support. If one action is rarely observed, its payoff is weakly
   identified.
3. Normalization. If the exit or absorbing anchor is invalid, reward levels can
   drift.
4. State-only rewards. The canonical theta validation needs action-dependent
   features.
5. Transition tensor. NFXP needs stochastic transition rows and the
   action-state-next-state orientation expected by the estimator.
6. Inner tolerance. Use `1e-10` to `1e-12` for structural validation.
7. High beta. Use `inner_solver="hybrid"` near `beta = 1`; pure value iteration
   is intentionally slow there.
8. Hard gates. Do not accept a non-smoke run that only warns. Known-truth
   validation should pass hard gates or raise.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/nfxp.py`
- Lower-level estimator: `../../../src/econirl/estimation/nfxp.py`
- Known-truth harness: `../../../validation/known_truth.py`
- Validation runner: `../../../validation/estimators/nfxp/run.py`
- Validation results: `../../../validation/results/nfxp.json`
- Public RTD source: `../../../docs/estimators/nfxp.md`
