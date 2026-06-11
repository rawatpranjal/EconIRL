# CCP Internal Notes

CCP estimates tabular dynamic discrete choice models by replacing repeated
Bellman fixed-point solves with first-stage conditional-choice-probability
inversion. It is the fast structural counterpart to NFXP when empirical policy
support is strong.

This page is the maintainer and AI reference for the Hotz-Miller inversion,
NPL mechanics, support failures, validation evidence, and the boundary between
CCP as a structural estimator and CCP as a fragile first-stage object.

## Source Boundary

The public RTD page should explain when CCP is appropriate, how to run the
wrapper, and what evidence supports the current package claim. It should not
carry the full inversion derivation or debugging matrix.

Durable source context:

- Paper context: `papers.md`
- Vertical links: `links.md`
- Public RTD source: `../../../docs/estimators/ccp.md`
- Validation runner: `../../../validation/estimators/ccp/run.py`
- Validation results: `../../../validation/results/ccp.json`

## Purpose

CCP is useful when the state-action space is tabular, transitions are known or
first-stage estimated, empirical action support is strong, and solving the
dynamic program inside every likelihood evaluation is too expensive. It starts
from an empirical policy, inverts that policy into continuation values, and
then estimates payoff parameters through a pseudo-likelihood.

CCP should not be used as a recovery benchmark when many states are rarely
observed or when an action is almost never chosen. The first-stage CCPs are
the estimator's foundation; if they are poorly supported, the inversion
amplifies noise.

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

Given a policy `pi`, define the policy-weighted transition matrix:

```math
F_\pi(s,s') = \sum_a \pi(a \mid s) P_a(s,s').
```

The logit emax correction is:

```math
e_\pi(s,a) = \gamma_{\mathrm{Euler}} - \log \pi(a \mid s).
```

Hotz-Miller inversion writes the integrated value under policy `pi` as:

```math
\bar V_\pi
=
(I-\beta F_\pi)^{-1}
\sum_a \pi(a \mid s)\{u_\theta(s,a)+e_\pi(s,a)\}.
```

For linear rewards, this separates into augmented features:

```math
\bar V_\pi(s) = W_\phi(s)^\top\theta + W_e(s),
```

where:

```math
W_\phi
=
(I-\beta F_\pi)^{-1}
\sum_a \pi(a \mid s)\phi(s,a),
```

and:

```math
W_e
=
(I-\beta F_\pi)^{-1}
\sum_a \pi(a \mid s)e_\pi(s,a).
```

The pseudo-likelihood choice-specific value is:

```math
\tilde Q_\theta(s,a;\pi)
=
\phi(s,a)^\top\theta
+ \beta \sum_{s'}P_a(s,s')\{W_\phi(s')^\top\theta+W_e(s')\}.
```

The K-stage CCP/NPL estimator solves:

```math
\hat\theta_K
=
\arg\max_\theta
\sum_{i,t}
\log
\frac{\exp(\tilde Q_\theta(s_{it},a_{it};\hat\pi^{K-1})/\sigma)}
{\sum_b \exp(\tilde Q_\theta(s_{it},b;\hat\pi^{K-1})/\sigma)}.
```

## Estimator Mechanics

The implementation follows five steps:

1. Estimate empirical CCPs with Laplace smoothing.
2. Compute `F_pi`, `e_pi`, `W_phi`, and `W_e` once per NPL step.
3. Build augmented features and maximize a multinomial logit
   pseudo-likelihood in `theta`.
4. Update CCPs from fitted choice-specific values.
5. Repeat until the requested NPL count or convergence condition is reached.

The cost shift relative to NFXP is the key design point. NFXP solves a Bellman
fixed point inside every likelihood evaluation. CCP pays for a matrix
inversion per NPL step, then each optimizer step is a logit problem on
augmented features.

## Identification And Support Requirements

CCP inherits the same payoff identification restrictions as NFXP and adds
first-stage support requirements:

- Reward rank. The action-dependent feature matrix must have full column rank
  after dropping zero rows.
- Action support. Every action must have enough empirical mass. Near-zero
  CCPs make `gamma - log pi(a | s)` unstable.
- State coverage. The policy is estimated nonparametrically by state. Sparse
  states should be pooled, smoothed, or excluded from a main validation claim.
- Normalization. The reward scale is fixed and the canonical synthetic DGP
  uses an exit action and absorbing state as a zero-reward anchor.
- Transition separation. The current validation supplies the true transition
  tensor, so the result isolates the payoff estimator rather than transition
  estimation error.

## Known-Truth Validation Design

The current CCP validation is synthetic known-truth evidence, not a real-data
replication. It uses the same low-dimensional action-dependent structural
benchmark as NFXP.

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

For CCP, support diagnostics are not optional. The estimator starts from
empirical CCPs.

Current diagnostics from `validation/results/ccp.json`:

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

The current result JSON records release status `Certified with support
conditions` for the target "Hotz-Miller and NPL pseudo-likelihood for tabular
DDC." In internal docs, this means the committed synthetic result passes the
hard gates and the first-stage support diagnostics in this cell. It does not
mean arbitrary sparse empirical panels are safe for CCP.

Run summary:

| Quantity | Value |
| --- | ---: |
| Converged | true |
| NPL iterations | 8 |
| Requested policy iterations | 10 |
| Log likelihood | -174875.9375 |
| Estimation time | 16.31 seconds |
| Observations | 160,000 |
| Mode | npl |
| First-stage smoothing | 1e-06 |
| Minimum initial CCP | 0.152756 |
| Minimum final CCP | 0.151002 |

Parameter recovery:

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083896 | 0.009450 | -0.016104 |
| `action_0_progress` | 0.500000 | 0.528517 | 0.009222 | 0.028517 |
| `action_1_intercept` | 0.000000 | -0.014458 | 0.026148 | -0.014458 |
| `action_1_progress` | -0.200000 | -0.200517 | 0.020787 | -0.000517 |

Recovery metrics:

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017902 |
| Parameter relative RMSE | 0.065367 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019436 |
| Q RMSE | 0.022430 |
| Policy KL | 9.21e-05 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018902 |

Hard recovery gates:

| Gate | Threshold | Value | Status |
| --- | ---: | ---: | --- |
| NPL iterations | >= 5 | 8 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | >= 0.980000 | 0.998867 | pass |
| Parameter relative RMSE | <= 0.150000 | 0.065367 | pass |
| Policy total variation | <= 0.030000 | 0.005697 | pass |
| Value RMSE | <= 0.100000 | 0.019436 | pass |
| Q RMSE | <= 0.100000 | 0.022430 | pass |
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
| Type C | 0.003548 | 3.56e-05 | 0.000114 | 0.000086 |

## Recommended Non-Smoke Settings

Use these settings for structural validation:

```python
CCPEstimator(
    num_policy_iterations=10,
    ccp_min_count=1,
    ccp_smoothing=1e-6,
    convergence_tol=1e-6,
    outer_tol=1e-6,
    outer_max_iter=500,
    se_method="robust",
    compute_hessian=True,
)
```

Smoke settings only show that the estimator runs. They are not evidence of
support, inversion stability, or counterfactual recovery.

## Debugging Order

If CCP fails to recover known truth, debug in this order:

1. First-stage support. Check observed states, state-action coverage, minimum
   action share, and minimum positive CCP.
2. One-action states. If a state has only one observed action, Hotz-Miller
   inversion is extrapolating that state's counterfactual action values.
3. Smoothing. Too little smoothing makes logs unstable. Too much smoothing
   biases the first-stage policy toward uniform choice.
4. Feature rank. If rank is below the number of reward parameters, theta is
   not identified.
5. Normalization. If the exit or absorbing anchor is invalid, reward levels
   can drift.
6. NPL iterations. A one-step estimator can be consistent under assumptions,
   but the current validation claim uses the NPL path and hard gates.
7. Value convention. CCP inversion includes the Euler-constant emax correction
   internally. For validation, EconIRL compares values by evaluating the
   recovered policy and reward in the package soft-Bellman convention.
8. Hard gates. Do not accept a non-smoke run that only warns. Known-truth
   validation should pass hard gates or raise.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/ccp.py`
- Lower-level estimator: `../../../src/econirl/estimation/ccp.py`
- Known-truth harness: `../../../validation/known_truth.py`
- Validation runner: `../../../validation/estimators/ccp/run.py`
- Validation results: `../../../validation/results/ccp.json`
- Public RTD source: `../../../docs/estimators/ccp.md`
