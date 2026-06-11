# NNES Internal Notes

NNES estimates structural dynamic discrete choice models with a neural
approximation to the integrated value function. The target remains structural:
recover reward parameters, the implied policy, the value function, Q values,
and counterfactual policies. The neural object approximates continuation
values; it is not a neural reward plug-in for structural estimators.

This page is the maintainer and AI reference for the NPL connection, value
network role, orthogonality claim, validation caveats, and paper-to-package
boundary.

## Source Boundary

The public RTD page should explain that NNES is a structural estimator with
neural value approximation. It should not present NNES as model-free IRL or
as unrestricted neural reward recovery.

Durable source context:

- Paper context: `papers.md`
- Vertical links: `links.md`
- Public RTD source: `../../../docs/estimators/nnes.md`
- Validation runner: `../../../validation/estimators/nnes/run.py`
- Validation results: `../../../validation/results/nnes.json`

## Purpose

Use NNES when the reward is still finite-dimensional and structural, but the
state representation is large or encoded enough that repeated exact tabular
dynamic programming is not the right default. It is closest in spirit to CCP
and NPL, but replaces the tabular continuation object with a trained neural
value approximation.

NNES is not a behavioral cloning shortcut. In-sample action fit is not enough.
The package validation checks parameters, rewards, policy, value, Q, and
counterfactual regrets against known truth.

## Paper Logic

The source paper embeds a neural value function approximation in the nested
pseudo-likelihood framework. The important theoretical idea is a zero-Jacobian
or Neyman-orthogonality property: at the true conditional choice probabilities,
first-order value-approximation error drops out of the structural score.

The paper-level implication is asymptotic. If the value approximation is
accurate enough, the structural estimator can be root-n consistent and
efficient. This is not a finite-sample guarantee. The package still has to
verify that the final reward, policy, value, Q, and counterfactual objects are
mutually consistent.

## Model

Observed data are state-action-next-state trajectories:

```text
(s_it, a_it, s_i,t+1), i = 1,...,N, t = 1,...,T.
```

Flow payoff is linear:

```math
u_\theta(s,a)=\phi(s,a)^\top\theta.
```

Given continuation value `V`, the choice-specific value is:

```math
Q_\theta(s,a;V)
=
u_\theta(s,a)+\beta\sum_{s'}P_a(s,s')V(s').
```

The implied logit policy is:

```math
\pi_{\theta,V}(a\mid s)
=
\frac{\exp(Q_\theta(s,a;V)/\sigma)}
{\sum_b \exp(Q_\theta(s,b;V)/\sigma)}.
```

## Estimator Mechanics

Given CCPs `P(a | s)`, the finite-state NPL evaluation step profiles the value
representation as:

```math
W_\theta[P] = W_z[P]\theta + W_e[P].
```

Here `W_z[P]` is the expected discounted feature valuation under the fixed
policy iterate and `W_e[P]` is the expected discounted logit entropy term.
The implementation trains a ReLU MLP `V_gamma(s)` by supervised regression on
that profiled value target, then optimizes the structural parameters through
the profiled one-step NPL likelihood rather than freezing continuation values
at the previous parameter vector.

Operational loop:

1. Estimate initial CCPs from empirical state-action frequencies.
2. Compute the NPL value target for the current policy iterate.
3. Normalize the target by the anchor state.
4. Train the value network on all states.
5. Maximize the profiled pseudo-likelihood over `theta`.
6. Update CCPs from the implied logit policy.
7. Evaluate the final policy value on the Bellman scale for validation.

The current validation metadata records `profile_mode =
exact_finite_state_npl`, `n_outer_iterations = 3`, and value-regression losses
`0.0770`, `0.0269`, and `0.0299`.

## Comparison With SEES

SEES approximates `V` with deterministic basis functions and penalizes the
Bellman residual in a joint optimization problem. NNES uses a neural value
approximation and NPL orthogonality instead. In the finite-state validation
harness, the profiled NPL step is an exact policy-evaluation specialization of
the neural approximation idea.

## Known-Truth Validation Design

The current package result is the high-dimensional known-truth cell. The old
primer also described a low-dimensional sanity cell, but the tracked result
file uses `canonical_high_action` as the primary evidence.

Current validation cell:

| Quantity | Value |
| --- | ---: |
| Cell ID | `canonical_high_action` |
| Regular states | 80 |
| Absorbing states | 1 |
| Total states | 81 |
| State-feature dimension | 16 |
| Reward parameters | 32 |
| Actions | 3 |
| Exit action | 2 |
| Discount factor | 0.95 |
| Shock scale | 1.0 |
| Reward mode | Action-dependent |
| Simulated individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |

## Pre-Estimation Diagnostics

Current diagnostics from `validation/results/nnes.json`:

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 32 / 32 | pass |
| Feature condition number | 1.377 | pass |
| Transition row error | 2.42e-08 | pass |
| Observed states | 81 / 81 | pass |
| State-action coverage | 0.959 | support caveat |
| Action shares | 0.333, 0.386, 0.281 | pass |
| Minimum action share | 0.281 | pass |
| Single-action states | 0 | pass |
| Exit/absorbing anchor | true | pass |

The state-action coverage caveat matters. The result passes the current hard
gates, but it is not the same support profile as the fully covered
low-dimensional structural cell.

## Current Validation Results

The current result JSON records release status `Certified with caveat` for the
target "Neural value-function approximation for structural dynamic discrete
choice, validated on the high-dimensional known-truth DGP."

Run summary:

| Quantity | Value |
| --- | ---: |
| Converged | true |
| Outer iterations | 29 |
| NPL outer iterations | 3 |
| Final value-network loss | 0.029932 |
| Log likelihood | -160272.8803 |
| Estimation time | 36.46 seconds |
| Observations | 160,000 |
| Standard errors | not available in current result |

Recovery metrics:

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.014141 |
| Parameter relative RMSE | 0.135110 |
| Parameter cosine similarity | 0.991204 |
| Reward RMSE | 0.064012 |
| Reward normalized RMSE | 0.122843 |
| Value RMSE | 0.115620 |
| Value normalized RMSE | 0.037387 |
| Q RMSE | 0.137145 |
| Policy KL | 0.002694 |
| Policy total variation | 0.023834 |
| Policy max state L1 | 0.195824 |

Hard recovery gates:

| Gate | Threshold | Value | Status |
| --- | ---: | ---: | --- |
| NPL outer iterations | >= 3 | 3 | pass |
| Final value loss | <= 0.050000 | 0.029932 | pass |
| Parameter cosine | >= 0.950000 | 0.991204 | pass |
| Parameter relative RMSE | <= 0.300000 | 0.135110 | pass |
| Reward RMSE | <= 0.080000 | 0.064012 | pass |
| Policy total variation | <= 0.030000 | 0.023834 | pass |
| Value RMSE | <= 0.200000 | 0.115620 | pass |
| Q RMSE | <= 0.200000 | 0.137145 | pass |
| Type A regret | <= 0.050000 | 0.004865 | pass |
| Type B regret | <= 0.050000 | 0.005559 | pass |
| Type C regret | <= 0.050000 | 0.001314 | pass |

## Counterfactual Types

| Type | Intervention | Current regret |
| --- | --- | ---: |
| Type A | Reward shift with baseline transitions | 0.004865 |
| Type B | Transition change with baseline reward | 0.005559 |
| Type C | Disable one non-anchor action | 0.001314 |

Counterfactual policy total variation is `0.021578` for Type A, `0.021651`
for Type B, and `0.013053` for Type C.

## Caveats

- Current standard errors are `null` in the result file. Do not describe the
  current NNES result as a full inference validation.
- The claim is high-dimensional known-truth recovery under supplied
  transitions, not empirical Rust replication.
- The neural object is the value approximation, not the reward function.
- The current gate thresholds are broader than the exact structural
  estimators because this is an approximate neural-value path.

## Debugging Order

If NNES fails to recover known truth, debug in this order:

1. Confirm `bellman="npl"` or the intended profiled NPL path. The neural
   Bellman diagnostic path does not carry the same orthogonality claim.
2. Check state-action support and one-action states before changing neural
   architecture.
3. Check final value-network loss by outer iteration.
4. Confirm reward features are finite-dimensional and structural.
5. Confirm transitions are supplied or correctly estimated for validation.
6. Compare reward, policy, value, Q, and counterfactual gates together. A
   good value-network loss alone is not structural evidence.
7. Do not claim inference support until finite standard errors are present and
   tested.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/nnes.py`
- Lower-level estimator: `../../../src/econirl/estimation/nnes.py`
- Known-truth harness: `../../../validation/known_truth.py`
- Validation runner: `../../../validation/estimators/nnes/run.py`
- Validation results: `../../../validation/results/nnes.json`
- Public RTD source: `../../../docs/estimators/nnes.md`
