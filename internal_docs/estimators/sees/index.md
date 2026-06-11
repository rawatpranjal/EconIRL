# SEES Internal Notes

SEES estimates structural dynamic discrete choice models by approximating the
integrated value function with deterministic sieve basis functions. It keeps a
structural reward target but avoids exact nested fixed-point evaluation by
optimizing reward parameters and value-basis coefficients jointly.

This page is the maintainer and AI reference for basis construction, Bellman
penalty design, Schur-complement standard errors, validation caveats, and why
the public RTD surface for SEES remains controlled.

## Source Boundary

The public SEES source exists but is currently excluded from Sphinx. Internal
docs should preserve the full SEES context until the public evidence and
positioning are stable enough to expose.

Durable source context:

- Paper context: `papers.md`
- Vertical links: `links.md`
- Public RTD source, excluded: `../../../docs/estimators/sees.md`
- Validation runner: `../../../validation/estimators/sees/run.py`
- Validation results: `../../../validation/results/sees.json`
- Supplemental results: `../../../validation/results/sees_continuous_covariate_stress.json`, `../../../validation/results/sees_linear_value_spline.json`, `../../../validation/results/sees_rust_random_start.json`

## Purpose

Use SEES when the model is structural, the reward is parametric, transitions
are known or first-stage estimated, and the value function can be approximated
well by deterministic basis functions. It is a structural approximation
method, not a behavioral cloning method.

SEES is not a rescue path for non-identification. If the reward design is
rank-deficient, action support is weak, or the reward gauge is absent, larger
bases and stronger penalties will not create structural identification.

## Paper Logic

Let `Psi` be a sieve basis matrix and `V_K = Psi alpha`. SEES jointly optimizes
structural parameters and sieve coefficients:

```math
(\hat\theta,\hat\alpha)
=
\arg\max_{\theta,\alpha}
\ell_n(\theta,\alpha)
-
\omega_n
\left\|V_K(\alpha)-T_\theta V_K(\alpha)\right\|^2.
```

The theoretical target is not a rough simulation fit. Under smoothness,
identification, sieve-approximation, and penalty-rate conditions, the
structural parameter estimator is consistent and asymptotically normal. In the
finite-state limiting case, if the sieve spans the value vector and the
penalty is strong, SEES approaches the MPEC formulation.

## Model

Flow payoff is linear in known features:

```math
u_\theta(s,a)=\phi(s,a)^\top\theta.
```

Given value approximation `V_K = Psi alpha`, choice-specific values are:

```math
Q_\theta(s,a;V_K)
=
u_\theta(s,a)
+ \beta\sum_{s'}P_a(s,s')V_K(s').
```

The penalized criterion is:

```math
\mathcal Q_n(\theta,\alpha)
=
\sum_{i,t}\log \pi_{\theta,\Psi\alpha}(a_{it}\mid s_{it})
-
\omega
\sum_s
\left[
\Psi_s\alpha - T_\theta(\Psi\alpha)(s)
\right]^2.
```

The optimizer variable is `x = (theta, alpha)`. There is no nested value
iteration. The package reports the unpenalized choice log likelihood at the
final point and reports Bellman violation separately.

## Standard Errors

With Hessian blocks:

```math
H =
\begin{pmatrix}
H_{\theta\theta} & H_{\theta\alpha}\\
H_{\alpha\theta} & H_{\alpha\alpha}
\end{pmatrix},
```

SEES treats `alpha` as a nuisance approximation coefficient and uses the Schur
complement:

```math
\tilde H_\theta
=
H_{\theta\theta}
-
H_{\theta\alpha}
H_{\alpha\alpha}^{-1}
H_{\alpha\theta}.
```

The validation requires finite standard errors. A run with good point
estimates but singular nuisance information is not a full structural
validation.

## Known-Truth Validation Design

The current tracked result uses `canonical_high_action`, the encoded-state
known-truth DGP.

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

Current basis and penalty configuration:

| Quantity | Value |
| --- | ---: |
| Configured basis dimension | 81 |
| Encoded basis rank | 81 |
| Basis source | encoded state |
| Basis family | rbf_svd |
| Sphinx/public exposure | excluded |
| Penalty weight | 10000.0 |
| Bellman RMSE | 6.80e-07 |
| Bellman violation | 3.08e-06 |
| Initial value projection RMSE | 3.38e-14 |

## Pre-Estimation Diagnostics

Current diagnostics from `validation/results/sees.json`:

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

## Current Validation Results

The current result JSON records release status `certified` for the target
"Sieve value approximation for structural dynamic discrete choice, validated
on the encoded-state known-truth DGP."

Run summary:

| Quantity | Value |
| --- | ---: |
| Converged flag | false |
| Optimizer iterations | 6 |
| Log likelihood | -160289.8089 |
| Estimation time | 6.01 seconds |
| Observations | 160,000 |
| Standard errors | finite |
| Bellman violation | 3.08e-06 |

The false optimizer-converged flag is a caveat. The structural gates pass, and
the old primer notes that the sample-summed penalized objective can plateau
with a large absolute gradient while the structural objects are already
recovered. Do not hide this caveat in internal or public wording.

Recovery metrics:

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.000997 |
| Parameter relative RMSE | 0.009528 |
| Parameter cosine similarity | 0.999955 |
| Reward RMSE | 0.004432 |
| Reward normalized RMSE | 0.007684 |
| Value RMSE | 0.037836 |
| Value normalized RMSE | 0.010595 |
| Q RMSE | 0.031480 |
| Policy KL | 2.89e-05 |
| Policy total variation | 0.002117 |
| Policy max state L1 | 0.024416 |

Hard recovery gates:

| Gate | Threshold | Value | Status |
| --- | ---: | ---: | --- |
| Bellman violation | <= 0.050000 | 3.08e-06 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | >= 0.990000 | 0.999955 | pass |
| Parameter relative RMSE | <= 0.150000 | 0.009528 | pass |
| Reward RMSE | <= 0.030000 | 0.004432 | pass |
| Policy total variation | <= 0.020000 | 0.002117 | pass |
| Value RMSE | <= 0.100000 | 0.037836 | pass |
| Q RMSE | <= 0.100000 | 0.031480 | pass |
| Type A regret | <= 0.010000 | 0.000113 | pass |
| Type B regret | <= 0.010000 | 0.000183 | pass |
| Type C regret | <= 0.010000 | 0.000014 | pass |

## Counterfactual Types

| Type | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A | 0.002096 | 0.000234 | 0.000113 |
| Type B | 0.002421 | 0.000309 | 0.000183 |
| Type C | 0.001231 | 0.000027 | 0.000014 |

## Debugging Order

If SEES fails to recover known truth, debug in this order:

1. Feature rank first. If the reward design is not identified, larger sieves
   and larger penalties only produce sharper nonsense.
2. Basis dimension and rank. Low basis dimension is the common finite-sample
   SEES failure mode.
3. Penalty strength. A high likelihood with loose Bellman residual is not a
   structural fit.
4. Standard errors. Non-finite Schur-complement standard errors mean the
   nuisance block is singular or nearly singular.
5. Optimizer flag. Do not ignore it, but interpret it alongside Bellman
   residual and recovery gates.
6. Scale and anchor. If the logit scale or exit-action anchor changes, reward
   levels can drift.
7. Public exposure. Keep SEES hidden from RTD until the caveat language and
   validation status are intentionally accepted.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/sees.py`
- Lower-level estimator: `../../../src/econirl/estimation/sees.py`
- Validation runner: `../../../validation/estimators/sees/run.py`
- Validation results: `../../../validation/results/sees.json`
- Supplemental stress results: `../../../validation/results/sees_continuous_covariate_stress.json`, `../../../validation/results/sees_linear_value_spline.json`, `../../../validation/results/sees_rust_random_start.json`
- Public RTD source, currently excluded: `../../../docs/estimators/sees.md`
