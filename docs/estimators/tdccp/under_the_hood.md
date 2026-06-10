# Under the Hood

TD-CCP estimates the recursive terms that appear in a CCP pseudo-likelihood.
It does not solve a Bellman equation inside each likelihood evaluation and it
does not need a transition-density model while estimating `theta`.

## Model

The observed data are current and successor state-action tuples.

$$
(a_t, x_t, a_{t+1}, x_{t+1})
$$

The flow payoff is finite-dimensional and linear in known features.

$$
u_\theta(a, x) = z(a, x)^\top \theta
$$

The CCP index uses two recursive objects.

$$
\tilde Q_\theta(a, x) = h(a, x)^\top \theta + g(a, x)
$$

The implied choice probability is the usual soft-max over choice-specific
indices.

$$
\pi_\theta(a \mid x)
=
\frac{\exp\{h(a, x)^\top\theta + g(a, x)\}}
     {\sum_b \exp\{h(b, x)^\top\theta + g(b, x)\}}
$$

The recursive terms satisfy TD fixed-point equations.

$$
h(a, x) = z(a, x) + \beta E[h(a', x') \mid a, x]
$$

$$
g(a, x) =
\beta E[e(a', x') + g(a', x') \mid a, x],
\quad e(a, x) = \gamma_E - \log P(a \mid x)
$$

## Paper Alignment

| Paper step | EconIRL implementation and evidence |
| --- | --- |
| Estimate CCPs `P(a given x)` | Logit first stage with degree-2 encoded-state features in the certified artifact. |
| Estimate `h` | Semigradient projected TD solve using encoded basis functions. |
| Estimate `g` | Semigradient projected TD solve using the log-CCP shock correction. |
| Obtain preliminary `tilde theta` | Fold-specific plug-in CCP pseudo-likelihood solve with projected-gradient diagnostics. |
| Estimate `lambda` | Fold-specific backward recursion; residual norms are written to the artifact. |
| Solve zeta moment | Held-out fold solves the locally robust moment using opposite-fold nuisances. |
| Report covariance | Fold covariance is averaged and clustered by individual for the reported SEs. |

## Algorithm Sketch

TD-CCP first estimates conditional choice probabilities from observed actions.
It then extracts current and successor state-action tuples from the panel.

1. Estimate CCPs and the Type I Extreme Value shock correction.
2. Estimate recursive terms `h` and `g`.
3. Compute a preliminary plug-in `theta` for each training fold.
4. Estimate the backward `lambda` correction on the same nuisance fold.
5. Solve the held-out locally robust `zeta` moment using the opposite-fold
   nuisances.
6. Average fold estimates and compute the locally robust covariance.

Known transitions are not used for the parameter-estimation step. In EconIRL
validation, supplied transitions are used after estimation to evaluate the
recovered reward through policies, values, Q functions, and counterfactual
oracle comparisons.

## Semigradient and AVI

The certified release path is semigradient TD. With linear basis functions,
the `h` and `g` recursions reduce to projected TD normal equations, so the
recursive-term step is a matrix solve rather than a transition-density
integration problem.

AVI is the more flexible paper path. It repeatedly solves prediction problems
for the next recursive approximation and can use machine-learning learners.
EconIRL exposes this path, but the current RTD certification does not use it
for the release claim because the artifact's finite-theta inference evidence
is attached to the semigradient Algorithm 2 path.

## Locally Robust Inference

The reported standard errors come from the Algorithm 2 moment, not from a
plain plug-in Hessian. Each fold stores the preliminary plug-in solution, the
lambda fixed-point residual, the zeta mean, and the fold covariance. The final
artifact averages fold estimates and reports individual-clustered covariance.
Optimizer stationarity diagnostics are archived for both preliminary and final
robust solves.

## Certified Path

The certified TD-CCP artifact uses:

| Setting | Value |
| --- | --- |
| Method | `semigradient` |
| Basis | `encoded`, degree 2 |
| CCP model | `logit`, degree 2 |
| Cross-fitting | `True`, split by individual |
| Robust standard errors | `True`, Algorithm 2 locally robust covariance clustered by individual |
| Policy iterations | `1` |
| Outer optimizer tolerance | `1e-7` |

This is the strongest package evidence because the hard case keeps the paper's
finite-dimensional `theta` target while using encoded state features and
transition-free TD estimates of the recursive CCP terms.
