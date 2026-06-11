# Under the Hood

TD-CCP estimates the reward parameters in two stages. First it learns the
future-utility terms that enter the CCP likelihood. Then it estimates the
reward parameters using those learned terms.

The important boundary is that known transition tensors are not used to
estimate `theta`. They may be used later for artifact checks, policy
evaluation, or counterfactual analysis.

## Model Objects

The observed panel supplies current and next state-action tuples.

$$
(a_t, x_t, a_{t+1}, x_{t+1})
$$

The flow payoff is linear in known reward features.

$$
u_\theta(a, x) = z(a, x)^\top \theta
$$

The CCP likelihood needs an index for each action. TD-CCP writes that index
using two learned continuation objects, `h` and `g`.

$$
\tilde Q_\theta(a, x) = h(a, x)^\top \theta + g(a, x)
$$

The implied policy is a soft-max over those action indices.

$$
\pi_\theta(a \mid x)
=
\frac{\exp\{h(a, x)^\top\theta + g(a, x)\}}
     {\sum_b \exp\{h(b, x)^\top\theta + g(b, x)\}}
$$

The objects `h` and `g` satisfy TD fixed-point equations. The first equation
tracks the future reward features. The second tracks the future choice-shock
correction.

$$
h(a, x) = z(a, x) + \beta E[h(a', x') \mid a, x]
$$

$$
g(a, x) =
\beta E[e(a', x') + g(a', x') \mid a, x],
\quad e(a, x) = \gamma_E - \log P(a \mid x)
$$

## Estimation Flow

The reported EconIRL path follows the semigradient version of the paper.

1. Estimate conditional choice probabilities from observed actions.
2. Build observed current and successor tuples from the panel.
3. Learn `h` and `g` with projected TD equations.
4. Estimate a preliminary reward parameter vector.
5. Estimate the correction recursion used by the locally robust estimator.
6. Solve the held-out moment equation in each fold.
7. Average fold estimates and compute standard errors clustered by individual.

The correction recursion is the paper's `lambda` object. The held-out moment
equation is the paper's `zeta` moment. EconIRL records both diagnostics in the
simulation artifact, but the user-facing interpretation is simple: they correct
for first-stage estimation error and support valid standard errors.

## Semigradient and AVI

The semigradient path uses linear basis functions for the continuation terms.
That makes the recursive-term step a matrix problem rather than an integration
problem over a fitted transition density.

The AVI path is more flexible. It repeatedly solves prediction problems for
the continuation terms and can use machine-learning learners. EconIRL exposes
this path, but the current simulation study is tied to the semigradient,
locally robust path.

## Paper Alignment

| Paper step | EconIRL implementation |
| --- | --- |
| Estimate CCPs `P(a given x)` | Logit first stage with degree-2 encoded-state features |
| Estimate `h` | Semigradient TD solve using encoded basis functions |
| Estimate `g` | Semigradient TD solve using the log-CCP shock correction |
| Preliminary parameter estimate | Fold-specific CCP pseudo-likelihood solve |
| Correction recursion | Fold-specific backward recursion with residual diagnostics |
| Locally robust moment | Held-out fold solve using quantities learned on the other fold |
| Standard errors | Fold covariance averaged and clustered by individual |

## Reported Settings

| Setting | Value |
| --- | --- |
| Method | `semigradient` |
| Basis | `encoded`, degree 2 |
| CCP model | `logit`, degree 2 |
| Cross-fitting | `True`, split by individual |
| Robust standard errors | `True`, locally robust covariance clustered by individual |
| Policy iterations | `1` |
| Outer optimizer tolerance | `1e-7` |
