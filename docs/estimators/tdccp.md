# TD-CCP

## Important Links

- [Quick Start](tdccp/quick_start.md)
- [Pre-Estimation Checks](tdccp/pre_estimation.md)
- [Evidence](tdccp/validation.md)
- [Counterfactuals](tdccp/counterfactuals.md)
- [Rust Bus Engine Example](tdccp/rust_bus.md)

TD-CCP estimates finite structural reward parameters from observed current and
successor choices. Its parameter stage learns recursive continuation terms
without fitting a transition density. A transition environment is still needed
to compute fitted policies, values, simulations, and counterfactuals.

The default is the
paper's linear semigradient method with two-fold cross-fitting and locally
robust inference. Neural and gradient-boosting approximate value iteration are
also available.

## Source Papers

{ref}`Adusumilli and Eckardt (2025) <adusumilli-eckardt-2025>` introduces the
temporal-difference recursions, the semigradient and approximate value iteration
estimators, and Algorithm 2 inference. {ref}`Hotz and Miller (1993)
<hotz-miller-1993>` provides the conditional choice probability foundation.

## Model

Let $x$ denote the state and $a$ the action. The flow payoff is linear in known
features:

$$
u_\theta(a,x) = z(a,x)^\top \theta.
$$

TD-CCP works with transition tuples
$(a_t,x_t,a_{t+1},x_{t+1})$. The reward accumulator $h$ and shock accumulator
$g$ satisfy

$$
h(a,x) = z(a,x) + \beta\,
\mathbb{E}[h(a',x')\mid a,x],
$$

and

$$
g(a,x) = \beta\,
\mathbb{E}[e(a',x') + g(a',x')\mid a,x],
\qquad
e(a,x) = \gamma_{\mathrm E} - \log P(a\mid x).
$$

Here $P(a\mid x)$ is the first-stage conditional choice probability and
$\gamma_{\mathrm E}$ is the Euler constant. Both expectations can be learned
directly from successor tuples. The parameter stage therefore avoids a model
for $P(x'\mid a,x)$.

The choice-specific value becomes

$$
\widetilde Q_\theta(a,x) = h(a,x)^\top\theta + g(a,x).
$$

The package normalizes the Type I extreme-value shock scale to one. The implied
choice probability is

$$
\pi_\theta(a\mid x)
= \frac{\exp\{\widetilde Q_\theta(a,x)\}}
       {\sum_b \exp\{\widetilde Q_\theta(b,x)\}}.
$$

## Identification

TD-CCP relies on the following identifying assumptions and design conditions.

- State transitions are Markov conditional on the current state and action.
- Choice shocks are additive Type-I extreme value draws with fixed scale.
- The reward is linear in a known finite feature vector.
- A baseline action or equivalent reward normalization anchors the model.
- The action-contrast feature matrix has full column rank.
- The discount factor is known and fixed.

State-only features that are copied across every action disappear from choice
contrasts. The public wrapper rejects this design before fitting. It also
checks raw feature rank, state and action support, and transition orientation.

## Algorithm

### Linear Semigradient Method

Let $\phi(a,x)$ be the basis used to approximate the recursive terms. The
semigradient coefficient for $h$ solves

$$
\widehat\omega_h
= \left[
\mathbb E_n\{\phi(a,x)(\phi(a,x)-\beta\phi(a',x'))^\top\}
\right]^{-1}
\mathbb E_n\{\phi(a,x)z(a,x)^\top\}.
$$

The same left-hand matrix is used for $g$. Its right-hand target is
$\beta e(a',x')$. The default basis uses action-interacted polynomials of the
state index. `basis_type="encoded"` instead applies polynomial terms to a
user-supplied state encoder. `basis_type="tabular"` uses one indicator for
each state-action pair.

The preliminary parameter estimate maximizes

$$
\widetilde\theta
= \arg\max_\theta
\sum_{i,t}\log\pi_\theta(a_{it}\mid x_{it}).
$$

The criterion uses the same $T-1$ transition-indexed observations as the
recursive-term equations.

## Locally Robust Inference

The plug-in score treats $\widehat h$ and $\widehat g$ as fixed. Algorithm 2
adds a correction for their first-stage estimation error. Define

$$
m(a,x;\theta,h,g)
= \partial_\theta\log\pi_\theta(a\mid x).
$$

The backward function $\lambda$ solves

$$
\lambda(a,x)
= -m(a,x;\widetilde\theta,\widehat h,\widehat g)
+ \beta\,\mathbb E[\lambda(a_{t-1},x_{t-1})\mid a_t=a,x_t=x].
$$

The corrected moment is

$$
\zeta
= m(a,x;\theta,\widehat h,\widehat g)
+ \lambda(a,x)\left\{
z(a,x)^\top\widetilde\theta
+ \beta e(a',x')
+ \beta V(a',x';\widetilde\theta,\widehat h,\widehat g)
- V(a,x;\widetilde\theta,\widehat h,\widehat g)
\right\}.
$$

The package splits by individual. Each fold learns choice probabilities,
$h$, $g$, the preliminary estimate, and $\lambda$. The other fold evaluates
the corrected moment. The two parameter estimates are averaged. Standard
errors use the individual-clustered fold moments.

For a binary linear semigradient, the corrected estimating equation uses the
paper's Appendix B.5 closed form. Its covariance still uses the backward
function in the fold moments. Other semigradient designs use the backward
recursion throughout. Neural AVI uses the configured AVI function class for
both the forward terms and the backward correction.

## Approximate Value Iteration

`method="neural"` replaces the linear projection with Algorithm 1. Each round
fits a new target

$$
h_{j+1}(a,x)
\approx z(a,x) + \beta h_j(a',x').
$$

The same recursion is applied to $g$. The default function class is a neural
network. `avi_functional_class="gbm"` uses histogram gradient boosting by
default. That default requires scikit-learn. Set `avi_regressor` to a
zero-argument callable that returns a fresh object with `fit` and `predict`
methods to use another regressor.

Neural AVI is useful when the continuation terms are nonlinear in encoded
state features. The semigradient path is faster and remains the default.

## Product Surface

The public `TDCCP` estimator supports the shared structural workflow.

| Capability | Support |
| --- | --- |
| DataFrame and panel input | Yes |
| Supplied transition tensor in `(A, S, S)` orientation | Yes |
| Transition estimation from the fitted panel | Yes |
| Algorithm 2 locally robust inference | Yes |
| Pairs-cluster bootstrap | Yes |
| Encoded-state semigradient | Yes |
| Neural and boosting AVI | Yes |
| Prediction and simulation | Yes |
| Reward and transition counterfactuals | Yes |
| Pickle round trip | Yes |

The transition tensor does not enter the TD parameter equations. The wrapper
stores it for the shared prediction, simulation, and counterfactual methods.

## Choosing TD-CCP

Use TD-CCP when the panel records successor state-action pairs and transition
density estimation is the main obstacle. Use NFXP when the state space and
transition tensor are small enough for repeated Bellman solutions. Use CCP
when a transition model is readily available and Hotz-Miller inversion is the
natural route.

TD-CCP remains a structural estimator. Its reward is linear in known features.
It is not a replacement for neural reward models in the IRL family.

## References

- Adusumilli, K., and Eckardt, D. (2025). "Temporal-Difference Estimation of
  Dynamic Discrete Choice Models." {ref}`reference entry
  <adusumilli-eckardt-2025>`.
- Hotz, V. J., and Miller, R. A. (1993). "Conditional Choice Probabilities and
  the Estimation of Dynamic Models." _Review of Economic Studies_, 60(3),
  497-529. {ref}`reference entry <hotz-miller-1993>`.

```{toctree}
:hidden:

tdccp/quick_start
tdccp/pre_estimation
tdccp/validation
tdccp/counterfactuals
tdccp/rust_bus
```
