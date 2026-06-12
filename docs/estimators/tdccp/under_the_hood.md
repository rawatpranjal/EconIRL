# Under the Hood

## Model

The data are current and successor state-action tuples $(a_t, x_t, a_{t+1}, x_{t+1})$ from
a stationary infinite-horizon dynamic discrete choice model. Flow utility is linear in
known reward features $z(a, x)$ with parameter vector $\theta$, discount factor $\beta$,
and i.i.d. logit taste shocks:

$$u_\theta(a, x) = z(a, x)^\top \theta.$$

## Action-Value Decomposition

The CCP likelihood needs choice-specific continuation values. TD-CCP writes the
Q-function as a sum of two learned objects:

$$\tilde{Q}_\theta(a, x) = h(a, x)^\top \theta + g(a, x),$$

where $h$ accumulates discounted future reward features and $g$ accumulates discounted
future choice-shock corrections. The implied policy is the softmax over these indices:

$$\pi_\theta(a \mid x)
= \frac{\exp\{\tilde{Q}_\theta(a, x)\}}{\sum_{b}\exp\{\tilde{Q}_\theta(b, x)\}}.$$

## TD Fixed-Point Equations

The objects $h$ and $g$ satisfy Bellman-style recursions. The reward-feature accumulation
satisfies

$$h(a, x) = z(a, x) + \beta\,\mathbb{E}\bigl[h(a', x') \mid a, x\bigr].$$

The choice-shock accumulation satisfies

$$g(a, x) = \beta\,\mathbb{E}\bigl[e(a', x') + g(a', x') \mid a, x\bigr],
\qquad e(a, x) = \gamma_{\mathrm{E}} - \log P(a \mid x),$$

where $\gamma_{\mathrm{E}}$ is the Euler-Mascheroni constant and $P$ are the
conditional choice probabilities. Neither recursion requires modeling the transition
density; both are identified from observed successor pairs $(a_t, x_t, a_{t+1}, x_{t+1})$.

## Semigradient Estimation

With basis functions $\phi(a, x)$, the semigradient estimator replaces the Bellman
expectation with a sample average over observed transitions. The closed-form solve for $h$ is

$$\hat{\omega}
= \Bigl[\mathbb{E}_n\bigl\{\phi(a,x)\bigl(\phi(a,x) - \beta\phi(a',x')\bigr)^\top\bigr\}\Bigr]^{-1}
  \mathbb{E}_n\bigl\{\phi(a,x)\,z(a,x)\bigr\},$$

with an analogous equation for $g$ using target $\beta\,e(a', x')$. Once $\hat{\omega}$ is
in hand, the preliminary reward parameter $\theta$ is estimated by a CCP pseudo-log-likelihood.

## Locally Robust Inference

Plug-in pseudo-likelihood standard errors understate uncertainty because they ignore
estimation error in the first-stage objects $h$ and $g$. The locally robust estimator adds
a correction term (the paper's $\lambda$ object) that accounts for that error. With
cross-fitting (Algorithm 2 of the paper), the corrected moment is evaluated on held-out
observations, giving valid inference at parametric rates. Standard errors are clustered
by individual.

## Pseudocode

```text
Input: panel, reward features, discount, basis functions, fold splits
Estimate first-stage CCPs from observed actions
for each fold:
    learn h(a, x) by semigradient TD on reward features
    learn g(a, x) by semigradient TD on log-CCP shock corrections
    solve the preliminary CCP pseudo-likelihood for theta
    estimate the correction recursion lambda
    solve the held-out locally robust moment equation for theta
Average fold estimates; compute standard errors clustered by individual
return theta, standard errors, continuation diagnostics, policy objects
```

## Semigradient and AVI

The semigradient path uses linear basis functions for the continuation terms,
making the recursive-term step a single matrix solve rather than a transition-density
integration. The AVI path (Algorithm 1 of the paper) is more flexible and can use
machine-learning learners for the continuation functions. EconIRL exposes both paths;
the current simulation study uses the semigradient path.

## Paper Alignment

| Paper step | EconIRL implementation |
| --- | --- |
| Estimate CCPs $P(a \mid x)$ | Logit first stage with degree-2 encoded-state features |
| Estimate $h$ | Semigradient TD solve using encoded basis functions |
| Estimate $g$ | Semigradient TD solve using log-CCP shock correction |
| Preliminary parameter estimate | Fold-specific CCP pseudo-likelihood solve |
| Correction recursion | Fold-specific backward recursion ($\lambda$) |
| Locally robust moment | Held-out fold solve using quantities learned on the other fold |
| Standard errors | Fold covariance averaged, clustered by individual |

## Implementation Notes

The implementation lives in `econirl.estimation.td_ccp`. The semigradient path supports
polynomial, encoded-state, and tabular bases; conditions are scored only at state-action
pairs with observed support. The current simulation study uses an encoded two-dimensional
basis with a logit first-stage CCP and Algorithm 2 cross-fitting.

| Setting | Value |
| --- | --- |
| Method | `semigradient` |
| Basis | `encoded`, degree 2 |
| CCP model | `logit`, degree 2 |
| Cross-fitting | `True`, split by individual |
| Robust standard errors | `True`, locally robust covariance clustered by individual |
| Policy iterations | `1` |
| Outer optimizer tolerance | `1e-7` |
