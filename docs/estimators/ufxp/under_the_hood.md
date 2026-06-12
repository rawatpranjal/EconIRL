# Under the Hood

## Model

The data are state, action, next-state triples $(s, a, s')$ from a stationary
infinite-horizon dynamic discrete choice model with linear flow utility
$u_\theta(s, a) = \varphi(s, a)^\top \theta$, known discount factor $\beta$,
transition kernels $F_a(s' \mid s)$, and i.i.d. logit taste shocks with scale
$\sigma$. The agent's value function solves the soft Bellman equation, and
choice probabilities are the softmax of choice-specific values.

## First-Order Conditions

Fix a reference action $A$. Under logit shocks, for every state $x$ and action
$a \neq A$,

$$
\sigma \log \frac{P_a(x)}{P_A(x)}
= u_\theta(x, a) - u_\theta(x, A)
+ \beta \sum_{x'} \bigl(F_a(x' \mid x) - F_A(x' \mid x)\bigr) V_P(x'),
$$

where $V_P$ is the value of following the choice probabilities $P$:

$$
V_P = (I - \beta F_P)^{-1}
\Bigl[\textstyle\sum_a P_a \circ \bigl(u_\theta(\cdot, a) - \sigma \log P_a\bigr)\Bigr],
\qquad
F_P = \textstyle\sum_a \operatorname{diag}(P_a) F_a .
$$

Replacing $P$ with the empirical choice probabilities $\hat{P}$ turns these
conditions into estimating equations for $\theta$. The obstacle is $V_P$,
which appears to require a linear solve per candidate $\theta$.

## Unnesting the Fixed Point

The conditions only use $V_P$ through fixed linear functionals $w^\top V_P$.
Each has a dual representation,

$$
w^\top V_P = \lambda^\top u_P,
\qquad
\lambda = w + \beta F_P^\top \lambda,
$$

and the dual $\lambda$ depends only on $\hat{P}$, $\beta$, and the
transitions — not on $\theta$. One matrix factorization of
$(I - \beta F_P^\top)$, computed before the parameter search, removes the
value function from the problem entirely. For linear utility every estimating
equation is then affine in $\theta$, so the estimator is closed form.

## Optimal Weighting

The estimating equations at state $x$ have sampling covariance
$\Gamma(x) \Sigma(x) \Gamma(x)^\top / \eta(x)$, where
$\Sigma(x) = \operatorname{diag}(\hat{P}(x)) - \hat{P}(x)\hat{P}(x)^\top$ is
the multinomial covariance of the estimated choice probabilities,
$\Gamma(x)$ is the Jacobian of the inverse choice-probability map, and
$\eta(x) = N(x)/N$ is the state's sample share. The optimal weights are

$$
z(x) = \Bigl[\Gamma(x)\,\Sigma(x)\,\Gamma(x)^\top / \eta(x)\Bigr]^{-1} G(x),
\qquad
G(x) = \Delta \frac{\partial u_\theta(x)}{\partial \theta}
+ \beta \sum_{x'} \Delta F(x' \mid x)\,
\frac{\partial V_{\hat{P}}(x')}{\partial \theta},
$$

and for linear utility $\partial V_{\hat{P}} / \partial \theta =
(I - \beta F_{\hat{P}})^{-1} \Phi_{\hat{P}}$ is itself independent of
$\theta$, so the weights are fully pre-computable. Theorem 2 of
{ref}`Oguz and Bray (2026) <oguz-bray-2026>` shows the optimally weighted
estimator is as asymptotically efficient as maximum likelihood. Rarely visited
states carry weight proportional to their sample share, and unvisited states
drop out of the conditions entirely.

## Standard Errors

At the optimal weights the asymptotic variance collapses to

$$
\widehat{\operatorname{Var}}(\hat{\theta})
= \Bigl(\sum_x z(x)^\top G(x)\Bigr)^{-1} \Big/\, N,
$$

the efficient moment variance, which the package reports through its standard
asymptotic-inference pipeline.

## Pseudocode

```
estimate empirical choice probabilities P-hat from the panel
factorize (I - beta * F_Phat) once
build the per-state optimal weights z(x)
solve the weighted linear moment system for theta   # closed form
solve the model once at theta-hat                   # policy, value, likelihood
report theta-hat with the efficient moment variance
```

## Implementation Notes

The implementation lives in `econirl.estimation.ufxp`. Choice probabilities
use the same frequency estimator and smoothing conventions as the CCP
estimator; conditions are scored only at states observed at least
`ccp_min_count` times. The random-projection mode (`weights="random"`)
implements the paper's baseline construction with `m` projections and the
same dual trick, and is consistent but less efficient.
