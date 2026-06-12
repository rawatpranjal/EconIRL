# Under the Hood

## Model

The data are state, action, next-state triples $(s_{it}, a_{it}, s_{i,t+1})$ from a
stationary infinite-horizon dynamic discrete choice model with linear flow utility
$u_\theta(s, a) = \varphi(s, a)^\top \theta$, known discount factor $\beta$,
transition kernels $P_a(s' \mid s)$, and i.i.d. logit taste shocks with scale $\sigma$.

The transition law is estimated from the panel or supplied directly. The first step
is to estimate the empirical policy from state-action counts:

$$
\hat\pi(a \mid s) = \frac{\#(s, a)}{\#(s)}.
$$

## Hotz-Miller Inversion

Given a policy $\hat\pi$, the policy-weighted transition matrix is:

$$
F_{\hat\pi}(s, s') = \sum_a \hat\pi(a \mid s)\, P_a(s, s').
$$

Under logit shocks, the emax correction for each state-action pair is:

$$
e_{\hat\pi}(s, a) = \gamma_{\mathrm{Euler}} - \log \hat\pi(a \mid s),
$$

where $\gamma_{\mathrm{Euler}} \approx 0.5772$. The integrated value under $\hat\pi$ is:

$$
\bar{V}_{\hat\pi} = (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)
\bigl\{u_\theta(s, a) + e_{\hat\pi}(s, a)\bigr\}.
$$

For linear rewards this separates into:

$$
\bar{V}_{\hat\pi}(s) = W_\varphi(s)^\top \theta + W_e(s),
$$

where:

$$
W_\varphi = (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)\, \varphi(s, a),
\qquad
W_e = (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)\, e_{\hat\pi}(s, a).
$$

$W_\varphi$ and $W_e$ depend on $\hat\pi$ and the transitions but not on $\theta$.
One factorization of $(I - \beta F_{\hat\pi})$ per NPL step replaces the per-evaluation
Bellman solve that NFXP pays at every likelihood call.

## Pseudo-Likelihood

The choice-specific pseudo-value combines the flow utility with the discounted
continuation term from the inversion:

$$
\tilde{Q}_\theta(s, a; \hat\pi) = \varphi(s, a)^\top \theta
+ \beta \sum_{s'} P_a(s, s')\bigl\{W_\varphi(s')^\top \theta + W_e(s')\bigr\}.
$$

The K-step NPL estimator maximizes the pseudo log-likelihood at each iteration:

$$
\hat\theta_K = \arg\max_\theta \sum_{i,t} \log \frac{
  \exp\!\bigl(\tilde{Q}_\theta(s_{it}, a_{it}; \hat\pi^{K-1}) / \sigma\bigr)
}{
  \sum_b \exp\!\bigl(\tilde{Q}_\theta(s_{it}, b; \hat\pi^{K-1}) / \sigma\bigr)
}.
$$

After each step the policy is updated from $\hat\theta_K$ and the augmented features
are rebuilt. Setting $K = 1$ gives the one-step Hotz-Miller estimator; iterating
until convergence is the NPL fixed point.

## Pseudocode

```
estimate empirical CCPs pi-hat(a | s) from state-action counts
for each NPL iteration:
    build F_pi from pi-hat and the transition tensor
    factorize (I - beta * F_pi)   # one matrix solve per step
    compute W_phi and W_e
    build augmented logit features from phi, W_phi, and W_e
    maximize the pseudo log-likelihood over theta
    update pi-hat from the implied choice-specific values
return theta, policy, value function, standard errors, diagnostics
```

## Implementation Notes

The frequency estimator uses float64 arrays so high-discount problems avoid JAX dtype
promotion warnings. CCP probabilities are clamped before the log correction so exact
zero empirical frequencies cannot create infinities.

The logit pseudo-likelihood kernel is JIT-compiled once at module level. The augmented
features are passed as dynamic arguments, so the same compiled kernel is reused across
all NPL iterations without recompilation.

The returned `value_` follows the package soft-Bellman convention (log-sum-exp without
the Euler-gamma constant). CCP uses the Euler-constant correction internally for the
Hotz-Miller inversion step, but final diagnostics evaluate the recovered policy under
the recovered structural reward in the standard convention.

Standard errors are computed from the full Bellman-constrained likelihood Hessian and
per-observation gradients. This keeps the fitted summary compatible with the shared
inference interface used by NFXP.
