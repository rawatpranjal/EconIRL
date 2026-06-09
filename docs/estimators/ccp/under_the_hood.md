# Under the Hood

CCP replaces repeated Bellman solves with first-stage policy inversion. The
estimator starts from empirical conditional choice probabilities, converts
them into continuation-value terms, and maximizes a pseudo likelihood over
augmented features.

## Model

The observed data are state, action, and next-state trajectories.

$$
(s_{it}, a_{it}, s_{i,t+1})
$$

The flow payoff is linear in known features.

$$
u_\theta(s, a) = \phi(s, a)^\top \theta
$$

CCP first estimates the observed policy.

```text
pi_hat(a | s) = count(s, a) / count(s)
```

The estimator then computes the policy-weighted transition matrix.

```text
F_pi(s, s_next) = sum_a pi_hat(a | s) P_a(s, s_next)
```

With logit shocks, the Hotz-Miller correction is the Euler constant minus the
log CCP.

```text
e_pi(s, a) = gamma_euler - log pi_hat(a | s)
```

For linear rewards, the continuation value separates into a term that
multiplies the structural parameters and a term that comes from the logit
shock correction.

```text
W_phi = inv(I - beta F_pi) sum_a pi_hat(a | s) phi(s, a)
W_e = inv(I - beta F_pi) sum_a pi_hat(a | s) e_pi(s, a)
```

The choice-specific value in each pseudo-likelihood step is an augmented
feature logit.

```text
Q_tilde(s, a) = phi(s, a) theta
             + beta E[W_phi(s_next)] theta
             + beta E[W_e(s_next)]
```

NPL repeats two steps. Given a policy, estimate parameters by logit
pseudo-likelihood. Given parameters, update the policy from the implied
choice-specific values.

## Algorithm Sketch

The estimation loop first estimates transitions or accepts a supplied
transition tensor. It then estimates empirical CCPs, builds the
policy-weighted transition matrix, computes Hotz-Miller continuation terms,
maximizes the augmented-feature logit pseudo likelihood, and updates the
policy for the next NPL step. The fitted object returns structural parameters,
standard errors, policy, value function, and diagnostics.

The frequency estimator uses float64 arrays so high-discount problems avoid
JAX dtype promotion warnings. CCP probabilities are clamped before the log
correction so exact zero empirical frequencies cannot create infinities.

The linear pseudo-likelihood kernel is JIT compiled once. The augmented
features are passed as dynamic arguments, so the same compiled kernel is reused
across NPL iterations.

The returned `value_` follows the package soft-Bellman convention. CCP uses the
Euler-constant correction internally for inversion, but final diagnostics
evaluate the recovered policy under the recovered structural reward.

## Score Calculation

For linear rewards, each NPL step is a multinomial logit over augmented
features. The score has the same logit form as NFXP, but the derivatives are
taken with respect to the CCP-augmented choice-specific value rather than a
newly solved Bellman fixed point at every likelihood evaluation.

Standard errors are computed from the full Bellman-constrained likelihood
Hessian and per-observation gradients when requested. This keeps the fitted
summary compatible with the shared inference interface used by NFXP.
