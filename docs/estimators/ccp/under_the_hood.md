# Under The Hood

CCP starts from the empirical policy. For each state and action, EconIRL counts
observed choices and applies smoothing.

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

## Implementation Notes

The frequency estimator uses float64 arrays so high-discount problems avoid
JAX dtype promotion warnings. CCP probabilities are clamped before the log
correction so exact zero empirical frequencies cannot create infinities.

The linear pseudo-likelihood kernel is JIT compiled once. The augmented
features are passed as dynamic arguments, so the same compiled kernel is reused
across NPL iterations.

The returned `value_` follows the package soft-Bellman convention. CCP uses the
Euler-constant correction internally for inversion, but final diagnostics
evaluate the recovered policy under the recovered structural reward.
