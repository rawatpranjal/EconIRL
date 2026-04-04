# MPEC

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Su and Judd (2012) | Linear | Yes | Analytical (Schur complement) | No |

## Background

NFXP nests a fixed-point solve inside every optimizer step, which means solving the Bellman equation from scratch each time the candidate parameters change. Su and Judd (2012) proposed a different formulation that avoids the nested solve entirely. MPEC treats the value function V as an explicit set of decision variables optimized jointly with the structural parameters theta. The Bellman equation becomes a constraint rather than a subroutine. The optimizer sees one large but flat problem instead of two nested loops.

The constrained problem is solved via an augmented Lagrangian method. Each outer iteration minimizes a penalized objective over the joint (theta, V) space using L-BFGS-B, then tightens the penalty on the Bellman constraint violation. When the penalty grows large enough, the optimizer is forced to satisfy the Bellman equation at the solution, recovering the same maximum likelihood estimates that NFXP would produce.

## Key Equations

$$
\hat\theta_{\mathrm{MPEC}} = \arg\min_{\theta, V} \; -\sum_{(s,a) \in \mathcal{D}} \log \pi_{\theta,V}(a \mid s)
\quad \text{s.t.} \quad V = T_\sigma(V; \theta),
$$

where $T_\sigma$ is the soft Bellman operator. The augmented Lagrangian relaxation is

$$
\mathcal{L}_\rho(\theta, V, \lambda) = -\mathcal{LL}(\theta, V)
+ \lambda^\top [V - T_\sigma(V; \theta)]
+ \frac{\rho}{2} \|V - T_\sigma(V; \theta)\|^2.
$$

## Pseudocode

```
MPEC(D, r_theta, p, beta, sigma):
  1. Initialize theta, solve Bellman at theta to get V_0
  2. Set lambda = 0, rho = rho_0
  3. Repeat until constraint violation < tol:
     a. Minimize L_rho(theta, V, lambda) over (theta, V) via L-BFGS-B
     b. Compute constraint violation c = V - T(V; theta)
     c. Update multipliers: lambda <- lambda + rho * c
     d. Increase penalty: rho <- min(rho * growth, rho_max)
  4. Compute standard errors from analytical score at (theta*, V*)
  5. Return theta*, V*, SEs
```

## Strengths and Limitations

MPEC delivers the same statistically efficient maximum likelihood estimates as NFXP because the Bellman equation holds exactly at the solution. The joint formulation can be faster than NFXP when the state space is moderate, because the optimizer exploits sparsity in the Bellman constraint Jacobian rather than iterating the contraction mapping to convergence at every step. Standard errors use the same analytical score formula as NFXP once V satisfies the equilibrium constraint.

The drawback is that the decision variable space grows by the number of states, since V adds one variable per state on top of the structural parameters. This makes the L-BFGS-B subproblems more expensive as the state space grows. The augmented Lagrangian also introduces tuning parameters for the penalty schedule, and poor choices can slow convergence or cause numerical instability. For small state spaces NFXP is simpler and equally fast. MPEC is most useful when the Bellman contraction is slow to converge (high discount factors near one) but the state space remains tractable for joint optimization.

## References

- Su, C.-L. and Judd, K. L. (2012). Constrained Optimization Approaches to Estimation of Structural Models. *Econometrica*, 80(5), 2213-2230.
- Iskhakov, F., Lee, J., Rust, J., Schjerning, B., and Seo, K. (2016). Comment on "Constrained Optimization Approaches to Estimation of Structural Models." *Econometrica*, 84(1), 365-370.
