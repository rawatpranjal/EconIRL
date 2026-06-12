# Under the Hood

## Model

The data are state, action, next-state triples $(s, a, s')$ from a stationary
infinite-horizon dynamic discrete choice model with linear flow utility
$u_\theta(s, a) = \phi(s, a)^\top \theta$, known discount factor $\beta$,
transition kernels $P_a(s' \mid s)$ in action-state-next-state orientation,
and i.i.d. logit taste shocks with scale $\sigma$.

The choice-specific value combines the flow payoff with the discounted
continuation:

$$
Q_\theta(s, a; V)
= u_\theta(s, a)
  + \beta \sum_{s'} P_a(s, s') V(s').
$$

The soft Bellman operator maps any value vector to its log-sum-exp:

$$
T_\theta V(s)
= \sigma \log \sum_a \exp\!\Bigl(Q_\theta(s, a; V) / \sigma\Bigr).
$$

The logit choice probability at any $({\theta}, V)$ pair is:

$$
\pi_{\theta,V}(a \mid s)
= \frac{\exp\bigl(Q_\theta(s, a; V) / \sigma\bigr)}
       {\sum_b \exp\bigl(Q_\theta(s, b; V) / \sigma\bigr)}.
$$

## Constrained Estimator

NFXP solves for $V_\theta = T_\theta V_\theta$ inside each likelihood
evaluation, hiding the fixed point from the optimizer. MPEC keeps $V$ as
an explicit optimization variable and writes the Bellman condition as a
per-state equality constraint:

$$
(\hat\theta,\, \hat V)
= \arg\max_{\theta,\, V}
  \sum_{i,t} \log \pi_{\theta,V}(a_{it} \mid s_{it})
\quad \text{s.t.} \quad
V - T_\theta V = 0.
$$

At any feasible point the constraint forces $V = V_\theta$, so MPEC and NFXP
evaluate the same dynamic discrete choice likelihood. The difference is
numerical geometry, not the structural target. The fitted summary exposes the
final Bellman residual directly; NFXP hides it inside each inner solve.

The implementation uses SLSQP with JAX-supplied objective gradients and
Bellman constraint Jacobians. The value vector is initialized at the Bellman
fixed point of the starting $\theta$, giving the optimizer a near-feasible
start.

## Pseudocode

```text
initialize theta; solve V = T_theta(V) for the starting value vector
define the constraint c(theta, V) = V - T_theta(V)
while the constrained optimizer has not stopped:
    compute Q(s, a) from theta, V, transitions, and beta
    compute log pi(a | s) by the log-softmax rule
    evaluate the conditional log likelihood
    evaluate c(theta, V) and its Jacobian via JAX
    update theta and V with SLSQP
return theta, V, policy, standard errors, and constraint diagnostics
```

## Implementation Notes

Standard errors follow the same implicit score logic as NFXP. At the
constrained optimum, the sensitivity of the value function to the reward
parameters satisfies

$$
(I - \beta P_\pi)\,\frac{\partial V}{\partial\theta}
= \sum_a \pi(a \mid s)\,\phi(s, a),
$$

where $P_\pi$ is the policy-weighted transition matrix. Per-observation score
contributions are computed from this expression after convergence. The
implementation gates on the final Bellman constraint violation as a numerical
check alongside the standard convergence flag.

The estimator lives in `econirl.estimation.mpec`. Use
`MPECConfig(solver="sqp")` for the recommended SLSQP path. The
`augmented_lagrangian` solver is retained for comparison but is less reliable
at high discount factors.
