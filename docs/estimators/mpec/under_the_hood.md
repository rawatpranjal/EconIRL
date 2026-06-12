# Under the Hood

The model, the constrained program, the standard-error identity, and the
consistency argument live on the [MPEC overview](../mpec.md). This page is the
operational dive: how the constrained problem is actually solved.

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

The joint variable is $x = (\theta, V)$ and the equality constraint has one row
per state. Objective gradients and constraint Jacobians come from JAX, not finite
differences. The value vector is initialized at the Bellman fixed point of the
starting $\theta$, so the optimizer begins feasible and the SQP steps stay near
the constraint surface. No Bellman fixed point is solved inside the objective; the
constraint carries it.

## Implementation Notes

Standard errors use the implicit-score identity stated on the overview page: at
the constrained optimum, per-observation score contributions are computed from
$\partial V / \partial \theta$ after convergence, and the robust covariance is
their outer product. The fitted summary exposes
`metadata["final_constraint_violation"]`; gate on it alongside the convergence
flag, since a high likelihood with a violated constraint is not a solution.

The estimator lives in `econirl.estimation.mpec`. Use `MPECConfig(solver="sqp")`
for the recommended SLSQP path. The `augmented_lagrangian` solver is retained for
comparison but is less reliable at high discount factors.
