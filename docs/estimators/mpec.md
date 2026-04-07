# MPEC

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Su and Judd (2012) | Linear | Yes | Analytical (MLE) | No |

## What this estimator does

NFXP nests the Bellman solve inside the optimizer: at each candidate $\theta$, it runs the contraction $V = T(V;\theta)$ to convergence, then evaluates the likelihood. MPEC eliminates this inner loop by treating $V$ as an explicit decision variable and imposing the Bellman equation $V = T(V;\theta)$ as an equality constraint. The outer optimizer sees a single constrained problem over $(\theta, V) \in \mathbb{R}^{p+n}$ and runs zero inner iterations.

At $\beta = 0.9999$ on the 90-state Rust bus, MPEC-SQP converges in 0.6 seconds versus 4.1 seconds for NFXP-SA and 7.4 seconds for NFXP-NK, recovering identical parameters (OC = 0.001233, RC = 3.0295).

At convergence the Bellman constraint is satisfied to machine precision, so MPEC and NFXP solve the same MLE. Standard errors use the implicit function theorem at the common MLE fixed point.

## How it works

The implementation uses scipy SLSQP with JAX-computed gradients. The Bellman residual $V - T(V;\theta)$ is enforced as an equality constraint. JAX JIT-compiled functions supply the objective gradient and constraint Jacobian; scipy SLSQP handles the outer SQP loop.

$$
\min_{\theta, V} \; -\mathcal{L}(\theta, V) \quad \text{subject to} \quad V = T(V;\theta).
$$

Initialisation: value iteration at the starting $\theta_0$ supplies a feasible $V_0$, giving the optimizer a feasible starting point.

## When to use it

Use MPEC when the inner Bellman solve is the bottleneck and the state space fits in memory. MPEC is faster than NFXP-NK at all discount factors we tested because it avoids the inner loop entirely. The solver adds no tuning parameters beyond `outer_max_iter` and `tol`.

## Quick start

```python
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.preferences.linear import LinearUtility

env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
panel = env.generate_panel(n_individuals=500, n_periods=100)
utility = LinearUtility(env.feature_matrix, env.parameter_names)

mpec = MPECEstimator(MPECConfig(solver="sqp"))
result = mpec.estimate(panel, utility, env.problem_spec, env.transition_matrices)
print(result.parameters)   # [OC, RC]
# Expected: ~[0.001233, 3.0295] in ~0.6s
```

## References

- Su, C.-L. and Judd, K. L. (2012). Constrained Optimization Approaches to Estimation of Structural Models. *Econometrica*, 80(5), 2213-2230.
- Iskhakov, F., Rust, J., and Schjerning, B. (2016). Comment on "Constrained Optimization Approaches to Estimation of Structural Models." *Econometrica*, 84(1), 365-370.

Full derivation, algorithm pseudocode, and simulation results: [MPEC primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/mpec/mpec.pdf).
