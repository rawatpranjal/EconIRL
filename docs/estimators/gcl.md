# GCL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse (Neural) | Finn et al. (2016) | Neural | Yes | No | Yes |

## Background

Most IRL methods estimate the partition function (the normalizing constant of the trajectory distribution) by solving the full MDP at each step. Guided Cost Learning takes a different approach. Finn et al. (2016) proposed using importance-weighted samples from the current policy to approximate the partition function gradient, which avoids a complete MDP solve per gradient step in the original formulation. The cost function is a neural network that maps state-action pairs to scalar costs, giving the model enough flexibility to capture nonlinear reward structure.

The algorithm alternates between two phases. In the first phase, trajectories are sampled from the current policy and weighted by their importance ratios to form a Monte Carlo estimate of the partition function gradient. In the second phase, the cost network is updated by gradient descent on the difference between demonstration costs and importance-weighted sample costs. After each cost update, the policy is re-solved via soft value iteration on the implied reward. The importance weights are clipped for stability and normalized so that no single trajectory dominates the gradient.

## Key Equations

$$
\nabla_\theta \mathcal{L}_{\mathrm{IOC}} = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta c_\theta(\tau_i^{\mathrm{demo}})
- \sum_{j=1}^{M} \tilde{w}_j \, \nabla_\theta c_\theta(\tau_j^{\mathrm{sample}}),
$$

where

$$
\tilde{w}_j \propto \frac{\exp(-c_\theta(\tau_j))}{q(\tau_j)},
$$

and $q$ is the sampling policy. The weights are normalized and clipped for numerical stability.

## Pseudocode

```
GCL(D_expert, p, beta, sigma, max_iterations):
  1. Initialize neural cost network c_theta, policy pi (uniform)
  2. For each iteration:
     a. Sample M trajectories from pi
     b. Compute importance weights w_j = exp(-c(tau_j)) / q(tau_j)
     c. Normalize and clip weights
     d. Compute gradient: g = E_demo[grad c] - E_weighted[grad c]
     e. Update c_theta via Adam
     f. Solve soft Bellman with reward = -c_theta to get new pi
  3. Return c_theta, pi
```

## Strengths and Limitations

GCL can learn complex nonlinear cost functions that go beyond what linear reward specifications can capture. The importance sampling approach is conceptually elegant and avoids the need to differentiate through the value iteration inner loop. The neural network parameterization lets it scale to environments where hand-crafted features would be insufficient.

The main limitation is training instability. Importance weights can have high variance when the sampling policy diverges from the optimal policy, leading to noisy gradient estimates. The clipping helps but introduces bias. GCL also requires known transition dynamics for the inner soft value iteration step, and it does not produce standard errors or confidence intervals for the learned cost function. For problems where interpretable parameters matter, the neural cost function must be projected onto features post-hoc, losing the statistical guarantees of likelihood-based estimation.

## References

- Finn, C., Levine, S., and Abbeel, P. (2016). Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization. *ICML 2016*.
