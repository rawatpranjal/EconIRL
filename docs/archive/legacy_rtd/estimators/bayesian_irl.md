# Bayesian IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ramachandran and Amir (2007) | Linear | Yes | Posterior (exact) | No |

## What this estimator does

Every other IRL estimator in the library produces a point estimate of the reward parameters with confidence intervals from asymptotic approximations. Ramachandran and Amir (2007) take a different approach. They place a prior distribution over reward parameters and use Metropolis-Hastings MCMC to sample from the posterior. Each step proposes new parameters, solves the MDP, computes the likelihood of the observed behavior under the resulting policy, and accepts or rejects via the Metropolis-Hastings ratio. The posterior mean serves as the point estimate and the posterior standard deviation provides uncertainty quantification that is valid at any sample size without asymptotic assumptions.

## How it works

The posterior over reward parameters given demonstrated behavior is

$$
p(\theta \mid D) \propto p(D \mid \theta) \, p(\theta), \quad p(D \mid \theta) = \prod_{(s,a) \in D} \pi_\theta(a \mid s).
$$

The likelihood requires solving the MDP under $\theta$ to get the softmax policy $\pi_\theta$. Each MCMC step proposes $\theta'$ from a Gaussian centered at the current $\theta$, solves the MDP, and accepts with the Metropolis-Hastings ratio. After discarding burnin samples, the posterior samples provide exact finite-sample uncertainty quantification.

## When to use it

Bayesian IRL is the only estimator in the library with exact finite-sample uncertainty quantification, independent of sample size. With 10 trajectories the posterior is wide and honest. With 1000 trajectories it concentrates and converges to the MLE. The limitation is computational cost. Each MCMC step requires solving the full MDP, making the method tractable only for small state spaces of roughly 100 states or fewer. Convergence diagnostics are essential for verifying that the chain has mixed.

## References

- Ramachandran, D. and Amir, E. (2007). Bayesian Inverse Reinforcement Learning. *IJCAI 2007*.

A primer for this estimator is not yet available.
