# Bayesian IRL

| Category | Citation | Reward | Transitions | SEs | Scales | Transfer |
|----------|----------|--------|-------------|-----|--------|----------|
| Inverse | Ramachandran and Amir (2007) | Linear | Yes | Posterior (exact) | No | No |

## Background

Every other IRL estimator in the library produces a point estimate of the reward parameters. Confidence intervals come from asymptotic approximations, either Hessian-based standard errors or bootstrap. These approximations require large samples to be reliable. With few demonstrations, the Hessian may be poorly conditioned and bootstrap distributions may be noisy. Ramachandran and Amir (2007) took a different approach entirely. They placed a prior distribution over reward parameters and used Metropolis-Hastings MCMC to sample from the posterior. Each MCMC step proposes new parameters, solves the MDP, computes the likelihood of the observed behavior under the resulting policy, and accepts or rejects via the Metropolis-Hastings ratio. The posterior mean serves as the point estimate and the posterior standard deviation provides uncertainty quantification that is valid at any sample size without asymptotic assumptions.

## Key Equations

The posterior over reward parameters theta given demonstrated behavior D is proportional to the likelihood times the prior.

$$
p(\theta \mid D) \propto p(D \mid \theta) \, p(\theta)
$$

The likelihood is computed by solving the MDP under theta to get policy pi, then evaluating the probability of each observed state-action pair.

$$
p(D \mid \theta) = \prod_{(s,a) \in D} \pi_\theta(a \mid s)
$$

## Pseudocode

```
BayesianIRL(D_expert, p, beta, sigma, n_samples, burnin):
  1. Initialize theta from prior
  2. For each MCMC iteration:
     a. Propose theta' ~ N(theta, proposal_sigma^2 I)
     b. Solve MDP under theta' to get policy pi'
     c. Compute log p(D | theta') + log p(theta')
     d. Accept/reject via Metropolis-Hastings ratio
  3. Discard burnin samples
  4. Return posterior mean, posterior std
```

## Strengths and Limitations

Bayesian IRL is the only estimator in the library that provides exact finite-sample uncertainty quantification. The posterior is valid regardless of sample size. With 10 trajectories, the posterior is wide and honest about uncertainty. With 1000 trajectories, the posterior concentrates and converges to the MLE. No other IRL method in the library provides this guarantee.

The limitation is computational cost. Each MCMC step requires solving the full MDP, making the method tractable only for small state spaces (roughly under 100 states). The method also requires choosing a prior, a proposal distribution, and tuning the number of samples and burnin period. Convergence diagnostics (trace plots, effective sample size, acceptance rate) are essential for verifying that the chain has mixed.

## In econirl

The implementation is `BayesianIRLEstimator` in `econirl.contrib.bayesian_irl`. Key parameters are `n_samples` (total MCMC iterations, default 2000), `burnin` (samples to discard, default 500), `proposal_sigma` (Gaussian proposal width, default 0.1), and `prior_sigma` (Gaussian prior width, default 5.0). The `confidence` parameter controls the inverse temperature of the likelihood.

## References

- Ramachandran, D. & Amir, E. (2007). Bayesian Inverse Reinforcement Learning. *IJCAI 2007*, pp. 2586-2591.
- Choi, J. & Kim, K. (2011). MAP Inference for Bayesian Inverse Reinforcement Learning. *NeurIPS 2011*.
