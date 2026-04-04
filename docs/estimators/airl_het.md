# AIRL-Het

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse (Neural, Heterogeneous) | Lee, Sudhir, and Wang (2026) | Neural (per-segment) | No (adversarial) | No | Yes |

## Background

Standard AIRL assumes all individuals in the data share the same reward function. In many applications this is unrealistic. Consumers have different tastes, drivers have different risk tolerances, and patients have different treatment preferences. Lee, Sudhir, and Wang (2026) extended AIRL with an EM algorithm that discovers latent consumer segments, each with its own reward function and policy. The E-step computes posterior segment membership probabilities from trajectory likelihoods under each segment's policy. The M-step runs a weighted AIRL for each segment, where expert transitions are weighted by the posterior probability that the trajectory belongs to that segment.

The second innovation is anchor identification. Standard AIRL recovers rewards only up to potential-based shaping, which means the recovered reward could be a shaped perturbation of the true structural reward rather than the true reward itself. Lee, Sudhir, and Wang resolve this by designating an exit action with zero reward and an absorbing state with zero continuation value. These anchor constraints pin down the reward and shaping functions uniquely, so that the recovered g equals the true structural reward r and the recovered h equals the true value function V. This is critical for counterfactual analysis, where the level of the reward matters and not just its ordinal ranking.

## Key Equations

The E-step computes posterior segment probabilities for each trajectory $i$ and segment $k$,

$$
\gamma_{ik} \propto \pi_k \prod_{t} \pi_{\theta_k}(a_t \mid s_t),
$$

where $\pi_k$ is the segment prior and $\pi_{\theta_k}$ is the segment-specific policy.

The M-step updates segment $k$'s reward by running weighted AIRL with the discriminator loss

$$
\mathcal{L}_k = \sum_i \gamma_{ik} \sum_t \log D_k(s_t, a_t, s_{t+1}) + \sum_j \sum_t \log(1 - D_k(\tilde{s}_t, \tilde{a}_t, \tilde{s}_{t+1})),
$$

subject to anchor constraints $r(s, a_{\mathrm{exit}}) = 0$ for all $s$ and $r(s_{\mathrm{absorb}}, a) = 0$ for all $a$.

## Pseudocode

```
AIRL_Het(D, transitions, beta, K, exit_action, absorbing_state):
  1. Initialize K segment-specific rewards, policies, and priors
  2. For each EM iteration:
     --- E-step ---
     a. For each trajectory i and segment k:
        Compute trajectory log-likelihood under pi_k
     b. Compute posteriors gamma[i,k] via Bayes rule with log-sum-exp
     c. Apply within-individual consistency smoothing
     d. Update segment priors
     --- M-step ---
     e. For each segment k:
        Run weighted AIRL with expert weights gamma[.,k]
        Enforce anchor constraints on reward
        Re-solve soft Bellman for updated policy and value
     --- Convergence ---
     f. Compute mixture log-likelihood
     g. If relative LL change < tol, stop
  3. Return segment rewards, policies, posteriors, priors
```

## Strengths and Limitations

AIRL-Het is the only estimator in the package that jointly handles unobserved heterogeneity and reward recovery without a parametric utility specification. The EM framework is flexible and can accommodate any number of segments. The anchor identification guarantees that the recovered rewards are the true structural rewards rather than shaped equivalents, enabling meaningful counterfactual welfare analysis across segments. The within-individual consistency constraint encourages the same person to be assigned to the same segment across multiple choice episodes, which regularizes the posterior and reduces label switching.

The main limitation is computational cost. Each EM iteration runs K separate AIRL inner loops, and each AIRL loop involves multiple rounds of discriminator and policy updates. Convergence can be slow when the number of segments is large or the segments are poorly separated. The method does not produce standard errors for the segment-specific reward parameters. Anchor identification requires that the exit action and absorbing state exist in the data and are correctly specified by the researcher. Misspecification of the anchors will bias the recovered rewards.

## References

- Lee, P. S., Sudhir, K., and Wang, T. (2026). Modeling Serialized Content Consumption: Adversarial IRL for Dynamic Discrete Choice. Yale School of Management Working Paper.
- Fu, J., Luo, K., and Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR 2018*.
