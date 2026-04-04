# MaxEnt IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ziebart et al. (2008) | Linear (state-only) | Yes | Bootstrap | No |

## Background

Maximum Entropy IRL was the first principled probabilistic formulation of inverse reinforcement learning. Ziebart et al. (2008) showed that among all policies consistent with the observed feature expectations, the maximum entropy policy is the least committed and therefore the most robust. The algorithm finds reward weights that match the empirical feature expectations of the demonstrations while maximizing the entropy of the induced trajectory distribution.

The key insight is that the gradient of the log-likelihood takes a clean form. It is simply the difference between the empirical feature expectations from the demonstrations and the expected features under the current policy. This allows efficient optimization via L-BFGS. The reward is linear in state features, so the recovered reward function is fully interpretable. However, the state visitation computation does not account for causal structure in the way that MCE-IRL does. In stochastic environments, MCE-IRL (Ziebart 2010) is strictly better because it correctly handles the causal entropy over action sequences rather than trajectory distributions. MaxEnt IRL remains useful as a baseline and for understanding the historical development of the field.

## Key Equations

$$
\hat\theta_{\mathrm{MaxEnt}} = \arg\max_\theta \; \mathbb{E}_{\mathcal{D}}[R(s; \theta)] - \log Z(\theta),
$$

with gradient

$$
\nabla_\theta \mathcal{L} = \bar\phi_{\mathrm{expert}} - \mathbb{E}_{\pi_\theta}[\phi(s)],
$$

where $\bar\phi_{\mathrm{expert}}$ is the empirical feature mean over demonstrations and $\mathbb{E}_{\pi_\theta}[\phi(s)]$ is the expected feature under the current policy.

## Pseudocode

```
MaxEntIRL(D, phi, p, beta, sigma):
  1. Compute empirical feature expectations: phi_bar = mean(phi(s)) over D
  2. Initialize theta
  3. Repeat until convergence:
     a. Compute reward: R(s) = theta . phi(s)
     b. Solve soft Bellman for V and pi
     c. Compute expected features under pi
     d. Gradient: g = phi_bar - E_pi[phi(s)]
     e. Update theta via L-BFGS
  4. Standard errors via bootstrap
  5. Return theta, SEs
```

## Strengths and Limitations

MaxEnt IRL produces interpretable linear reward weights with a clear probabilistic interpretation. The gradient has a simple closed form, making optimization straightforward. The algorithm is well understood theoretically and has been the workhorse of applied IRL for over a decade.

The main limitation is that MaxEnt IRL does not account for causal structure in the state visitation computation. It treats the trajectory distribution as if actions at each timestep do not causally affect future states, which leads to incorrect state visitation frequencies in stochastic environments. MCE-IRL fixes this by using causal entropy, which correctly marginalizes over the randomness in transitions. MaxEnt IRL also restricts rewards to be state-only, meaning the cost of an action cannot depend on which action is chosen. For action-dependent rewards, use MCE-IRL with action-dependent features instead.

## References

- Ziebart, B. D., Maas, A. L., Bagnell, J. A., and Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning. *AAAI Conference on Artificial Intelligence*.
- Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Carnegie Mellon University.
