# GAIL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ho and Ermon (2016) | None | No (adversarial) | No | Yes |

## What this estimator does

Traditional IRL solves a two-step problem. First recover the reward from expert demonstrations, then find the optimal policy for that reward. Ho and Ermon (2016) observe that if the goal is only to match the expert's behavior and not to understand the reward, the two steps can be collapsed into one. Their key insight is that IRL is dual to occupancy measure matching. Two policies produce the same behavior if and only if they have the same occupancy measure. So matching the expert's behavior reduces to matching its occupancy measure, which can be done directly via a GAN without ever recovering a reward.

The tradeoff is clear. GAIL is faster and simpler than IRL but does not produce a reward function. This means GAIL cannot answer counterfactual questions because there is no structural primitive to re-optimize. The discriminator $D(s,a)$ is not a reward function. It is a classifier that distinguishes expert from policy transitions. At the optimum $D = 1/2$ everywhere and carries no structural information. GAIL serves as a strong imitation learning baseline that exploits MDP structure without requiring feature specification.

## How it works

The estimator solves the minimax problem

$$
\min_\pi \max_D \; \mathbb{E}_{\rho_\pi}[\log D(s,a)] + \mathbb{E}_{\rho_E}[\log(1 - D(s,a))] - \lambda \mathcal{H}(\pi),
$$

where the inner maximization over $D$ computes the Jensen-Shannon divergence between learned and expert occupancy measures, and the outer minimization finds the policy whose occupancy is closest to the expert's. The reward signal $\tilde{r} = -\log(1 - D)$ encourages the policy to generate transitions that fool the discriminator. Standard errors are available only via bootstrap because there is no structural likelihood.

## When to use it

GAIL is the right choice when the goal is policy matching rather than reward recovery, and when no feature specification is available. It scales to high-dimensional continuous state-action spaces via neural discriminators and PPO. For structural estimation and counterfactual analysis, AIRL or MCE-IRL are needed because they recover a reward function. GAIL's role in the econirl pipeline is as a strong behavioral baseline that exploits MDP structure.

## References

- Ho, J. and Ermon, S. (2016). Generative Adversarial Imitation Learning. *NeurIPS 2016*.

The full derivation, algorithm, and simulation results are in the [GAIL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/gail.pdf).
