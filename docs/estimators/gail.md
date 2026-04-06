# GAIL

| Category | Citation | Reward | Transitions | SEs | Scales | Transfer |
|----------|----------|--------|-------------|-----|--------|----------|
| Inverse | Ho and Ermon (2016) | Tabular (implicit) | No (adversarial) | Bootstrap only | No | No |

## Background

Before GAIL, inverse reinforcement learning was a two-step process. First recover a reward function from demonstrations, then solve the MDP under that reward to get a policy. Ho and Ermon (2016) observed that if the goal is imitation rather than reward recovery, the intermediate reward step is unnecessary. They framed imitation learning as a GAN: a discriminator learns to distinguish expert state-action pairs from agent state-action pairs, and the agent's policy is updated to fool the discriminator. The resulting policy matches the expert's occupancy measure without ever producing an explicit reward function.

## Key Equations

The discriminator is trained to minimize binary cross-entropy between expert and policy state-action distributions. The reward signal for the policy is derived from the discriminator output.

$$
R(s,a) = -\log(1 - D(s,a))
$$

The policy is updated by solving the MDP under this induced reward via soft value iteration.

## Pseudocode

```
GAIL(D_expert, p, beta, sigma, max_rounds):
  1. Initialize discriminator D(s,a)
  2. For each round:
     a. Compute state-action occupancy under current policy
     b. Update discriminator to classify expert vs policy
     c. Derive reward: R(s,a) = -log(1 - D(s,a))
     d. Update policy via soft value iteration under R
  3. Return policy (no explicit reward)
```

## Strengths and Limitations

GAIL is the most direct path from demonstrations to a matching policy. It does not require feature specification for the reward, does not require computing feature expectations, and converges when the policy's occupancy measure matches the expert's. For practitioners who need only to replicate expert behavior and do not need reward interpretability, GAIL is the simplest adversarial approach.

The core limitation is that GAIL does not produce a transferable reward function. The discriminator is entangled with the training environment's dynamics, so the learned signal does not transfer to new environments. If counterfactual analysis, welfare computation, or deployment under different dynamics is needed, AIRL or MCE-IRL is the better choice. GAIL also inherits the training instability of GANs, with sensitivity to learning rates and discriminator capacity.

Standard errors are available only through bootstrap, which requires re-running the full adversarial training loop for each replicate.

## In econirl

The implementation is `GAILEstimator` in `econirl.contrib.gail`. It uses `TabularDiscriminator` or `LinearDiscriminator` from the adversarial module, and soft value iteration for the generator. Configuration is via `GAILConfig`, which controls discriminator learning rate, number of discriminator steps per round, and convergence tolerance.

## References

- Ho, J. & Ermon, S. (2016). Generative Adversarial Imitation Learning. *NeurIPS 2016*.
- Fu, J., Luo, K., & Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR 2018*.
