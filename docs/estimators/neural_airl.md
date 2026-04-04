# Neural AIRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse (Neural) | Fu et al. (2018) | Neural | No (adversarial) | No | Yes |

## Background

The tabular AIRL estimator in econirl works on discrete state spaces where the reward and shaping functions can be stored as lookup tables. Neural AIRL extends this to continuous or high-dimensional state spaces by replacing the tabular reward and shaping functions with neural networks. It also adds context conditioning, so the reward and policy networks can depend on observable individual-level covariates in addition to the state.

The discriminator architecture follows Fu et al. (2018). The discriminator output is decomposed into a reward network g(s,a,ctx) and a shaping network h(s,ctx), with the discriminator logit equal to g + gamma*h(s') minus h(s) minus log pi(a given s). This decomposition guarantees that the recovered reward g transfers across environments with different dynamics, because the shaping term absorbs everything that depends on the transition structure. The policy network is trained adversarially to fool the discriminator, while the discriminator learns to distinguish expert transitions from policy-generated ones.

## Key Equations

$$
D_\phi(s,a,s') = \frac{\exp(f_\phi(s,a,s'))}{\exp(f_\phi(s,a,s')) + \pi_\psi(a \mid s)},
$$

where

$$
f_\phi(s,a,s') = g_\phi(s,a) + \beta \, h_\phi(s') - h_\phi(s).
$$

The discriminator is trained to maximize the binary cross-entropy between expert and policy-generated transitions. The policy is trained to minimize its negative log-likelihood weighted by the discriminator reward signal.

## Pseudocode

```
NeuralAIRL(D_expert, n_actions, beta, max_epochs):
  1. Initialize reward network g, shaping network h, policy network pi
  2. For each epoch:
     a. For each minibatch of expert transitions (s, a, s'):
        i.  Sample policy actions from pi(.|s)
        ii. Compute discriminator logits for expert and policy transitions
        iii. Update (g, h) to classify expert as 1, policy as 0
     b. Update pi to maximize discriminator reward signal
  3. Extract reward: R(s,a) = g(s,a)
  4. Optionally project R onto features for structural parameters
  5. Return g, h, pi
```

## Strengths and Limitations

Neural AIRL scales to continuous state spaces and high-dimensional observations where tabular methods cannot be applied. The context conditioning lets it learn reward functions that vary with observable individual characteristics, which is important for demand estimation and personalized policy design. The adversarial formulation does not require a transition model, making it applicable when the dynamics are unknown or too complex to specify.

The limitations are significant. Adversarial training is unstable and sensitive to hyperparameters for the learning rates, batch size, and the ratio of discriminator to policy updates. The method does not produce standard errors, so formal inference is not possible without additional bootstrapping. The recovered reward is identified only up to shaping (and up to a constant within each equivalence class), so comparing reward magnitudes across different training runs requires careful normalization. For tabular discrete-choice problems where transitions are known, the tabular AIRL or MCE-IRL estimators are simpler and more stable.

## References

- Fu, J., Luo, K., and Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR 2018*.
- Ho, J. and Ermon, S. (2016). Generative Adversarial Imitation Learning. *NeurIPS 2016*.
