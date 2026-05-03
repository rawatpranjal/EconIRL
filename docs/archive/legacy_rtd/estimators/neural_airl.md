# Neural AIRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse (Neural) | Fu et al. (2018) | Neural | No (adversarial) | No | Yes |

## What this estimator does

The tabular AIRL estimator works on discrete state spaces where the reward and shaping functions are stored as lookup tables. Neural AIRL extends this to continuous or high-dimensional state spaces by replacing the tabular functions with neural networks. It also adds context conditioning, so the reward and policy networks can depend on observable individual-level covariates in addition to the state.

The discriminator architecture follows Fu et al. (2018). The discriminator logit decomposes into a reward network $g(s,a,\text{ctx})$ and a shaping network $h(s,\text{ctx})$, with the same potential-based structure as tabular AIRL. This guarantees that the recovered reward transfers across environments with different dynamics, under the conditions of the disentanglement theorem.

## How it works

The algorithm is identical to tabular AIRL but with neural parameterization. The discriminator is trained via binary cross-entropy on expert versus policy transitions, and the policy is updated via PPO to maximize the discriminator reward signal. The discriminator logit is

$$
f_\phi(s,a,s') = g_\phi(s,a) + \beta h_\phi(s') - h_\phi(s).
$$

Standard errors are not available analytically. The recovered reward is identified only up to shaping and a constant within each equivalence class.

## When to use it

Neural AIRL is the right choice for continuous state spaces where tabular AIRL cannot operate. The context conditioning makes it suitable for demand estimation and personalized policy design where rewards vary with observable individual characteristics. For tabular discrete-choice problems where transitions are known, tabular AIRL or MCE-IRL are simpler and more stable.

## References

- Fu, J., Luo, K., and Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR 2018*.

The full derivation and theory are in the [AIRL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/airl.pdf). Neural AIRL extends the tabular version with neural parameterization and context conditioning.
