## Estimator: AIRL (Adversarial Inverse Reinforcement Learning)
## Paper(s): Fu, Luo, Levine 2018 "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning," ICLR. At `papers/foundational/2018_adversarial_IRL.pdf` (and `.md`).
## Code: `src/econirl/estimation/adversarial/airl.py`

### Loss / objective

- Paper formula (Fu et al. 2018, eq. 4 and Theorem 5.1): the discriminator has the structured form

  ```
  D(s, a, s') = exp(f(s, a, s')) / (exp(f(s, a, s')) + pi(a | s))
  ```

  with the disentangled reward decomposition

  ```
  f(s, a, s') = g(s, a) + gamma * h(s') - h(s)
  ```

  where `g` is the recovered reward and `h` is a learned shaping potential. The discriminator is trained against `pi`, the learned policy is updated via standard policy gradient (PPO/TRPO in the original; the package uses Adam-on-the-soft-Bellman as a simpler, DDC-friendly substitute).

  The discriminator loss is binary cross-entropy:

  ```
  L_D = -E_E[log D(s, a, s')] - E_pi[log (1 - D(s, a, s'))]
  ```

  with the policy update minimizing the cross-entropy from the other side.

- Code implementation: `airl.py:91-339`. The `AIRLEstimator` constructs a `TabularDiscriminator` or `LinearDiscriminator` (per `adversarial/discriminator.py`) with the structured `f = g + gamma h(s') - h(s)` decomposition. The discriminator and policy are trained alternately for `AIRLConfig.num_outer_iterations` rounds. The loss is BCE on demonstration vs policy samples. State-only `g(s)` is the default (controlled by `AIRLConfig.state_only_reward`); state-action `g(s, a)` is selectable.

- Match: **yes** for the loss form and the disentanglement decomposition. The policy-update path is a soft-Bellman update rather than PPO, which is appropriate for DDC and explicitly noted in the module docstring.

### Gradient

- Paper formula: standard adversarial gradients (BCE on the discriminator side, soft policy gradient on the policy side).

- Code implementation: JAX autodiff. The discriminator uses Optax Adam; the policy update uses the soft Bellman closed form when feasible (tabular) and gradient descent otherwise.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: alternating discriminator and policy updates. No explicit Bellman fixed-point; the policy is updated via RL.

- Code algorithm: matches in the discriminator-side; the policy-update side uses soft Bellman in the tabular case (faster than RL) and gradient descent on a parameterized policy in the neural case.

- Match: **yes** with the noted policy-side choice.

### Identification assumptions

- Paper conditions: Theorem 5.1 of Fu et al. 2018 requires (a) state-only reward `g(s)` (not state-action), (b) deterministic, decomposable dynamics, (c) full state observability. Under these the recovered reward is *disentangled* from shaping and transfers across dynamics. With state-action reward `g(s, a)`, the disentanglement guarantee fails (Theorem 5.2 negative result).

- Code enforcement: the `state_only_reward` flag in `AIRLConfig` controls which version is used. The default is `True` (state-only, to match the paper's positive result). The code does **not** explicitly check whether dynamics are deterministic — that is a property of the data, not the estimator.

- Match: **yes**, with the documented assumption being the user's responsibility.

### Hyperparameter defaults vs paper defaults

- `AIRLConfig.num_outer_iterations`: 100 (paper used 1000+ for MuJoCo; DDC needs fewer).
- `AIRLConfig.num_disc_iterations_per_outer`: 1.
- `AIRLConfig.num_policy_iterations_per_outer`: 1.
- `AIRLConfig.disc_learning_rate`: 1e-3.
- `AIRLConfig.state_only_reward`: True (matches the disentanglement-positive setting).

Match: **yes** for the structure; **lower iteration count** than the paper because DDC tabular settings converge faster than MuJoCo continuous-control.

### Findings / fixes applied

- **No code fixes required.** The `f = g + gamma h(s') - h(s)` decomposition is implemented correctly. The `state_only_reward=True` default matches the paper's positive disentanglement result.

- **Caveat for the paper**: AIRL's transfer guarantee assumes deterministic dynamics. The shape-shifter ss-det-T cell is the regime where the guarantee should hold; the ss-spine cell with stochastic transitions is *not* covered by Theorem 5.1, so the empirical capability table should mark AIRL as "transfer-supported only on deterministic-T cells." This is a paper-table footnote, not a code change.

- VALIDATION_LOG.md status: **Pending** (Tier 4 ss-spine and ss-neural-r will validate).
