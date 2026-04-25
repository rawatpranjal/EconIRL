## Estimator: IQ-Learn (Inverse soft-Q Learning for Imitation)
## Paper(s): Garg, Chakraborty et al. 2021 "IQ-Learn: Inverse soft-Q Learning for Imitation," NeurIPS. At `papers/foundational/2022_iq_learn.pdf`.
## Code: `src/econirl/estimation/iq_learn.py`

### Loss / objective

- Paper formula (Garg et al. 2021, eq. 12; the chi-squared / "simple" objective): represent the reward implicitly through a Q-function, and minimize the chi-squared divergence between expert and policy occupancies. The closed-form objective is

  ```
  J(Q) = E_E[ phi(Q(s, a) - gamma * V(s')) ] - (1 - gamma) * E_rho_0[V(s_0)]
  ```

  where `phi` is the chi-squared generator, `V(s) = sigma * log sum_a exp(Q(s, a) / sigma)` is the soft value function, and `rho_0` is the initial state distribution. The reward is recovered as `R(s, a) = Q(s, a) - gamma * E_{s'}[V(s')]`.

  Garg et al. show this objective is *concave* in Q (Theorem 4.2), which is the major selling point over GAIL/AIRL.

- Code implementation: `iq_learn.py:_optimize` constructs the chi-squared loss with the paper's `phi(x) = x - x^2 / 4` generator. The Q-function is parameterized as a `(S, A)` tabular tensor (linear) or a small MLP (neural). The initial state distribution comes from `panel.compute_initial_distribution`.

- Match: **yes** for the chi-squared "simple" form.

### Gradient

- Paper formula: gradient of the chi-squared loss w.r.t. Q (concave maximization).

- Code implementation: JAX autodiff. The gradient passes through the soft-V function and the linear-quadratic generator.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: no inner Bellman fixed-point. The Q-function is updated directly by gradient descent on the chi-squared objective; the soft-V and reward are recovered post-hoc as side computations.

- Code algorithm: matches. No inner VI loop.

- Match: **yes**.

### Identification assumptions

- Paper conditions: the chi-squared objective is concave but the reward is identified only up to a state-only function (per Garg et al. Section 5.1, "Reward Recovery"). The same identifiability caveat applies as MCE-IRL.

- Code enforcement: the wrapper exposes the recovered Q and reward; no anchor identification is enforced. The user is expected to handle the identifiability post-fit.

- Match: **yes**.

### Hyperparameter defaults vs paper defaults

- `divergence`: "chi2" (the paper's default).
- `learning_rate`: 3e-4 (Adam).
- `num_iterations`: 1000.

Match: **yes**, mirrors Garg et al.'s tabular experiments (their MuJoCo experiments used larger nets and more iterations).

### Findings / fixes applied

- **No code fixes required.** The chi-squared objective and the soft-V recovery are implemented per the paper.

- **Caveat per project memory `iq_learn_primer_framing.md`**: IQ-Learn's tabular Q does not propagate to unvisited states (no Bellman backup). On panels where state coverage is incomplete, IQ-Learn's reward is undefined on the unvisited subset. The shape-shifter ss-spine cell uses uniform initial distribution and 100-period trajectories, which usually achieves full state coverage; this caveat would bite only on smaller panels.

- VALIDATION_LOG.md status: **Pending** (Tier 4 ss-spine and Tier 2 rust-small).
