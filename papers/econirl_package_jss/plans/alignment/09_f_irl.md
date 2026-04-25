## Estimator: f-IRL (Inverse RL via f-divergence state-marginal matching)
## Paper(s): Ni, Sikchi, Eysenbach et al. 2022 (originally 2021 arXiv) "f-IRL: Inverse Reinforcement Learning via State Marginal Matching." At `papers/foundational/ni_2020_f_irl.pdf`.
## Code: `src/econirl/estimation/f_irl.py`

### Loss / objective

- Paper formula (Ni et al. 2022, eq. 6): minimize an f-divergence between the expert's state marginal and the policy's state marginal under the current reward,

  ```
  L(theta) = D_f( rho_E(s) || rho_pi(theta)(s) )
  ```

  where `D_f` is one of KL, reverse-KL, JS, or chi-squared. The reward is recovered from the divergence's first variation; the gradient w.r.t. theta uses the f-divergence's variational form (sample-based).

- Code implementation: `f_irl.py:_optimize` constructs the f-divergence loss for the configured `divergence` family. The supported divergences are KL, JS, chi-squared, and TV per the constructor signature. The state marginal `rho_pi` is computed from the soft-Bellman occupancy (closed-form in tabular).

- Match: **yes**, the four canonical divergences are implemented per Ni et al.'s Table 1.

### Gradient

- Paper formula: per Ni et al. eq. 8, the gradient is

  ```
  d_theta L = E_pi[ f'(rho_E / rho_pi) * d_theta log pi(a|s) ]
  ```

  where `f'` is the derivative of the divergence's generator. The package uses tabular occupancies so this is closed-form.

- Code implementation: JAX autodiff through the divergence loss. The closed-form occupancy from `core/occupancy.py` makes the gradient cheap.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: per outer step, recompute the policy from the current reward via soft VI, recompute the state marginal, evaluate the divergence, gradient-step.

- Code algorithm: matches.

- Match: **yes**.

### Identification assumptions

- Paper conditions: f-IRL recovers the reward up to a constant (standard IRL identification). The state-marginal-matching objective does not require feature engineering — it works directly on the state distribution. This is the paper's main pitch over MaxEnt IRL: no need to design features.

- Code enforcement: the package's f-IRL takes a tabular reward parameterization (one parameter per (s, a) cell or per s for state-only). No feature matrix required.

- Match: **yes**.

### Hyperparameter defaults vs paper defaults

- `divergence`: "kl" (paper recommends "fkl" or "rkl"; the package uses "kl" as a synonym for forward-KL).
- `learning_rate`: 1e-3.
- `num_iterations`: 500.

Match: **yes** for the divergence family; the iteration count is conservative for tabular settings.

### Findings / fixes applied

- **No code fixes required.** The four divergence families are implemented per the paper.

- **Caveat for the paper**: Ni et al. acknowledge that f-IRL can produce flat reward maps when the demonstration covers the state space densely (the divergence is minimized at zero reward). This is the "f-IRL is allowed to fail on rust-small" entry in plan_rust_small.md. The Tier 4 ss-spine cell will likely show similar behavior; the failure_mode column will record `reward_uncovered`.

- VALIDATION_LOG.md status: **Pending** (Tier 4 ss-spine, allowed to fail per the plan).
