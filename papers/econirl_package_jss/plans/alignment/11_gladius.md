## Estimator: GLADIUS (Offline IRL via bi-conjugate Bellman error)
## Paper(s): Kang, Yoganarasimhan, Jain 2025 "Offline Inverse RL and Dynamic Discrete Choice Models" (working paper). At `papers/foundational/2025_kang_yoganarasimhan_jain_erm_offline_irl.pdf` (and `.md`).
## Code: `src/econirl/estimation/gladius.py`

### Loss / objective

- Paper formula (Kang et al. 2025, Section 4): two networks, a Q-network `Q_w(s, a)` and an EV-network `V_w(s)`, trained jointly. The training objective combines:

  - **Conditional log-likelihood** of observed actions: standard logit on Q,

    ```
    L_NLL = -E[ log softmax(Q(s, a)) ]
    ```

  - **Bi-conjugate Bellman error** that anchors the Q values to the (observed) reward:

    ```
    L_Bellman = E[ (Q(s, a) - r(s, a) - gamma * V(s'))^2 ]
    ```

  The two are summed with a weight `lambda` (paper's eq. 7).

  When **rewards are observed in the data** (the paper's intended use case), the Bellman penalty anchors the Q-values to the correct scale and the joint objective recovers (Q, V, R) up to known transforms.

- Code implementation: `gladius.py:_optimize` constructs `L = L_NLL + lambda * L_Bellman` per the paper. The two networks are equinox MLPs trained with Adam.

- Match: **yes** for the loss decomposition.

### Gradient

- Paper formula: standard gradient of the joint loss w.r.t. network weights.

- Code implementation: JAX autodiff through both networks and both loss components.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: no inner fixed-point. The Bellman error term plays the role of the fixed-point regularization without explicit iteration.

- Code algorithm: matches.

- Match: **yes**.

### Identification assumptions

- Paper conditions: **rewards must be observed in the data** for the bi-conjugate term to anchor Q. With unobserved rewards (the IRL setting), the NLL term alone identifies Q only up to a state-dependent constant `c(s)`, which propagates asymmetrically through the reward decomposition.

- Code enforcement: the package allows training without observed rewards (IRL mode). Per the project root CLAUDE.md, this produces a documented structural bias on the Rust bus: replacement cost recovered within 8 percent, operating cost overestimated by ~40 percent. NFXP recovers both within 5 percent. **The bias is structural, not a tuning problem.**

- Match: **yes** for theory; **package extends the paper's intended use** to the IRL setting and exposes the resulting bias.

### Hyperparameter defaults vs paper defaults

- `network_width`: 64 (paper used 128–256 for continuous-state envs).
- `network_depth`: 2.
- `learning_rate`: 1e-3.
- `lambda_bellman`: 1.0 (paper's recommended weight).
- `num_iterations`: 5000.

Match: **yes** for the structure; lower width because the package targets DDC tabular more than continuous-control.

### Findings / fixes applied

- **No code fixes required.** The bi-conjugate loss is implemented per the paper.

- **Caveat documented in the paper Section 3 already**: GLADIUS is appropriate for continuous-state environments where tabular methods cannot be applied, *or* when rewards are observed in the data. Do not use it for IRL on small tabular DDC problems where NFXP/CCP suffice. The Tier 4 ss-large-S cell is the regime where GLADIUS pays off relative to tabular CCP.

- VALIDATION_LOG.md status: **Pass with caveat (structural bias on tabular IRL; intended for continuous-state)** — matches the existing root CLAUDE.md note.
