# GLADIUS Paper Context

Primary source: Kang, Yoganarasimhan, and Jain (2025) for the GLADIUS offline
IRL-DDC objective. Public citations live in `../../../docs/references.md`.
Broader paper routing lives in `../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Neural Q function | `Q(s, a)` network | Induces policy through softmax/logsumexp. |
| Expected continuation value | `zeta(s, a)` network | Approximates `E[V(s') | s, a]`. |
| Bellman error | Raw Bellman reward diagnostic | Fails current strict release gates. |
| Bi-conjugate decomposition | Q/EV alternating training | Avoids double-sampling issue in the paper setup. |
| Observed reward anchor | Package anchor-moment variant | Latent-reward IRL does not have the same anchor. |
| Structural parameter recovery | Action-difference projection | Useful but not full certification. |
| Counterfactual use | Strict gate bundle | Requires more than low perturbation regret. |

## Derivation Checklist

1. Define Q, soft value, continuation value, and implied reward.
2. Derive the Bellman error and the bi-conjugate EV representation.
3. Explain why observed rewards anchor the paper objective.
4. Explain the latent-reward package variant and why it is weaker.
5. Show the action-difference projection used to recover theta.
6. Separate raw Bellman reward, projected reward, policy, value, Q, and
   counterfactual metrics.
7. Preserve the current strict non-certification status until gates pass.

## Old Primer Material To Retain

The retired primer captured:

- Q and EV network roles;
- the empirical-risk objective;
- the action-difference projection;
- the state-dependent Q-constant issue;
- the warning that latent-reward IRL lacks the paper's observed-reward anchor;
- high-state motivation.

Do not restore the TeX primer or old generated tables. Keep the material in
Markdown and current JSON validation artifacts.
