# IQ-Learn Paper Context

Primary source: Garg et al. (2021) for inverse soft-Q learning. Public
citations live in `../../../docs/references.md`. Broader paper routing lives in
`../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Implicit Q function | Tabular or neural Q head | Main learned object. |
| Soft value | `sigma logsumexp(Q / sigma)` | Used for policy and inverse Bellman reward. |
| Inverse Bellman reward | `Q - beta P V` | Diagnostic reward, not automatically anchored. |
| Divergence choice | Configured objective family | Current validation uses chi-squared. |
| Regularized BC connection | Penalized conditional log-likelihood | Explains good imitation with weak reward recovery. |
| Expert support | State and state-action coverage gates | Required before structural interpretation. |
| Sparse-support failure | Guard artifact | Non-support metrics can pass while release status fails. |

## Derivation Checklist

1. Define Q, V, policy, and temporal difference.
2. Show that `Q - V = sigma log pi`.
3. Interpret the IQ-Learn loss as conditional log-likelihood plus implied
   reward regularization.
4. Explain how the inverse Bellman operator recovers an implied reward from Q.
5. State the reward-shaping non-identification problem.
6. Explain why support gates precede policy and counterfactual metrics.
7. Contrast soft regularization with NFXP/MPEC hard Bellman constraints.

## Old Primer Material To Retain

The retired primer contained durable material on:

- IQ-Learn as regularized behavioral cloning;
- the penalized MLE derivation;
- inverse-Bellman reward calculation;
- sparse expert support and discriminator instability context;
- invalid counterfactual uses when Q is not a structural Bellman fixed point.

Do not restore the TeX primer. Preserve this material in Markdown and current
validation artifacts.
