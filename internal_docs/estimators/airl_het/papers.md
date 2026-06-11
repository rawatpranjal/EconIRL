# AIRL-Het Paper Context

Primary sources: Fu, Luo, and Levine (2018) for AIRL and Lee, Sudhir, and Wang
(2026) for the serialized-content heterogeneous dynamic choice setting. Public
citations live in `../../../docs/references.md`. Broader paper routing lives in
`../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Latent consumer types | Segment-specific reward and policy objects | Current validation uses two segments. |
| Segment prior | `lambda_k` | Compared through prior L1 and max error. |
| Posterior assignment | `q_ik` | Compared through segment assignment accuracy. |
| AIRL discriminator | Segment-specific `f_k = g_k + beta h_k(s') - h_k(s)` | Extends base AIRL. |
| Exit payoff normalization | `g_k(s, exit) = 0` | Reward anchor for action-dependent DDC. |
| Absorbing-state normalization | `h_k(absorbing) = 0` | Value-potential anchor. |
| Serialized-content panel | Package validation environment | Paper context, not manuscript workspace. |

## Derivation Checklist

1. Start from base AIRL's shaping ambiguity.
2. Show why state-only/decomposable AIRL is insufficient for DDC exits.
3. Add action-dependent `g_k(s, a)`.
4. Impose exit-action and absorbing-state anchors.
5. Define segment priors and trajectory posteriors.
6. Write the EM assignment/update loop.
7. State how segment permutation is handled in validation metrics.
8. Separate package validation from ORE manuscript exposition.

## Old Primer Material To Retain

The retired primer contained durable material on:

- latent segment notation;
- exit-action and absorbing-state anchors;
- the EM-AIRL algorithm;
- conservative policy mixing;
- serialized-content validation;
- contrast with pooled AIRL and behavioral cloning.

Do not restore the TeX primer. Preserve useful content as Markdown in this
folder or the public RTD source.
