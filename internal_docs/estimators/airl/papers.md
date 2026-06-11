# AIRL Paper Context

Primary source: Fu, Luo, and Levine (2018) for adversarial inverse
reinforcement learning and potential-based reward shaping. Public citations
live in `../../../docs/references.md`. Broader paper routing lives in
`../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Structured discriminator | `f = g + beta h(s') - h(s)` | Separates reward candidate and shaping potential. |
| Reward component | `g_theta` | State-only in the original guarantee. |
| Shaping potential | `h_phi` | Value-like object that can absorb non-transferable dynamics terms. |
| Policy update | Package adversarial loop | Stabilized in tabular validation with conservative settings. |
| Transfer reward | Fixed-gauge reward map | Validated only in the state-only cell. |
| Action-dependent rewards | Negative diagnostic in AIRL | Use AIRL-Het or an anchor design for DDC claims. |

## Derivation Checklist

1. State the potential-based shaping ambiguity.
2. Define the discriminator decomposition.
3. Show the log-odds reward signal.
4. Explain why `D = 1/2` at the adversarial optimum implies `f = log pi`.
5. State the state-only and decomposability conditions.
6. Explain why action-dependent payoffs violate the original theorem.
7. Connect this limitation to the AIRL-Het anchor design.

## Old Primer Material To Retain

The retired primer contained useful material on:

- reward shaping and transfer failure under changed dynamics;
- the state-only AIRL theorem;
- the discriminator algorithm;
- state-only versus state-action simulation cells;
- why AIRL is not automatically preferable to NFXP or MCE in small tabular
  environments.

Do not restore the TeX primer. Convert any missing details into this folder or
the public RTD source as Markdown.
