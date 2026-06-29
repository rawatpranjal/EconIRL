# Theory

This section collects the proof ideas behind the public core estimators. Read it
as the bridge between the estimator pages and the source papers: the estimator
pages say what each method does, while these pages say which mathematical
restriction makes the object interpretable.

The proofs and proof sketches in this section draw from
{ref}`Kang (2026) <kang-2026-lecture>`, {ref}`Rawat and Rust (2026)
<rawat-rust-2026>`, and the classical sources cited inline. The notation is
adapted to the EconIRL estimator pages.

## How to Read This Section

Start with the soft Bellman page if the connection between dynamic discrete
choice and maximum-entropy IRL is unfamiliar. Then read identification before
choosing an estimator: most estimator differences are consequences of what is
known, what is normalized, and whether the transition kernel must be estimated.

| Topic | Main question | Estimators it supports |
| --- | --- | --- |
| [Soft Bellman and DDC-MaxEnt Equivalence](soft_bellman_equivalence.md) | Why do logit DDC and entropy-regularized IRL produce the same policy equation? | NFXP, CCP, TD-CCP, MCE-IRL, AIRL, GLADIUS. |
| [Identification and Anchors](identification.md) | What is recoverable from behavior alone, and what does an anchor action add? | NFXP, CCP, TD-CCP, AIRL-Het, GLADIUS. |
| [Classical DDC Proof Routes](classical_ddc.md) | How do NFXP, CCP, and TD-CCP enforce the identifying equations? | NFXP, CCP, TD-CCP. |
| [IRL Boundaries](irl_boundaries.md) | What do MCE-IRL, AIRL, GAIL, and IQ-Learn identify? | MCE-IRL, Neural MCE-IRL, AIRL, AIRL-Het, IQ-Learn. |
| [GLADIUS and ERM](gladius_erm.md) | How does the ERM route avoid transition-density estimation while keeping the Bellman identification condition? | GLADIUS. |

## The Common Object

All pages use the same finite-action discounted decision problem. A state is
$s$, an action is $a$, the transition kernel is $P(s' \mid s,a)$, the reward is
$r(s,a)$, and the discount factor is $\beta \in (0,1)$. The choice-specific
value is $Q(s,a)$, the soft value is

$$
V_Q(s) = \log \sum_{a \in A} \exp Q(s,a),
$$

and the implied policy is

$$
\pi_Q(a \mid s) =
\frac{\exp Q(s,a)}{\sum_b \exp Q(s,b)}.
$$

Read $Q$ as the object behavior identifies first. The reward $r$ is recovered
only after a Bellman equation and a normalization pin down the state-wise level
that the softmax cannot see.

```{toctree}
:hidden:
:maxdepth: 1

soft_bellman_equivalence
identification
classical_ddc
irl_boundaries
gladius_erm
```
