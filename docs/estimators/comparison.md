# Comparing Estimators

Use these tables to narrow the estimator choice before opening a method-specific
page. The core estimators carry the main identification stories and method
lineages. NFXP is the reference, the exact estimator replicated to the paper.

## Core estimators

| Estimator | Use when | Data / transition requirement | Reward target | State scale | Avoid when | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| [NFXP](nfxp.md) | You need the reference structural DDC likelihood and counterfactual policy analysis. | Discrete panel data; transitions known or estimated first. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Repeated exact Bellman solves are too expensive or transition modeling is the main bottleneck. | Synthetic tabular simulation and the Rust (1987) Table IX replication. |
| [CCP](ccp.md) | You want a faster Hotz-Miller or NPL tabular structural estimate. | Discrete panel data; transitions known or estimated first; strong empirical action support. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Many states have weak or one-action support, or you need the direct nested fixed-point likelihood. | Synthetic tabular simulation with support conditions. |
| [TD-CCP](tdccp.md) | Transition-density modeling is hard but the reward has known finite features. | Panel trajectories with current and next state-action information; transition environment still needed for post-fit counterfactuals. | Finite linear structural reward. | Encoded or higher-dimensional discrete states. | State space is small enough for tabular likelihood methods, support is sparse, or the target is a neural reward map. | Encoded-state finite-theta hard case with locally robust standard errors. |
| [MCE-IRL](mce_irl.md) | Demonstrations should be explained by maximum causal entropy feature matching. | Demonstrations from a discrete dynamic decision problem; transitions known or supplied. | Supplied finite reward features. | Tabular state-action spaces. | You need likelihood-based structural standard errors or reward features are unknown. | Synthetic supplied-feature simulations. |
| [Neural MCE-IRL](deep_mce_irl.md) | Demonstrations should be explained by an unrestricted neural reward under the maximum causal entropy objective. | Demonstrations from a discrete dynamic decision problem; transitions known or supplied. | Neural reward map. | Tabular or encoded state-action spaces. | You need finite structural parameters with standard errors, or supplied reward features are enough. | Synthetic neural-reward recovery simulation. |
| [AIRL](airl.md) | AIRL-1: adversarial recovery under the original state-only AIRL assumptions is the research object. | Demonstrations from a discrete dynamic decision problem; transitions available for validation or post-fit evaluation. | State-only reward with shaping term under a fixed normalization. | Discrete dynamic decision settings. | Reward is action-dependent, an absorbing-state normalization is central, or structural action-dependent recovery is required. | Synthetic state-only AIRL simulation. |
| [AIRL-Het](airl_het.md) | AIRL-2: latent segments have different dynamic preferences and segment-specific counterfactuals matter. | Repeated user trajectories; credible anchor action and absorbing-state normalization. | Segment-specific action-dependent reward. | Encoded discrete dynamic choice settings. | Segment membership is weakly identified, no credible reward anchor exists, or a homogeneous estimator is enough. | Synthetic serialized-content simulation. |
| [GLADIUS](gladius.md) | You want neural Q and continuation modeling with anchor-moment reward recovery. | Dynamic discrete choices; known transitions; credible anchor action with known rewards. | Neural reward recovered from neural Q/continuation objects. | High-dimensional encoded state features. | No credible anchor action exists or you need tabular structural estimation. | Preview: projected reward diagnostics. |

GLADIUS is the package's neural estimator. The `GLADIUS` class is the
`NeuralGLADIUS` implementation, so the two names refer to the same estimator.

## Other estimators

| Estimator | Use when | Current role |
| --- | --- | --- |
| [NNES](nnes.md) | The value object is too large or encoded for repeated exact dynamic programming. | Neural value approximation with finite structural parameters. |
| [MPEC](mpec.md) | You want a constrained-optimization check on the DDC likelihood. | Secondary structural check; overlaps with NFXP/CCP and has higher solver complexity. |
| [UFXP](ufxp.md) | You want maximum-likelihood-grade structural estimates without nesting any fixed point in the search. | Secondary structural speed/FOC variant. |
| [RHIP](rhip.md) | Route choice or graph planning needs a horizon-scaled entropy IRL method. | Horizon-parameterised entropy IRL for route graphs. |
| [f-IRL](f_irl.md) | The study question is state-marginal matching under an f-divergence. | Narrower state-marginal method. |
| [IQ-Learn](iq_learn.md) | Inverse soft-Q learning or imitation diagnostics are the estimator of interest. | Preview diagnostic. |

The tabular structural estimators are the usual starting point for small dynamic
discrete choice problems. Approximate structural estimators are for larger state
representations while keeping a finite reward target. IRL estimators are for reward
recovery from demonstrations, with scope determined by the reward form, support,
anchors, and transition information.
