# Comparing Estimators

Use these tables to narrow the estimator choice before opening a method-specific
page. Core estimators come first because they cover the main identified targets
and application domains.

## Core estimators

| Estimator | Use when | Data / transition requirement | Reward target | State scale | Avoid when | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| [NFXP](nfxp.md) | You need the reference structural DDC likelihood and counterfactual policy analysis. | Discrete panel data; transitions known or estimated first. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Repeated exact Bellman solves are too expensive or transition modeling is the main bottleneck. | Synthetic tabular simulation. |
| [CCP](ccp.md) | You want a faster Hotz-Miller or NPL tabular structural estimate. | Discrete panel data; transitions known or estimated first; strong empirical action support. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Many states have weak or one-action support, or you need the direct nested fixed-point likelihood. | Synthetic tabular simulation with support conditions. |
| [NNES](nnes.md) | The value object is too large, smooth, or encoded for repeated exact dynamic programming. | Discrete panel data; transitions known or estimated before estimation. | Finite parametric structural reward with neural value approximation. | Larger, encoded, smooth, or multidimensional states. | The reward itself must be an unrestricted neural function or transition estimation is the main problem. | Synthetic low- and high-dimensional structural DDC simulations. |
| [TD-CCP](tdccp.md) | Transition-density modeling is hard but the reward has known finite features. | Panel trajectories with current and next state-action information; transition environment still needed for post-fit counterfactuals. | Finite linear structural reward. | Encoded or higher-dimensional discrete states. | State space is small enough for tabular likelihood methods, support is sparse, or the target is a neural reward map. | Encoded-state finite-theta hard case with locally robust standard errors. |
| [MCE-IRL](mce_irl.md) | Demonstrations should be explained by maximum causal entropy feature matching. | Demonstrations from a discrete dynamic decision problem; transitions known or supplied. | Supplied finite reward features. | Tabular state-action spaces. | You need likelihood-based structural standard errors or reward features are unknown. | Synthetic supplied-feature simulations. |
| [RHIP](rhip.md) | Route choice or graph planning needs a horizon-scaled entropy IRL method. | Demonstrations, known graph transitions, and route features. | Reward representation judged by policy and horizon behavior. | Tabular or graph state-action spaces; scale motivation is route graphs. | You need finite structural DDC parameters or Google-scale infrastructure is the immediate claim. | Synthetic route-choice horizon spectrum. |
| [AIRL](airl.md) | AIRL-1: adversarial recovery under the original state-only AIRL assumptions is the research object. | Demonstrations from a discrete dynamic decision problem; transitions available for validation or post-fit evaluation. | State-only reward with shaping term under a fixed normalization. | Discrete dynamic decision settings. | Reward is action-dependent, an absorbing-state normalization is central, or structural action-dependent recovery is required. | Synthetic state-only AIRL simulation. |
| [AIRL-Het](airl_het.md) | AIRL-2: latent segments have different dynamic preferences and segment-specific counterfactuals matter. | Repeated user trajectories; credible anchor action and absorbing-state normalization. | Segment-specific action-dependent reward. | Encoded discrete dynamic choice settings. | Segment membership is weakly identified, no credible reward anchor exists, or a homogeneous estimator is enough. | Synthetic serialized-content simulation. |
| [GLADIUS](gladius.md) | You want neural Q and continuation modeling with anchor-moment reward projection. | Dynamic discrete choices; known transitions; credible anchor action with known rewards. | Projected structural reward from neural Q/continuation objects. | High-dimensional encoded state features. | No credible anchor action exists or you need tabular structural estimation. | Preview: projected reward diagnostics. |

## Other estimators

| Estimator | Use when | Current role |
| --- | --- | --- |
| [MPEC](mpec.md) | You want a constrained-optimization check on the DDC likelihood. | Secondary structural check; useful, but overlaps with NFXP/CCP and has higher solver complexity. |
| [UFXP](ufxp.md) | You want maximum-likelihood-grade structural estimates without nesting any fixed point in the search. | Secondary structural speed/FOC variant. |
| [Deep MCE-IRL](deep_mce_irl.md) | You need nonlinear reward-map recovery under the MCE objective. | Neural extension of MCE-IRL. |
| [f-IRL](f_irl.md) | The study question is state-marginal matching under an f-divergence. | Narrower state-marginal method. |
| [IQ-Learn](iq_learn.md) | Inverse soft-Q learning or imitation diagnostics are the estimator of interest. | Preview diagnostic. |

The tabular structural estimators are the usual starting point for small
dynamic discrete choice problems. Approximate structural estimators are for
larger state representations while keeping a finite reward target. IRL
estimators are for reward recovery from demonstrations, with scope determined
by the reward form, support, anchors, and transition information.
