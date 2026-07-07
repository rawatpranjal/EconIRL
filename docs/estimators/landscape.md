# Choosing an Estimator

EconIRL is a research-oriented library. It centers on one reference estimator and
treats the rest as answers to specific complications. The reference is the nested
fixed point (NFXP), the exact maximum-likelihood estimator for tabular structural
dynamic discrete choice. The other estimators exist because real problems break one
of the assumptions that make NFXP exact and cheap.

## The canonical case

NFXP is at its best when every one of these holds.

- A single forward-looking agent in a stationary, infinite-horizon, discrete-time problem.
- A small finite state space. Every state can be enumerated.
- A small discrete action set.
- A flow payoff linear in known features, u(s,a) = phi(s,a)^T theta, with a few parameters.
- A simple exogenous transition kernel, estimated in a first stage and separate from the payoff parameters.
- Additive i.i.d. Type-I extreme-value shocks, which give closed-form logit choice probabilities.
- A pinned normalization. An anchor action fixes the reward level, the logit scale is fixed, and the action-contrast feature rank equals the parameter count.

When all of this holds, the soft-Bellman operator is a contraction on a small grid.
The inner fixed point solves exactly and fast. The outer loop is full-information
maximum likelihood with an exact inner solve at every candidate parameter. The
result is the efficient estimator with standard errors from the information matrix.
The Rust (1987) bus-engine model is the canonical instance. See [NFXP](nfxp.md) and
the [replications ledger](../replications.md).

## What breaks it, and what answers each break

Each other estimator relaxes one assumption above, and pays for it somewhere else.

**The state space gets large or continuous.** A small state space is what makes the
exact nested solve cheap. Encoded, high-dimensional, or smooth states make the inner
fixed point infeasible. This motivates [CCP](ccp.md) (invert observed choice
probabilities, no nested solve), [MPEC](mpec.md) (one constrained optimization
instead of repeated inner solves), [UFXP](ufxp.md) (fold the fixed point into the
optimization), and [NNES](nnes.md) (a neural continuation value with finite reward
parameters).

**The transition density gets hard to model.** NFXP needs an exogenous, estimable
transition kernel. When that first stage is the bottleneck, [TD-CCP](tdccp.md)
estimates the reward parameters without modeling the transition density at all.

**The reward form is unknown.** NFXP assumes the payoff is linear in known features.
When the reward must be learned from demonstrations, the problem becomes inverse
reinforcement learning: [MCE-IRL](mce_irl.md), [Neural MCE-IRL](deep_mce_irl.md)
for an unrestricted reward map, [AIRL](airl.md) for a transferable state-only
reward, and [GLADIUS](gladius.md) for neural reward recovery in high-dimensional
states.

**The agents are not homogeneous.** The canonical case is one representative agent.
Latent types with different preferences motivate [AIRL-Het](airl_het.md), which
recovers segment-specific rewards under an anchor.

**The agent is not an optimal infinite-horizon planner.** Real demonstrators may
plan to a finite depth. [RHIP](rhip.md) makes the planning horizon a parameter and
scales to route-choice and graph problems.

## At a glance

| Source of complexity | Estimators it motivates |
| --- | --- |
| Large or continuous state space | CCP, MPEC, UFXP, NNES |
| Hard-to-model transition density | TD-CCP |
| Unknown reward form (IRL) | MCE-IRL, Neural MCE-IRL, AIRL, GLADIUS |
| Latent heterogeneity | AIRL-Het |
| Bounded or finite-horizon planning | RHIP |

## Core estimators side by side

The core estimators carry the main identification stories and method lineages.
Use this table to narrow the choice before opening a method page.

| Estimator | Use when | Data / transition requirement | Reward target | State scale | Avoid when | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| [NFXP](nfxp.md) | You need the reference structural DDC likelihood and counterfactual policy analysis. | Discrete panel data; transitions known or estimated first. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Repeated exact Bellman solves are too expensive or transition modeling is the main bottleneck. | Synthetic tabular simulation and the Rust (1987) Table IX replication. |
| [CCP](ccp.md) | You want a faster Hotz-Miller or NPL tabular structural estimate. | Discrete panel data; transitions known or estimated first; strong empirical action support. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Many states have weak or one-action support, or you need the direct nested fixed-point likelihood. | Synthetic tabular simulation with support conditions. |
| [MCE-IRL](mce_irl.md) | Demonstrations should be explained by maximum causal entropy feature matching. | Demonstrations from a discrete dynamic decision problem; transitions known or supplied. | Supplied finite reward features. | Tabular state-action spaces. | You need likelihood-based structural standard errors or reward features are unknown. | Synthetic supplied-feature simulations. |
| [Neural MCE-IRL](deep_mce_irl.md) | Demonstrations should be explained by an unrestricted neural reward under the maximum causal entropy objective. | Demonstrations from a discrete dynamic decision problem; transitions known or supplied. | Neural reward map. | Tabular or encoded state-action spaces. | You need finite structural parameters with standard errors, or supplied reward features are enough. | Synthetic neural-reward recovery simulation. |
| [AIRL](airl.md) | AIRL-1: adversarial recovery under the original state-only AIRL assumptions is the research object. | Demonstrations from a discrete dynamic decision problem; transitions available for validation or post-fit evaluation. | State-only reward with shaping term under a fixed normalization. | Discrete dynamic decision settings. | Reward is action-dependent, an absorbing-state normalization is central, or structural action-dependent recovery is required. | Synthetic state-only AIRL simulation. |
| [AIRL-Het](airl_het.md) | AIRL-2: latent segments have different dynamic preferences and segment-specific counterfactuals matter. | Repeated user trajectories; credible anchor action and absorbing-state normalization. | Segment-specific action-dependent reward. | Encoded discrete dynamic choice settings. | Segment membership is weakly identified, no credible reward anchor exists, or a homogeneous estimator is enough. | Synthetic serialized-content simulation. |
| [GLADIUS](gladius.md) | You want neural Q and continuation modeling with anchor-moment reward recovery. | Dynamic discrete choices; known transitions; credible anchor action with known rewards. | Neural reward recovered from neural Q/continuation objects. | High-dimensional encoded state features. | No credible anchor action exists or you need tabular structural estimation. | Preview: projected reward diagnostics. |

GLADIUS is the package's neural estimator. The `GLADIUS` class is the
`NeuralGLADIUS` implementation, so the two names refer to the same estimator.

## Other estimators side by side

| Estimator | Use when | Current role |
| --- | --- | --- |
| [TD-CCP](tdccp.md) | Transition-density modeling is the bottleneck but the reward has known finite features. | Reward estimation without modeling the transition density. |
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

On a small, well-specified tabular problem, none of these does anything NFXP cannot.
Their value appears only when one of these complications is real.
