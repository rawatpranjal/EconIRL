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

On a small, well-specified tabular problem, none of these does anything NFXP cannot.
Their value appears only when one of these complications is real.
