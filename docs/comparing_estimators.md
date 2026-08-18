# Choosing and Comparing Estimators

This page is the first stop when you are deciding which EconIRL estimator to
open from the menu. It assumes only the basic goal: you have observations of
choices over time, and you want to estimate either a structural dynamic
discrete-choice model or a reward function that explains observed behavior.

The estimators differ in what they treat as known, what they estimate, and what
kind of recovery claim they can support. A method that predicts choices well is
not automatically recovering the reward. Reward recovery also needs the right
normalization, support, transition information, and identification argument.

Detailed math, system views, examples, and usage live on the estimator pages.
Paper-number checks live in [Replications](replications.md). Problem-level
simulation results live in [Simulation Studies](simulation_studies/index.md).

## How to Read the Menu

Start from the simplest structural case and move only when a specific part of
that case fails.

1. If the problem is small, tabular, and structurally specified, start with
   [NFXP](estimators/nfxp.md).
2. If NFXP is the right target but the repeated Bellman solves are too costly,
   inspect [CCP](estimators/ccp.md), [MPEC](estimators/mpec.md),
   [UFXP](estimators/ufxp.md), and [NNES](estimators/nnes.md).
3. If transition-density estimation is the bottleneck, inspect
   [TD-CCP](estimators/tdccp.md).
4. If the reward itself must be learned from demonstrations, inspect the IRL
   estimators: [MCE-IRL](estimators/mce_irl.md),
   [Neural MCE-IRL](estimators/deep_mce_irl.md),
   [AIRL](estimators/airl.md), [NeuralAIRL](estimators/neural_airl.md),
   [AIRL-Het](estimators/airl_het.md), and
   [GLADIUS](estimators/gladius.md).
5. If the behavior is better described by bounded or finite-horizon planning,
   inspect [RHIP](estimators/rhip.md).

## The Canonical Case

The reference estimator is nested fixed point (NFXP). It estimates a small
tabular structural dynamic discrete-choice model by maximum likelihood. It
solves the agent's dynamic program inside the likelihood for each candidate
reward parameter.

NFXP is at its best when every one of these holds.

- A single forward-looking agent in a stationary, infinite-horizon,
  discrete-time problem.
- A small finite state space where every state can be enumerated.
- A small discrete action set.
- A flow payoff that is linear in known features,
  $u(s,a) = \phi(s,a)^\top\theta$, with a few parameters.
- A simple exogenous transition kernel, estimated in a first stage and separate
  from the payoff parameters.
- Additive i.i.d. Type-I extreme-value shocks, which give closed-form logit
  choice probabilities.
- A pinned normalization. An anchor action fixes the reward level, the logit
  scale is fixed, and the action-contrast feature rank equals the parameter
  count.

When this holds, the soft-Bellman operator is a contraction on a small grid.
The inner fixed point solves exactly and quickly. The outer loop is
full-information maximum likelihood with an exact inner solve at every
candidate parameter. The result is the efficient reference estimator, with
standard errors from the information matrix. The Rust (1987) bus-engine model
is the canonical example.

## What Breaks the Canonical Case

Each other estimator relaxes one part of the NFXP setup and pays for it
somewhere else.

| Source of complexity | Estimators it motivates | What changes |
| --- | --- | --- |
| Large or continuous state space | CCP, MPEC, UFXP, NNES | The exact nested solve is replaced by inversion, constraints, fixed-point first-order conditions, or value approximation. |
| Hard-to-model transition density | TD-CCP | Estimation uses observed successor tuples instead of a transition-density model. |
| Unknown reward form | MCE-IRL, Neural MCE-IRL, AIRL, NeuralAIRL, AIRL-Het, GLADIUS | The target shifts from structural likelihood estimation to reward recovery from demonstrations. |
| Latent heterogeneity | AIRL-Het | The reward is segment-specific and needs credible segment, exit-action, and absorbing-state anchors. |
| Bounded or finite-horizon planning | RHIP | The planning horizon becomes part of the behavioral model. |

Large or continuous state spaces break the cheap exact solve. Encoded,
high-dimensional, or smooth states make repeated tabular dynamic programming
unattractive. CCP avoids repeated solves by using first-stage choice
probabilities. MPEC and UFXP keep a structural likelihood target but change the
optimization route. NNES uses a neural continuation value with finite reward
parameters.

Hard transition-density problems motivate TD-CCP. The estimator still targets a
finite structural reward parameter, but it estimates recursive terms from
observed successor state-action pairs instead of first modeling the transition
density.

Unknown rewards motivate inverse reinforcement learning. MCE-IRL recovers
reward coefficients in supplied features. Neural MCE-IRL uses a neural reward
map. AIRL targets the original state-only transfer claim. NeuralAIRL keeps that
state-only boundary with neural function approximation. AIRL-Het adds
anchors for segment-specific action-dependent rewards. GLADIUS learns neural Q
and continuation objects, then reads reward information through anchored action
contrasts.

## Core Lineage

The core estimators carry the main identification stories in EconIRL.

| Estimator | Main question | What changes relative to NFXP |
| --- | --- | --- |
| [NFXP](estimators/nfxp.md) | What is the exact tabular structural likelihood estimate? | Nothing. It is the reference under the maintained DDC assumptions. |
| [CCP](estimators/ccp.md) | Can we avoid repeated Bellman solves? | Uses first-stage choice probabilities and Hotz-Miller inversion. NPL iterates the pseudo-likelihood route. |
| [TD-CCP](estimators/tdccp.md) | Can we avoid transition-density estimation during parameter estimation? | Uses observed successor state-action pairs to estimate recursive terms. |
| [MCE-IRL](estimators/mce_irl.md) | Can demonstrations identify a reward? | Replaces structural likelihood with maximum causal entropy feature matching. |
| [Neural MCE-IRL](estimators/deep_mce_irl.md) | Can the reward be nonlinear? | Replaces fixed linear reward features with a neural reward map. |
| [AIRL](estimators/airl.md) | Can a state-only reward transfer across dynamics? | Separates state reward from shaping under the original AIRL assumptions. |
| [NeuralAIRL](estimators/neural_airl.md) | Can that state-only reward be nonlinear? | Uses neural approximators for the state reward, shaping potential, and generator policy. |
| [AIRL-Het](estimators/airl_het.md) | Can anchored rewards differ across latent segments? | Adds exit-action and absorbing-state anchors, then estimates segment-specific rewards. |
| [GLADIUS](estimators/gladius.md) | Can high-dimensional offline DDC avoid repeated solves? | Learns Q and continuation objects, then reads projected action contrasts. |

## Core Estimators Side by Side

Use this table to narrow the choice before opening a method page.

| Estimator | Use when | Data and transition requirement | Reward target | State scale | Avoid when | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| [NFXP](estimators/nfxp.md) | You need the reference structural DDC likelihood and counterfactual policy analysis. | Discrete panel data; transitions known or estimated first. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Repeated exact Bellman solves are too expensive or transition modeling is the main bottleneck. | Synthetic tabular simulation and the Rust (1987) Table IX replication. |
| [CCP](estimators/ccp.md) | You want a faster Hotz-Miller or NPL tabular structural estimate. | Discrete panel data; transitions known or estimated first; strong empirical action support. | Finite parametric structural reward. | Small or moderate tabular state-action spaces. | Many states have weak or one-action support, or you need the direct nested fixed-point likelihood. | Synthetic tabular simulation with support conditions. |
| [TD-CCP](estimators/tdccp.md) | Transition-density modeling is hard but the reward has known finite features. | Panel trajectories with current and next state-action information; an environment is still needed for post-fit counterfactuals. | Finite linear structural reward. | Encoded or higher-dimensional discrete states. | State space is small enough for tabular likelihood methods, support is sparse, or the target is a neural reward map. | Encoded-state finite-parameter hard case with locally robust standard errors. |
| [MCE-IRL](estimators/mce_irl.md) | Demonstrations should be explained by maximum causal entropy feature matching. | Demonstrations from a discrete dynamic decision problem; transitions known or supplied. | Supplied finite reward features. | Tabular state-action spaces. | You need likelihood-based structural standard errors or reward features are unknown. | Synthetic supplied-feature simulations. |
| [Neural MCE-IRL](estimators/deep_mce_irl.md) | Demonstrations should be explained by an unrestricted neural reward under the maximum causal entropy objective. | Demonstrations from a discrete dynamic decision problem; transitions known or supplied. | Neural reward map. | Tabular or encoded state-action spaces. | You need finite structural parameters with standard errors, or supplied reward features are enough. | Synthetic neural-reward recovery simulation. |
| [AIRL](estimators/airl.md) | Adversarial recovery under the original state-only AIRL assumptions is the research object. | Demonstrations from a discrete dynamic decision problem; transitions available for validation or post-fit evaluation. | State-only reward with shaping term under a fixed normalization. | Discrete dynamic decision settings. | Reward is action-dependent, an absorbing-state normalization is central, or structural action-dependent recovery is required. | Synthetic state-only AIRL simulation. |
| [NeuralAIRL](estimators/neural_airl.md) | The AIRL state reward is nonlinear in known state inputs. | Complete demonstration trajectories and an explicit transition tensor. | Nonlinear state-only reward with neural shaping and policy functions. | Tabular or finite encoded state inputs. | Heterogeneity or action-dependent structural reward is the target. | Generated nonlinear behavioral recovery. |
| [AIRL-Het](estimators/airl_het.md) | Latent segments have different dynamic preferences and segment-specific counterfactuals matter. | Repeated user trajectories; credible exit-action and absorbing-state anchors. | Segment-specific action-dependent reward. | Encoded discrete dynamic choice settings. | Segment membership is weakly identified, no credible reward anchor exists, or a homogeneous estimator is enough. | Synthetic serialized-content simulation. |
| [GLADIUS](estimators/gladius.md) | You want neural Q and continuation modeling with anchor-moment reward recovery. | Dynamic discrete choices; known transitions; credible anchor action with known rewards. | Neural reward recovered from neural Q and continuation objects. | High-dimensional encoded state features. | No credible anchor action exists or you need tabular structural estimation. | Preview: projected reward diagnostics. |

## How the Papers Relate

The source papers are not all answering the same question. Some keep the
structural DDC target and change the computation. Others switch to reward
recovery from demonstrations.

| Route | Paper comparator | What to carry into EconIRL |
| --- | --- | --- |
| NFXP / Rust | Earlier empirical replacement models. | Treat NFXP as the exact small-tabular benchmark. Later methods usually criticize its cost, not its target. |
| CCP / Hotz-Miller and NPL | NFXP and maximum likelihood. | CCP is another route to the same finite DDC target when support is strong. |
| TD-CCP / Adusumilli-Eckardt | NFXP, CCP, and transition-density-based DDC. | TD-CCP is transition-density-free for estimation. Counterfactuals still need an environment. |
| MCE-IRL / Ziebart | Apprenticeship learning and non-causal MaxEnt IRL. | MCE-IRL changes the estimand. It recovers reward only in the supplied feature span and normalization. |
| Neural MCE-IRL / DeepIRL | Linear MaxEnt IRL. | Neural MCE-IRL is a nonlinear reward-map extension of MCE-IRL. Raw weights are not the estimand. |
| AIRL / Fu-Luo-Levine | GAIL and shaped adversarial rewards. | Use AIRL for the original state-only reward-transfer claim. |
| NeuralAIRL / Fu-Luo-Levine | Tabular AIRL with a fixed reward basis. | Neural functions expand the reward map, but their raw weights are not structural parameters. |
| AIRL-Het / Lee-Sudhir-Wang | Homogeneous AIRL and pooled dynamic choice. | Exit and absorbing-state anchors do the identification work for action-dependent and segment-specific rewards. |
| GLADIUS / Kang-Yoganarasimhan-Jain | NFXP, CCP, TD-CCP, offline MaxEnt IRL, and Bellman-loss methods. | GLADIUS is the high-dimensional offline bridge. Its safest structural object is the projected action contrast. |

## Paper MDP Shapes

Use this table to match an estimator to the kind of decision process that
motivated it.

| Estimator | Source or showcase setting | Natural problem shape |
| --- | --- | --- |
| [NFXP](estimators/nfxp.md) | Rust bus-engine replacement. | Small tabular keep-or-replace panels with a full transition model. |
| [CCP](estimators/ccp.md) | Hotz-Miller dynamic-choice inversion. | Finite panels with reliable choice probabilities in each state. |
| [TD-CCP](estimators/tdccp.md) | Transition-density-free DDC panels. | Current and successor state-action tuples, with finite reward features. |
| [MCE-IRL](estimators/mce_irl.md) | Taxi route preference with road features. | Demonstrations in a known controlled process with credible reward features. |
| [Neural MCE-IRL](estimators/deep_mce_irl.md) | DeepIRL grid maps. | Nonlinear state or state-action reward maps with known transitions. |
| [AIRL](estimators/airl.md) | State-only transfer MDP. | State-only reward transfer under the Fu-Luo-Levine assumptions. |
| [NeuralAIRL](estimators/neural_airl.md) | Nonlinear finite-state transfer MDP. | State-only neural reward recovery with known transitions. |
| [AIRL-Het](estimators/airl_het.md) | Serialized-content choice with latent types. | Anchored action-dependent rewards and persistent latent segments. |
| [GLADIUS](estimators/gladius.md) | High-dimensional offline dynamic choice. | Offline panels where tabular dynamic programming is too costly and action contrasts are enough. |

## Main Axes

### State Scale

| State setting | Natural estimators | Why |
| --- | --- | --- |
| Small tabular state space | NFXP, CCP, MCE-IRL, AIRL | The full grid can be enumerated. |
| Small or moderate tabular space with speed pressure | CCP, MPEC, UFXP | The structural target remains tabular, but the computation changes. |
| Encoded or higher-dimensional state space with finite reward parameters | TD-CCP, NNES | The reward is still finite-dimensional, but transition or value objects become harder. |
| Nonlinear reward over tabular or encoded states | Neural MCE-IRL, NeuralAIRL | The reward map is neural, but planning still uses supplied transitions. |
| Repeated choices with latent segments | AIRL-Het | It estimates segment-specific rewards and policies. |
| High-dimensional offline state features | GLADIUS | It learns Q and continuation objects instead of repeated tabular solves. |

### Reward Form

| Reward target | Core estimators | Main caution |
| --- | --- | --- |
| Finite linear structural reward | NFXP, CCP, TD-CCP | Needs dynamic feature rank and a fixed normalization. Direct action-contrast rank is a conservative sufficient check. |
| Linear IRL reward basis | MCE-IRL | Identified only inside the supplied feature basis. |
| Neural reward map | Neural MCE-IRL, NeuralAIRL | The reward matrix is the object. The raw weights are not. |
| State-only transferable reward | AIRL, NeuralAIRL | Matches the original AIRL claim only under its state-only assumptions. |
| Segment-specific action-dependent reward | AIRL-Het | Needs credible exit-action and absorbing-state anchors, persistent segments, and enough trajectory support per segment. |
| Projected action contrast | GLADIUS | Stronger than raw Bellman reward levels in the package route. |

### Transition Information

| Transition input | Estimators | Meaning |
| --- | --- | --- |
| Explicit transition tensor | NFXP, CCP, MCE-IRL, Neural MCE-IRL, AIRL, NeuralAIRL, AIRL-Het | The estimator or policy update uses a transition model. |
| Observed successor pairs for estimation | TD-CCP | Estimation uses successor tuples instead of a transition-density model. |
| Offline next states | GLADIUS | Training uses sampled next states and learned continuation objects. |

TD-CCP still needs an environment for counterfactuals. GLADIUS still needs a
credible anchor to support reward interpretation. AIRL's unanchored
state-action discriminator should be read as behavior-fitting evidence, not as
identified structural reward.

## Recovery Conditions

Here "recovers" means population-level recovery of the stated reward object
after the required normalization. Matching choices is weaker than recovering
the reward.

| Estimator or version | Reward object | Can recover it? | Conditions that matter most |
| --- | --- | --- | --- |
| NFXP | Finite linear $R(s,a)$ | Yes. | Correct DDC model, Markov state, exogenous transitions, fixed discount and logit scale, enough support, and global likelihood optimum. |
| CCP one-step | Same finite target as NFXP. | Yes, in population. | Same structural conditions as NFXP, plus reliable first-stage CCPs and no zero-support cells. |
| CCP fixed-stage NPL | Same finite target as NFXP. | Yes, in population. | Same as one-step CCP. Every fixed positive stage is first-order equivalent to partial maximum likelihood under the Aguirregabiria-Mira assumptions. |
| CCP fixed-point NPL | Same finite target as NFXP. | Yes, in population. | Same as fixed-stage CCP, plus joint parameter and policy convergence to the relevant NPL fixed point. |
| TD-CCP semigradient | Finite reward parameters. | Yes, in population. | Successor tuples, consistent CCPs, recursive terms in the projection span, support, and correct normalization. |
| TD-CCP neural | Same finite reward parameters. | Conditional. | Same target as semigradient TD-CCP, with enough data and capacity to learn the recursive terms. |
| MCE-IRL | Reward coefficients in supplied features. | Yes, in population. | Known transitions, true reward in the feature span, full-rank moments, support, and fixed normalization. |
| Neural MCE-IRL | Anchored reward matrix. | Conditional. | Known transitions, representable reward, sufficient occupancy, and exact optimization under a fixed anchor. |
| AIRL | State-only reward up to a constant. | Yes under the original AIRL assumptions. | State-only reward, decomposable dynamics, sufficient expert and learner samples, and adversarial equilibrium. |
| NeuralAIRL | Nonlinear state-only reward up to a constant. | Conditional. | State-only reward, deterministic decomposable dynamics, representable state inputs, complete observed-state coverage, sufficient samples, and recovery of the optimal discriminator score. |
| AIRL-Het | Segment-specific anchored $R_k(s,a)$. | Conditional. | Exit-action reward anchor, absorbing-state value anchor, correct segment count, segment separation, support, fixed discount, and exact policy solution. |
| GLADIUS dual anchor-moment | Projected action contrasts. | Conditional. | Credible anchor, learned Q and continuation objects, action-contrast rank, and support. |
| GLADIUS `q_only` | Full reward. | No. | Useful as a diagnostic mode, but not enough for reward recovery. |

## Other Estimators

The non-core estimator pages are useful when a specific computational or
behavioral complication is the main reason not to use the reference route.

| Estimator | Use when | Current role |
| --- | --- | --- |
| [NNES](estimators/nnes.md) | The value object is too large or encoded for repeated exact dynamic programming. | Neural value approximation with finite structural parameters. |
| [MPEC](estimators/mpec.md) | You want a constrained-optimization check on the DDC likelihood. | Secondary structural check; overlaps with NFXP/CCP and has higher solver complexity. |
| [UFXP](estimators/ufxp.md) | You want maximum-likelihood-grade structural estimates without nesting any fixed point in the search. | Secondary structural speed and first-order-condition variant. |
| [RHIP](estimators/rhip.md) | Route choice or graph planning needs a horizon-scaled entropy IRL method. | Horizon-parameterized entropy IRL for route graphs. |
| [f-IRL](estimators/f_irl.md) | The study question is state-marginal matching under an f-divergence. | Narrower state-marginal method. |
| [IQ-Learn](estimators/iq_learn.md) | Inverse soft-Q learning or imitation diagnostics are the estimator of interest. | Preview diagnostic. |

On a small, well-specified tabular problem, these alternatives do not answer a
better question than NFXP. Their value appears when one of the complications on
this page is real.

## Linear Reading Guide

If you are reading the docs in order, use this route.

1. Read [NFXP](estimators/nfxp.md) first. It defines the structural benchmark
   and the normalization issues that recur across the docs.
2. Read [CCP](estimators/ccp.md) next if you want the same finite DDC target
   without repeated nested solves.
3. Read [TD-CCP](estimators/tdccp.md) if transition-density modeling is the
   estimation bottleneck.
4. Read [MCE-IRL](estimators/mce_irl.md) when demonstrations define the
   problem and reward features are supplied.
5. Read [Neural MCE-IRL](estimators/deep_mce_irl.md) when the reward is
   nonlinear and transitions are known.
6. Read [AIRL](estimators/airl.md) when state-only reward transfer is the
   object.
7. Read [NeuralAIRL](estimators/neural_airl.md) when that reward is nonlinear.
8. Read [AIRL-Het](estimators/airl_het.md) when anchored latent heterogeneity
   is the object.
9. Read [GLADIUS](estimators/gladius.md) when high-dimensional offline state
   features make repeated dynamic-programming solves unattractive and projected
   action contrasts are enough.
10. Read [Other Estimators](estimators/other.md) when the complication is value
   approximation, constrained structural optimization, finite-horizon planning,
   f-divergence matching, or inverse soft-Q diagnostics.
