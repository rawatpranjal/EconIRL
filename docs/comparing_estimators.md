# Comparing Core Estimators

This page is a map across the core estimators. It compares what each method
tries to recover, what data it needs, and where its identification claim comes
from.

Detailed math, system views, examples, and usage live on the estimator pages.
Paper-number checks live in [Replications](replications.md). Problem-level
simulation results live in [Simulation Studies](simulation_studies/index.md).

## Core Lineage

NFXP is the reference. It estimates a small tabular structural dynamic
discrete choice model by maximum likelihood. It solves the agent's dynamic
program inside the likelihood for each candidate reward parameter.

The rest of the core exists because one part of that setup becomes too slow,
too restrictive, or too hard to observe.

| Estimator | Main question | What changes relative to NFXP |
| --- | --- | --- |
| [NFXP](estimators/nfxp.md) | What is the exact tabular structural likelihood estimate? | Nothing. It is the reference under the maintained DDC assumptions. |
| [CCP](estimators/ccp.md) | Can we avoid repeated Bellman solves? | Uses first-stage choice probabilities and Hotz-Miller inversion. NPL iterates the pseudo-likelihood route. |
| [TD-CCP](estimators/tdccp.md) | Can we avoid transition-density estimation during parameter estimation? | Uses observed successor state-action pairs to estimate recursive terms. |
| [MCE-IRL](estimators/mce_irl.md) | Can demonstrations identify a reward? | Replaces structural likelihood with maximum causal entropy feature matching. |
| [Neural MCE-IRL](estimators/deep_mce_irl.md) | Can the reward be nonlinear? | Replaces fixed linear reward features with a neural reward map. |
| [AIRL](estimators/airl.md) | Can a state-only reward transfer across dynamics? | Separates state reward from shaping under the original AIRL assumptions. |
| [AIRL-Het](estimators/airl_het.md) | Can anchored rewards differ across latent segments? | Adds exit-action and absorbing-state anchors, then estimates segment-specific rewards. |
| [GLADIUS](estimators/gladius.md) | Can high-dimensional offline DDC avoid repeated solves? | Learns Q and continuation objects, then reads projected action contrasts. |

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
| [AIRL-Het](estimators/airl_het.md) | Serialized-content choice with latent types. | Anchored action-dependent rewards and persistent latent segments. |
| [GLADIUS](estimators/gladius.md) | High-dimensional offline dynamic choice. | Offline panels where tabular dynamic programming is too costly and action contrasts are enough. |

## Main Axes

### State Scale

| State setting | Natural estimators | Why |
| --- | --- | --- |
| Small tabular state space | NFXP, CCP, MCE-IRL, AIRL | The full grid can be enumerated. |
| Small or moderate tabular space with speed pressure | CCP | It keeps the NFXP target but avoids repeated solves. |
| Encoded or higher-dimensional state space with finite reward parameters | TD-CCP | It avoids transition-density modeling during estimation. |
| Nonlinear reward over tabular or encoded states | Neural MCE-IRL | The reward map is neural, but planning still uses supplied transitions. |
| Repeated choices with latent segments | AIRL-Het | It estimates segment-specific rewards and policies. |
| High-dimensional offline state features | GLADIUS | It learns Q and continuation objects instead of repeated tabular solves. |

### Reward Form

| Reward target | Core estimators | Main caution |
| --- | --- | --- |
| Finite linear structural reward | NFXP, CCP, TD-CCP | Needs action-contrast feature rank and a fixed normalization. |
| Linear IRL reward basis | MCE-IRL | Identified only inside the supplied feature basis. |
| Neural reward map | Neural MCE-IRL | The reward matrix is the object. The raw weights are not. |
| State-only transferable reward | AIRL | Matches the original AIRL claim only under its state-only assumptions. |
| Segment-specific action-dependent reward | AIRL-Het | Needs credible exit-action and absorbing-state anchors, persistent segments, and enough trajectory support per segment. |
| Projected action contrast | GLADIUS | Stronger than raw Bellman reward levels in the package route. |

### Transition Information

| Transition input | Estimators | Meaning |
| --- | --- | --- |
| Explicit transition tensor | NFXP, CCP, MCE-IRL, Neural MCE-IRL, AIRL, AIRL-Het | The estimator or policy update uses a transition model. |
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
| CCP NPL | Same finite target as NFXP. | Yes, in population. | Same as one-step CCP, plus convergence to the relevant NPL fixed point. |
| TD-CCP semigradient | Finite reward parameters. | Yes, in population. | Successor tuples, consistent CCPs, recursive terms in the projection span, support, and correct normalization. |
| TD-CCP neural | Same finite reward parameters. | Conditional. | Same target as semigradient TD-CCP, with enough data and capacity to learn the recursive terms. |
| MCE-IRL | Reward coefficients in supplied features. | Yes, in population. | Known transitions, true reward in the feature span, full-rank moments, support, and fixed normalization. |
| Neural MCE-IRL | Anchored reward matrix. | Conditional. | Known transitions, representable reward, sufficient occupancy, and exact optimization under a fixed anchor. |
| AIRL | State-only reward up to a constant. | Yes under the original AIRL assumptions. | State-only reward, decomposable dynamics, sufficient expert and learner samples, and adversarial equilibrium. |
| AIRL-Het | Segment-specific anchored $R_k(s,a)$. | Conditional. | Exit-action reward anchor, absorbing-state value anchor, correct segment count, segment separation, support, fixed discount, and exact policy solution. |
| GLADIUS dual anchor-moment | Projected action contrasts. | Conditional. | Credible anchor, learned Q and continuation objects, action-contrast rank, and support. |
| GLADIUS `q_only` | Full reward. | No. | Useful as a diagnostic mode, but not enough for reward recovery. |

## Reading Guide

Use the core estimators in this order.

1. Start with NFXP if the problem is small, tabular, and structurally specified.
2. Use CCP when NFXP is the right target, repeated solves are too costly, and
   first-stage CCP support is strong.
3. Use TD-CCP when transition-density modeling is the estimation bottleneck.
4. Use MCE-IRL when demonstrations define the problem and reward features are
   supplied.
5. Use Neural MCE-IRL when the reward is nonlinear and transitions are known.
6. Use AIRL when state-only reward transfer is the object.
7. Use AIRL-Het when anchored latent heterogeneity is the object.
8. Use GLADIUS when high-dimensional offline state features make repeated
   dynamic-programming solves unattractive and projected action contrasts are
   enough.
