# Context

MCE-IRL is the tabular maximum causal entropy inverse reinforcement learning
estimator associated with Ziebart's dynamic IRL formulation. It estimates a
reward vector by asking which reward makes the demonstrated feature counts look
like the feature counts induced by the soft optimal policy.

The causal part matters. The policy at a state is conditioned on what the agent
knows at that decision time, and future uncertainty is integrated through the
soft Bellman equation rather than treated as a realized full trajectory.

## Source Ideas

The core objective is moment matching under an entropy-regularized dynamic
policy. The estimator solves a soft dynamic program for a candidate reward,
computes policy-induced occupancy and feature counts, and updates the reward
parameters until the model feature counts match the demonstrated feature
counts.

In finite tabular problems this makes MCE-IRL a bridge between inverse
reinforcement learning and dynamic discrete choice. It uses the same soft
choice structure as logit DDC models, but the estimation target is the reward
that explains demonstrations through feature moments.

## Where MCE-IRL Fits

MCE-IRL is useful when transitions and reward features are supplied and the
research question is reward recovery. NFXP, CCP, MPEC, SEES, NNES, and TD-CCP
target structural reward parameters through likelihood or estimating-equation
paths. Deep MCE-IRL keeps the causal-entropy objective but replaces the fixed
tabular reward vector with a neural reward map.
