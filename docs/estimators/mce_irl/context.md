# Context

MCE-IRL is the tabular maximum causal entropy inverse reinforcement learning
estimator from Ziebart's dynamic IRL formulation. It estimates the reward vector
that makes the model's soft optimal policy reproduce the demonstrated feature
counts.

The causal part matters. The policy at a state is conditioned on what the agent
knows at that decision time. Future uncertainty is integrated through the soft
Bellman equation rather than treated as a realized full trajectory.

## Source Ideas

The core objective is moment matching under an entropy-regularized dynamic
policy. For a candidate reward, the estimator solves a soft dynamic program,
computes the implied occupancy and feature counts, and updates the reward
parameters until the model moments match the demonstrated moments.

In finite tabular problems, this makes MCE-IRL a bridge between inverse
reinforcement learning and dynamic discrete choice. It uses the same soft
choice structure as logit DDC models, but estimates the reward through feature
moments rather than through a likelihood alone.

## Where MCE-IRL Fits

MCE-IRL is a good fit when transitions and reward features are supplied and the
main question is reward recovery. NFXP, CCP, MPEC, SEES, NNES, and TD-CCP
target structural reward parameters through likelihood or estimating-equation
paths. Deep MCE-IRL keeps the causal-entropy objective but replaces the fixed
tabular reward vector with a neural reward map.
