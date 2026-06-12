# Context

MCE-IRL is the tabular maximum causal entropy inverse reinforcement learning
estimator from Ziebart's dynamic IRL formulation. It estimates the reward
parameters that make the model's soft optimal policy reproduce the demonstrated
feature counts.

The causal part matters. At each state, the policy is conditioned only on
what the agent knows at that decision time. Future uncertainty is integrated
through the soft Bellman equation rather than realized as a full trajectory.
This is the critical distinction from trajectory maximum-entropy IRL and the
reason MCE-IRL is the default entropy IRL route for dynamic discrete choice
comparisons in this package.

## Source Ideas

The core objective is moment matching under an entropy-regularized dynamic
policy. For a candidate reward, the estimator solves a soft dynamic program,
computes the implied occupancy and feature counts, and updates the reward
parameters until the model moments match the demonstrated moments.

In finite tabular problems this connects MCE-IRL to logit dynamic discrete
choice: both use the same soft choice structure, but MCE-IRL estimates the
reward through feature moments rather than through a likelihood alone.

## Where MCE-IRL Fits

MCE-IRL is a good fit when transitions and reward features are supplied and
the main question is reward and policy recovery. NFXP, CCP, MPEC, SEES, NNES,
and TD-CCP target structural reward parameters through likelihood or
estimating-equation paths and report standard errors in the structural gauge.
Deep MCE-IRL keeps the causal-entropy objective but replaces the fixed tabular
reward vector with a neural reward map.
