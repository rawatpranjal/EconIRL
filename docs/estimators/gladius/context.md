# Context

Structural estimators like NFXP and UFXP solve or invert a dynamic program at
every candidate parameter vector. That works cleanly when the state space is
small and transitions are tabular, but the computational and representational
cost rises quickly as the state dimension grows. GLADIUS takes a different route:
it trains neural networks to approximate the Q-function and expected continuation
value directly from transition data, then recovers structural parameters by
projecting implied rewards onto a low-dimensional feature basis.

## Source Ideas

The construction comes from {ref}`Kang, Yoganarasimhan, and Jain (2025) <kang-2025>`.
The paper proposes an offline IRL approach for dynamic discrete choice models
where the reward function is latent but an anchor action with known rewards is
available. Under logit taste shocks, the Bellman equation relates Q-values,
continuation values, and rewards. The paper trains a Q-network and a separate
continuation-value network (the zeta network) using an alternating update rule,
then uses the anchor Bellman residual to pin the absolute level of Q-values.
Without the anchor, Q is identified only up to a state-dependent additive
constant, which propagates into implied rewards through asymmetric transitions.

The econirl implementation covers the IRL setting, where rewards are latent and
the anchor is supplied by the user. When known anchor rewards are omitted, the
estimator falls back to NLL-only training, which identifies the policy but not
the structural reward in the full Bellman sense.

## Where GLADIUS Fits

GLADIUS sits in the behavioral family alongside MCE-IRL, AIRL, and IQ-Learn. It
does not require solving a dynamic program for each parameter draw, so it scales
to state spaces that make the structural family expensive. The cost is that the
identified object is not in the same parameterization as the data-generating
reward. Action-difference projection removes a state-dependent constant. The
remaining projected parameters reflect the reward only up to what the anchor
pins. Full
structural counterfactual validity requires the raw Bellman reward and value
recovery checks to pass, and those do not currently pass on the evaluation cells.
GLADIUS is therefore most useful as a high-dimensional approximation and policy-
imitation tool rather than as a counterfactual engine.

Against simpler behavioral methods, GLADIUS adds explicit Bellman structure
through the Q/zeta architecture and the anchor Bellman loss, which gives it a
path toward structural parameter recovery that policy-gradient or behavioral
cloning methods do not have.
