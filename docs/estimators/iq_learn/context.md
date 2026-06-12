# Context

Most IRL estimators frame reward recovery as a min-max problem: choose a
reward function and a policy, then alternate between improving each. IQ-Learn
collapses this adversarial game into a single concave optimization over a soft
Q-function. The Q-function determines both the policy (through the softmax
mapping) and a Bellman-implied reward (through the inverse Bellman operator),
so there is no discriminator to train and no inner loop to stabilize.

## Source Ideas

The construction comes from {ref}`Garg et al. (2021) <garg-2021>`. The key
observation is that in a max-entropy policy framework the optimal policy is
always the softmax of Q, so it suffices to optimize over Q alone. The
chi-squared divergence objective adds a quadratic penalty on the temporal
difference term, which bounds the objective and prevents Q from drifting to
numerical overflow. The simple (TV-distance) objective is also implemented but
has no upper bound on a free tabular Q table and should not be used there.

## Where IQ-Learn Fits

IQ-Learn sits with AIRL, f-IRL, GLADIUS, and MCE-IRL in the IRL family. Like
the others it does not impose a structural Bellman fixed point as a hard
constraint. Unlike AIRL and f-IRL it avoids adversarial training; unlike
MCE-IRL and GLADIUS it does not start from a parameterized reward function but
from a parameterized Q directly.

The {ref}`Garg et al. (2021) <garg-2021>` paper shows that IQ-Learn
generalizes behavioral cloning: setting the divergence penalty weight to zero
recovers a pure log-likelihood maximizer over the Q-induced logit choice
probabilities. This connection makes IQ-Learn a natural bridge between BC and
Bellman-aware IRL, and the alpha hyperparameter controls where on that path
the estimator sits.

## Package Position

IQ-Learn is the right tool when you want an imitation policy without
adversarial training and a Q-based reward diagnostic alongside it. It is not
the right tool when structural counterfactual validity is required; the
current evidence shows that reward, value, and Q recovery do not pass their
checks on any tested cell. For structural use, see NFXP, CCP, or UFXP.
