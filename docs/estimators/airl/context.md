# Context

Structural IRL estimators like NFXP recover a reward that is disentangled from
dynamics by construction: the likelihood separates flow utility from transition
parameters. Adversarial IRL does not have a likelihood in that sense. It trains
a discriminator to distinguish expert transitions from policy-generated ones,
and hopes that training pressure forces the reward component to track the true
signal.

AIRL's contribution over plain adversarial imitation is its discriminator
structure. Instead of a black-box classifier, the discriminator logit is
constrained to decompose as a reward term plus a potential-based shaping term.
This structure is designed to let the optimizer carve out the transferable
reward component and absorb dynamics-dependent terms into the shaping network.

## Source Ideas

The construction comes from {ref}`Fu, Luo, and Levine (2018) <fu-2018>`. The
key result is that when the reward is a function of state only and the MDP
satisfies decomposability, the adversarial game at its optimum recovers the
true reward up to constants. The potential-based shaping term absorbs the
value-like quantity that would otherwise bleed into the reward estimate, making
the recovered reward invariant to dynamics changes.

## Where AIRL Fits

AIRL sits in the behavioral IRL family alongside MCE-IRL, f-IRL, and GLADIUS.
It does not recover structural parameters in the same parameterization as the
data-generating process, and it does not report standard errors. Its score is
behavioral: how close is the recovered policy to the oracle policy, and how
much welfare does it lose in counterfactual worlds.

Within the adversarial subfamily, AIRL is the tabular reference. AIRL-Het
extends the design to heterogeneous agents and anchored action-dependent rewards;
that extension is documented separately. GAIL is a simpler adversarial baseline
without the shaping structure.

The state-only restriction is not a package limitation. It follows from the
original identification theorem. When the DGP has action-dependent payoffs,
the structured discriminator cannot separate reward from shaping in the
action-contrast direction, and validation confirms this with an explicit
failed diagnostic.
