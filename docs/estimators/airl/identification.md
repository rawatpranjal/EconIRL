# Identification Boundary

## Potential-Based Shaping

Any reward function of the form

$$
r'(s, a, s') = r(s, a, s') + \beta\, h(s') - h(s)
$$

is observationally equivalent to $r$ under the original dynamics: the optimal
policy is unchanged and so is the agent's value function. AIRL's discriminator
is structured to absorb this shaping term into $h_\phi$ and isolate $g_\theta$
as the transferable reward. Whether that separation succeeds depends on what
$g_\theta$ is allowed to depend on.

## The State-Only Guarantee

Fu et al. (2018) Theorems 5.1-5.2 give the positive result: when (i) the reward
is a function of state only, $g_\theta(s)$, and (ii) the MDP satisfies
decomposability, the discriminator at the adversarial optimum recovers the true
reward up to a constant. The shaping potential $h_\phi$ absorbs the
continuation-value terms that would otherwise contaminate $g_\theta$.

The package implements this setting directly: `reward_arg="state"` projects the
reward matrix onto the state subspace by averaging across actions before
computing the discriminator logit.

## Why Action-Dependent Rewards Break the Guarantee

When payoffs differ by action, the reward signal sits in the action-contrast
direction. A state-only $g_\theta(s)$ assigns the same flow utility to every
action at a state, so the discriminator cannot represent the action contrast.
In practice, the policy collapses toward uniform across actions and policy TV
stays far from the oracle regardless of training length.

Setting `reward_arg="state_action"` lets $g_\theta(s, a)$ differ by action, but
the shaping structure then cannot separate $g_\theta$ from an action-dependent
shaping term $\beta h(s', a') - h(s, a)$, because the potential is defined on
states, not state-action pairs. The disentanglement guarantee no longer applies.

The action-dependent diagnostic cell `airl_anchor_action_dependent` confirms
this: all eight numerical checks fail, with policy TV of 0.40 and regret values
in double digits.

## Connection to AIRL-Het

AIRL-Het (Lee, Sudhir, and Wang 2026) adds two design elements to recover
action-dependent rewards in dynamic discrete choice: an anchor action whose
reward is pinned to zero to fix the reward gauge, and an absorbing-state row
pinned to zero to fix the level. These anchors turn the adversarial game into
one that can identify an action-dependent reward surface. The implementation is
in `econirl.estimation.adversarial.airl_het`, documented separately.

## Practical Guidance

Use the state-only mode when the DGP or theory supports a state-only reward. If
the empirical setting has action-dependent payoffs (entry/exit, product choice,
capital investment), a state-only reward cannot match the data by construction,
not by tuning. Switching to AIRL-Het or MCE-IRL is the right move, not
increasing training length or reward learning rate.

A quick diagnostic: fit with `reward_arg="state"` and inspect policy TV. If TV
stays above 0.10 after 200 rounds and the discriminator loss plateaus above
log(2), the reward argument is almost certainly misspecified.
