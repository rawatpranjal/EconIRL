# Identification Boundary

## Important Links

- [AIRL Overview](../airl.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Counterfactuals](counterfactuals.md)
- [AIRL-Het](../airl_het.md)

AIRL separates a reward candidate from potential-based shaping. That
separation has a narrow positive result.

## Potential-based shaping

The transformed reward

$$
r'(s,a,s') = r(s,a,s') + \beta h(s') - h(s)
$$

can preserve optimal behavior under the original dynamics. Behavior matching
alone therefore does not identify the reward used for a new transition system.

AIRL constrains its discriminator score to

$$
f(s,a,s') = g(s) + \beta h(s') - h(s).
$$

Both the true reward and the recovered reward must depend only on state. Under
deterministic, decomposable dynamics, Fu et al. show that if AIRL recovers the
optimal discriminator score, $g$ equals the true reward up to a constant.

## Why state-only matters

An action-dependent payoff contains a within-state action contrast. A
state-only reward assigns the same current reward to every action. It cannot
represent that contrast directly.

Allowing an unrestricted $g(s,a)$ removes the original disentanglement result.
The state potential no longer separates every action-dependent reward term from
shaping. A fitted policy can look reasonable while the recovered reward fails
under new dynamics.

For this reason, the public `AIRL` class rejects action-dependent features. It
does not expose a switch that disables this scientific boundary.

## Context and heterogeneity

Observed context and latent segments change the reward target. The public AIRL
class rejects `context=` before optimization. AIRL-Het provides the separate
anchored heterogeneous design. Its assumptions and evidence are documented on
the [AIRL-Het page](../airl_het.md).

## What is reported

The fitted `reward_` is centered over states. Normalized reward RMSE compares
reward shape after removing location and positive scale differences. Policy,
value, Q, transfer policy, and regret measures remain necessary because reward
shape alone does not establish useful behavior.

Raw discriminator weights are descriptive optimizer coordinates. Do not label
them structural preference coefficients or attach structural coefficient
standard errors to them.
