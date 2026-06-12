# Context

Standard AIRL learns a reward through an adversarial discriminator, but its
shaping potential leaves the reward identified only up to potential-based
transformations. In action-dependent settings with absorbing exits, this
ambiguity is severe enough that the recovered reward cannot support structural
counterfactuals without further constraints. AIRL-Het resolves both problems at
once: it imposes two anchor normalizations that uniquely pin down the reward,
and it wraps the single-segment adversarial loop inside an EM algorithm that
discovers latent segments from pooled data.

## Source Ideas

The base adversarial construction comes from
{ref}`Fu, Luo, and Levine (2018) <fu-2018>`. In that setting a discriminator
distinguishes expert transitions from policy transitions through a shaped score
$f(s, a, s') = g(s, a) + \beta h(s') - h(s)$, and gradient descent on a
binary cross-entropy loss drives the reward toward the expert reward.

{ref}`Lee, Sudhir, and Wang (2026) <lee-sudhir-wang-2026>` show that, in a
serialized-content dynamic discrete choice environment with an absorbing exit,
two normalizations together uniquely identify the structural reward. Setting the
exit-action reward to zero and the absorbing-state shaping potential to zero
removes the residual degrees of freedom that make standard AIRL fail in that
class of problems. They then add latent consumer segments and an EM loop that
assigns trajectories to segments and updates each segment's adversarial reward.

## Where AIRL-Het Fits

AIRL-Het sits at the intersection of adversarial IRL and structural DDC. Like
MCE-IRL, GLADIUS, and base AIRL it estimates reward through a behavioral
objective rather than a likelihood, so standard errors for the reward parameters
are not available. Unlike them, its anchor normalization lets the recovered
reward support structural counterfactuals in the same gauge as the truth,
provided the anchor design is credible.

Against base AIRL, AIRL-Het adds two things: anchor identification for
action-dependent settings, and segment-specific reward and policy objects for
heterogeneous populations. The single-segment case with anchors is not
separately implemented; use AIRL-Het with `num_segments=1` if the anchor is the
only requirement.

Against the structural estimators (NFXP, CCP, MPEC, UFXP), AIRL-Het does not
maximize a likelihood and does not report standard errors. It is appropriate
when the research question is segment-level reward recovery and counterfactual
behavior, not when asymptotic inference on structural parameters is the target.
