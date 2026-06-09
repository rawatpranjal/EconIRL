# Context

MPEC was introduced for structural models where an equilibrium or dynamic
programming condition can be enforced directly inside the estimator. In dynamic
discrete choice, it estimates the same primitive payoff parameters as NFXP
while treating the value function as an optimization variable.

The practical appeal is diagnostics. NFXP hides the Bellman fixed point inside
each likelihood evaluation. MPEC writes that fixed point as an equality
constraint, so the fitted result can report the final Bellman constraint
violation directly.

## Source Ideas

The primary source is Su and Judd's constrained-optimization approach to
structural estimation. Iskhakov, Rust, Schjerning, and Seo compare MPEC-style
and NFXP-style numerical strategies in dynamic discrete choice settings.

The core identification lesson matches NFXP and CCP. Reward scale and
location need a normalization. Transitions need to be separated from payoffs.
Reward features need enough action variation to identify structural
parameters.

MPEC adds one more requirement. The constrained optimization problem must be
small and well conditioned enough that the optimizer can jointly move reward
parameters and value variables.

## Where MPEC Fits

MPEC targets the same structural reward object as NFXP and CCP in finite
tabular dynamic discrete choice models. It is useful as a constrained
likelihood cross-check when NFXP is the reference and when Bellman constraint
diagnostics are important.

CCP is usually faster when first-stage choice probabilities are well
supported. NNES, SEES, and TD-CCP become attractive when the value-function
constraint is too large for exact tabular constrained optimization.
