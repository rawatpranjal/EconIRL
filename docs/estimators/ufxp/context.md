# Context

Every full-solution structural estimator pays the same bill: each candidate
parameter vector triggers a dynamic-programming solve, and the optimizer
visits many candidates. NFXP pays it with a nested fixed point, MPEC moves the
fixed point into the constraint set of a large optimizer. UFXP refuses to pay
it at all.

## Source Ideas

The construction comes from {ref}`Bray <bray-2019>`. Instead of maximizing a
likelihood, UFXP works with the first-order conditions of Bellman's equation:
under logit shocks, the log-odds of any two actions at a state must equal the
difference of their choice-specific values. Those conditions involve the value
function, which normally forces a solve per candidate. Bray's observation is
that the conditions only ever use the value function through fixed linear
functionals, and each such functional has a dual representation as a fixed
point in the empirical choice probabilities alone — independent of the
parameters. Computing the duals once, before the search, removes the value
function from the problem.

{ref}`Oguz and Bray (2026) <oguz-bray-2026>` add the optimal weighting of
those conditions, prove the weighted estimator is as asymptotically efficient
as maximum likelihood, and use the construction to train neural-network
utility functions inside dynamic discrete choice models. The econirl
implementation covers the linear-utility case, where the optimally weighted
estimator collapses to a single closed-form solve.

## Where UFXP Fits

UFXP sits with NFXP, CCP, and MPEC in the structural family: it recovers the
utility parameters in the same gauge as the data-generating process, supports
counterfactual re-solving, and reports standard errors. Like CCP it starts
from inverted empirical choice probabilities, so it shares CCP's sensitivity
to thin state coverage; unlike CCP it scores all of Bellman's restrictions
with efficient weights rather than inverting once, and its optimal weighting
restores maximum-likelihood efficiency. Against NFXP the trade is plain: NFXP
is the exact finite-sample MLE at meaningful compute cost, UFXP is its
asymptotic equal at almost no cost.
