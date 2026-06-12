# Context

Every structural estimator for dynamic discrete choice faces the same
problem: the agent's value function depends on the parameters being
estimated, so computing choice probabilities requires solving a dynamic
program first. NFXP makes that dependence explicit by nesting the Bellman
fixed point inside each likelihood evaluation. Rust's bus-engine replacement
model is the canonical example: a bus operator decides each period whether to
keep a deteriorating engine or pay a flat cost to replace it. Observed choices
identify the operating cost and replacement cost because the full dynamic
program links today's choices to tomorrow's states.

## Source Ideas

The primary source is {ref}`Rust (1987) <rust-1987>`, which introduces the
bus-engine replacement model and the nested fixed-point algorithm. The inner
loop solves the soft Bellman equation to obtain the value function; the outer
loop maximizes the conditional log likelihood over the reward parameters.

{ref}`Iskhakov et al. (2016) <iskhakov-2016>` compare the nested fixed-point
approach to constrained-optimization alternatives and establish benchmarks for
the hybrid inner solver used in the package: safe successive approximation far
from the fixed point, then Newton-Kantorovich updates near the solution.

## Where NFXP Fits

NFXP is the reference estimator for tabular structural estimation. CCP and
MPEC are useful comparisons because they target the same structural object
with different computational strategies. NNES and TD-CCP become attractive
when exact nested Bellman solves are too expensive. UFXP achieves the same
asymptotic efficiency as NFXP at a fraction of the cost by eliminating the
value-function dependence before the parameter search; the trade is that UFXP
is an asymptotic equivalent rather than the exact finite-sample MLE.
