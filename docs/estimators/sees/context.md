# Context

SEES is a structural dynamic discrete choice estimator for cases where exact
fixed-point methods become expensive but the value function still has a useful
low-dimensional representation. It estimates reward parameters and sieve
coefficients jointly under a Bellman-equilibrium penalty.

The estimator sits between MPEC and NNES. MPEC keeps one value variable per
state and enforces the Bellman equation as an equality constraint. SEES
replaces the value vector with a deterministic sieve expansion and penalizes
Bellman residuals. NNES replaces the deterministic sieve with a neural value
approximation.

## Source Ideas

The source target is Luo and Sang's sieve estimation approach for dynamic
discrete choice. The central idea is to estimate the structural payoff
parameters while approximating the continuation value with basis functions.
As the basis and penalty become rich enough, the estimator approaches the
structural likelihood target without requiring a full exact Bellman solve at
each likelihood evaluation.

The identification requirements match the other structural estimators. Reward
scale and location need a normalization, transitions must be separated from
payoffs, and reward features need enough action variation to identify
structural parameters.

## Where SEES Fits

Use SEES after checking the exact structural estimators. NFXP is the cleanest
reference when the state space is small. MPEC is useful when Bellman-constraint
diagnostics are the goal. SEES is useful when a deterministic basis can
represent the value function more compactly than a full tabular value vector.

SEES is not a black-box rescue for weak identification. If the chosen basis
cannot represent the Bellman solution, or if the reward features lack action
variation, the estimator can satisfy optimizer tolerances without recovering
the structural object.
