# Context

NNES is a structural dynamic discrete choice estimator for cases where the
reward remains finite-dimensional but the value function is too large or too
smooth to treat as a full tabular object. It estimates reward parameters while
training a neural approximation to the continuation value.

The estimator sits near NFXP, CCP, MPEC, and SEES. NFXP repeatedly solves the
exact dynamic program. CCP uses first-stage choice probabilities and
Hotz-Miller inversion. MPEC exposes the Bellman equation as a constrained
optimization problem. SEES uses a deterministic basis for the value object.
NNES replaces that basis with a neural value approximation and an NPL-style
policy iteration.

## Source Ideas

The source target is Nguyen's neural network estimation of structural dynamic
models. In EconIRL, the certified path keeps a parametric structural reward
and uses the neural network for value approximation. The validation claim is
therefore about recovering reward, policy, value, Q, and counterfactual
objects from a known-truth structural DGP, not about identifying an unrestricted
neural reward.

The identification requirements match the other structural estimators. Reward
scale and location need a normalization, transitions must be separated from
payoffs, and reward features need enough action variation to identify
structural parameters.

## Where NNES Fits

Use NNES after checking the exact structural estimators. NFXP is the clearest
reference when the state space is small. CCP is faster when empirical support
is strong. SEES is a deterministic approximation route. NNES is the flexible
value-approximation route when encoded states or richer state features make a
neural value function attractive.

NNES is not a fallback for weak identification. If reward features are low
rank, action support is sparse, or transitions are misoriented, the neural
value path can still produce a fitted policy without a credible structural
claim.
