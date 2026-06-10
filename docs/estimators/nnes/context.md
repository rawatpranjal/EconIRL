# Context

NNES is a structural dynamic discrete choice estimator for cases where the
reward remains finite-dimensional but the value function is too large or too
smooth to treat as a full tabular object. It estimates reward parameters while
training a neural approximation to the continuation value.

The estimator sits near NFXP, CCP, MPEC, and SEES. NFXP repeatedly solves an
exact tabular dynamic program. CCP uses first-stage choice probabilities and
Hotz-Miller inversion. MPEC exposes the Bellman equation as a constrained
optimization problem. SEES uses a deterministic sieve for the value object.
NNES uses a neural value approximation inside an NPL-style policy iteration.

## What NNES Adds Beyond NFXP

NFXP is the clean reference when the dynamic program is small enough to solve
exactly inside the likelihood. NNES targets the same kind of structural reward
parameter, but it does not require the value function to be stored as one free
number per tabular state. That is the practical distinction: keep the structural
reward low-dimensional, and let a neural network carry the high-dimensional or
smooth continuation-value nuisance.

NNES is therefore not "NFXP but neural rewards." The neural object is the value
function approximation. The reward remains finite-dimensional so the fitted
object can still support structural interpretation, standard errors, and
counterfactual policy analysis.

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

## Original Paper Simulation

Nguyen's main Monte Carlo reproduces a Rust-style bus-engine replacement
environment with two independent modules. The state is a continuous mileage
vector, actions are joint keep/replace decisions, and the true value function
is additively separable across modules. NNES is deliberately not told that
separability. It trains one two-dimensional value network, while the oracle
NFXP benchmark solves two one-dimensional dynamic programs using the true
decomposition.

The baseline design simulates 50 buses, discards 10 burn-in periods, keeps 20
periods, and therefore gives 1,000 observations per replication. The reported
tables average 100 replications at discount factor `0.9`. NNES closely matches
oracle NFXP means and standard errors, while taking more CPU time in that small
separable benchmark because the network has to learn the structure that oracle
NFXP is handed.

The paper also extends the design to four independent modules, with a
four-dimensional state and 16 joint actions. The same pattern holds: NNES
matches the oracle closely across the structural parameters, and the remaining
runtime gap reflects the oracle's module-wise shortcut rather than worse
statistical recovery. Appendix simulations show why value-function anchoring
matters when the discount factor is very high.

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
