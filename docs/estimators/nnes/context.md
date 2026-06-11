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

## Paper Nuances

NNES combines two ideas. From SEES, it takes a flexible approximator for the
value object. From NPL, it takes policy iteration on conditional choice
probabilities. That second choice is what gives the estimator its paper-level
inference claim: at the fixed point, the policy-iteration map has a
zero-Jacobian property, so the likelihood score is Neyman-orthogonal to
first-stage policy and value approximation error.

The practical implication is narrow but important. NNES can use a neural value
network without adding an explicit debiasing correction to the structural
score, as long as the first-stage approximation is accurate enough. The paper
states this as a fourth-root-style requirement: the nuisance approximation
error must be small enough that it enters the structural score only at second
order.

This is not a model-free estimator. The paper's NNES uses the transition law
inside policy evaluation. Model-free NNES is discussed as a future extension,
not as the current package claim. When transitions are unavailable, use a
transition-free estimator such as TD-CCP instead.

On small tabular problems, exact NFXP can still be the better estimator. NNES
is attractive when the value object is too large, smooth, encoded, or
continuous for repeated exact Bellman solves to be the natural workflow.

## Source Ideas

The source target is Nguyen's neural network estimation of structural dynamic
models. In EconIRL, the reported path keeps a parametric structural reward and
uses the neural network for value approximation. The simulation study is
therefore about recovering reward, policy, value, Q, and counterfactual
objects from a known-truth structural DGP, not about identifying an
unrestricted neural reward.

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
