# Pre-Estimation Checks

## Important Links

- [TD-CCP overview](../tdccp.md)
- [Quick Start](quick_start.md)
- [Evidence](validation.md)
- [Bus Engine Example](rust_bus.md)

Run these checks before interpreting a TD-CCP fit. The parameter stage avoids a
transition density, but it still needs reward identification and informative
successor tuples.

## Reward Design

The reward must be linear in a finite set of known features. Check both the raw
feature rank and the action-contrast rank. The contrast rank must equal the
number of parameters. A state-only column repeated across all actions cannot be
identified from choices.

Choose an explicit reward normalization. A common design sets the baseline
action's feature vector to zero. Keep the logit shock scale and discount factor
fixed.

## Panel Support

Inspect state coverage, state-action coverage, and action shares. Each recursive
equation uses current and successor state-action pairs. Thin support can make
the first-stage choice probabilities and TD projections unstable.

The public wrapper rejects out-of-range states and actions. It also reports
states with only one observed action. Treat those states as a design problem,
not an optimizer problem.

The backward correction learns the distribution of predecessor choices given
the current choice and state. Simulated panels should start from the stationary
distribution or discard a sufficient burn-in period. A short panel that starts
from an arbitrary state distribution can distort this backward recursion.

## Choice Probabilities

`ccp_method="frequency"` estimates tabular choice frequencies with smoothing.
It is appropriate when each state has strong support. `ccp_method="logit"`
fits a shared polynomial model. Set `ccp_use_encoder=True` to build that model
from encoded state coordinates. The paper recommends second- or third-order
polynomial terms for this first stage.

Choice probabilities near zero make the term
$\gamma_{\mathrm E}-\log P(a\mid x)$ unstable. Review minimum fitted
probabilities and action support before trusting the recursive shock term.

## TD Basis

For `basis_type="polynomial"`, inspect the conditioning of the
action-interacted state-index basis. For `basis_type="encoded"`, scale each
state coordinate and avoid redundant polynomial columns. A small ridge can
stabilize the normal equation. A pseudoinverse cutoff is available for weak
empirical directions.

The high-dimensional check appends 20 irrelevant state variables and compares
the resulting parameter error with the zero-nuisance design. It also shuffles
the next-state links. The shuffled design must produce a much larger dynamic
parameter error.

## Inference

Algorithm 2 must split by individual. A row split leaks transitions from the
same trajectory across folds and does not support the paper's inference
argument.

After fitting, inspect the following outputs.

- All point estimates and standard errors are finite.
- Standard errors are positive.
- The corrected moment norm is small.
- In each fold, the preliminary and locally robust optimizers meet their
  recorded stationarity tolerances.
- The covariance matrix is positive semidefinite up to numerical tolerance.
- The manager summary states the uncertainty method and its limitations.

## Transition Boundary

The transition tensor is not used in the structural parameter equations. The
public wrapper stores a supplied `(A, S, S)` tensor for policy, value,
simulation, and counterfactual methods. If no tensor is supplied, it estimates
one from the fitted panel for those post-fit tasks.
