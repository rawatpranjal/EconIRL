# Pre-Estimation Checks

Read this page before fitting TD-CCP. The goal is to catch weak feature support,
weak successor-tuple support, and unstable continuation bases before the
transition-density-free shortcut is interpreted structurally.

TD-CCP is most reliable when the reward features, observed choices, and panel
transitions support the structural parameter target. These checks are meant to
catch problems before a result is treated as recovery.

## Before Fitting

Confirm that the reward is written as a finite set of known features. The
features should have enough variation to identify the parameters, and one
action or reward level should be normalized so the model has an anchor.

Check that every relevant state-action pair has enough observed support. TD-CCP
uses observed successor tuples to learn continuation terms, so weak support in
the current or next state-action data can make the recursion unstable.

Finally, check the first-stage choice probabilities. The log correction used
by TD-CCP is unstable when fitted choice probabilities are too close to zero.

## During Fitting

Watch the conditioning of the basis used for the continuation terms. A nearly
singular basis can make the projected TD equations unstable even when the
likelihood optimizer appears to run.

For locally robust inference, split folds by individual rather than by row.
This keeps the held-out moment evaluation separate from the quantities learned
on the other fold and matches the covariance unit used in the simulation
results file.

## After Fitting

Do not judge the run only by a returned parameter vector. Check that the final
moment is small, the correction recursion is stable, the optimizer reached a
stationary point, and reported standard errors are finite and positive.

Also keep estimation and evaluation separate. The parameter-estimation step
should not silently use a transition tensor. Transition tensors may enter later
for policy values, Q functions, or counterfactual evaluation.

## Current Simulation Case

The simulation study uses two encoded state coordinates and a finite linear
reward. Action 0 is fixed as the baseline action, leaving six reward
parameters for the two non-baseline actions.

| Check | Current results file |
| --- | --- |
| States and actions | 81 states, 3 actions |
| Reward target | 6 finite reward parameters |
| Reward form | Linear in encoded state features |
| Choice model | Logit with degree-2 state features |
| TD basis | Encoded semigradient basis, degree 2 |
| Fold split | By individual |
| Inference | Cross-fitted locally robust standard errors |
| Max moment norm | 7.91e-06 |
| Max correction residual norm | 0.002610 |
| Optimizer status | Preliminary and final robust folds converged |

See the [simulation study page](validation.md) for the full results file,
numerical checks, and Monte Carlo standard-error check.
