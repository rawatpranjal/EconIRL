# Pre-Estimation Checks

TD-CCP can fail for reasons that are visible before optimization starts. Run
these checks before treating a result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Reward parameters are not identified when known features are collinear. |
| Reward normalization | Reward level and action baselines need an anchor. |
| State-action-successor coverage | TD estimates need support over observed current and next tuples. |
| Minimum positive CCP | The log CCP correction is unstable near zero. |
| Basis conditioning | Semi-gradient normal equations become unstable when the basis is nearly singular. |
| Bounded encoded features | Paper-style approximation arguments assume controlled feature support. |
| Individual-level fold split | Algorithm 2 inference should keep held-out moments separate from nuisance estimates. |
| Moment stationarity | The final `zeta` solve should have a small mean moment and optimizer gradient. |
| Lambda recursion residual | Locally robust correction needs a stable backward recursion. |
| State encoder meaning | Encoded bases should use real state features, not arbitrary labels. |
| Estimation/evaluation boundary | `theta` estimation should not silently use transition tensors reserved for evaluation. |
| Transition orientation for evaluation | Final policy/value checks expect action, state, next-state tensors. |

## Certified Hard-Case Checks

The certified TD-CCP hard case uses two-dimensional encoded state coordinates,
then builds a finite linear reward by interacting an intercept and the two
coordinates with non-baseline actions. Action 0 is fixed to zero as the utility
normalization.

| Check | Current state |
| --- | --- |
| Structural target | Finite-dimensional `theta`. |
| Feature source | Encoded state coordinates. |
| Reward form | Linear in known features. |
| Action normalization | Action 0 reward fixed to zero. |
| States and actions | 81 states, 3 actions. |
| Reward parameters | 6 finite parameters. |
| State-action support | Simulated from the known optimal policy. |
| First-stage CCP | Logit with degree-2 state features. |
| Basis | Encoded semigradient basis, degree 2. |
| Fold split | Cross-fitting by individual. |
| Inference path | Algorithm 2 cross-fitting with locally robust SEs. |
| Zeta moment | Max norm 7.91e-06 in the current artifact. |
| Lambda recursion | Max fold residual norm 0.002610 in the current artifact. |
| Optimizer stationarity | Preliminary projected-gradient max 2.00e-07; final robust folds 2/2 converged. |
| Estimation transition density | Not used. |
| Evaluation transitions | Supplied after estimation for oracle checks. |

See the [validation page](validation.md) for the generator script, rendered
table source, JSON artifact, and hard gates.

## Common Risk Patterns

Raw neural reward matrices do not provide a finite true `theta`, so parameter
recovery is not a meaningful certification gate. Sparse action support makes
the CCP correction noisy. High-dimensional encoded bases can be weakly
conditioned unless the basis is chosen carefully. Transition tensors with the
wrong orientation can make post-estimation policy/value diagnostics look
plausible while evaluating the wrong model.
