# Bayesian IRL Paper Context

Primary source: Ramachandran and Amir (2007) for Bayesian inverse
reinforcement learning. Broader paper routing lives in `../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Reward prior | Gaussian prior over parameters | Controlled by `prior_sigma`. |
| Demonstration likelihood | Softmax policy likelihood after solving MDP | Scaled by `confidence`. |
| Posterior sampling | Metropolis-Hastings chain | Controlled by proposal scale, burn-in, and sample count. |
| Point estimate | Posterior mean | Returned as parameters. |
| Uncertainty | Posterior standard deviation | Not a Hessian SE. |
| Convergence evidence | Missing release artifact | Needs traces and calibration before public docs. |

## Derivation Checklist

1. Define prior over reward parameters.
2. Define the induced policy likelihood after solving the MDP.
3. Derive the Metropolis-Hastings acceptance ratio.
4. Explain burn-in and posterior summaries.
5. Explain reward gauge/normalization for posterior interpretation.
6. Specify validation requirements before release exposure.

## Release Gap

The estimator has tests but no tracked validation JSON. Any future public page
needs posterior calibration, known-truth recovery, acceptance-rate diagnostics,
and counterfactual checks.
