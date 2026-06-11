# Bayesian IRL Internal Notes

Bayesian IRL is retained as a contributed estimator for posterior reward
reasoning. It uses Metropolis-Hastings sampling over reward parameters, solving
the MDP at each proposed parameter vector and evaluating the likelihood of the
observed actions under the induced softmax policy.

There is no current public RTD estimator page and no tracked validation JSON.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD page: none currently.
- Contrib implementation: `../../../src/econirl/contrib/bayesian_irl.py`.
- Validation result: none currently.
- Tests: `../../../tests/test_bayesian_irl.py`.

## Algorithm

The estimator samples from a posterior over reward parameters:

```text
log posterior(theta)
  = confidence * log p(data | theta)
    + log p(theta).
```

For each MCMC iteration:

1. propose `theta' = theta + epsilon`;
2. solve the MDP under `theta'`;
3. compute the log-likelihood of observed actions under the induced policy;
4. add the Gaussian prior log density;
5. accept or reject with the Metropolis-Hastings ratio;
6. after burn-in, summarize posterior samples.

The point estimate is the posterior mean. Posterior standard deviations are
reported as uncertainty summaries when available.

## Interpretation

Bayesian IRL is useful when the user wants:

- posterior uncertainty over reward parameters;
- prior sensitivity checks;
- small-model methodological comparisons;
- a non-asymptotic alternative to Hessian-based standard errors.

It is not currently release evidence because the repository lacks:

- known-truth posterior recovery artifacts;
- posterior calibration checks;
- convergence diagnostics such as acceptance rate targets and trace summaries;
- counterfactual validation;
- public RTD documentation.

## Debugging Order

1. Check reward feature dimension and prior scale.
2. Verify MDP solve tolerance before diagnosing MCMC behavior.
3. Inspect acceptance rate and log posterior trace.
4. Check burn-in and sample count.
5. Compare posterior mean policy against demonstrations.
6. Do not use posterior means for counterfactual claims without a validation
   artifact.

## Implementation Paths

- Contrib estimator: `../../../src/econirl/contrib/bayesian_irl.py`.
- Unit tests: `../../../tests/test_bayesian_irl.py`.

## Public Documentation Boundary

Do not add a public RTD estimator page until the package has a validation
runner, tracked result JSON, and posterior diagnostics.
