# Quick Start

This page shows IQ-Learn as an imitation and diagnostic estimator. Coverage
metadata matter because off-support Q values are extrapolated, not learned from
expert choices.

```python
from econirl.estimation import IQLearnConfig, IQLearnEstimator

config = IQLearnConfig(
    q_type="tabular",
    divergence="chi2",
    alpha=1.0,
)
estimator = IQLearnEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
print(summary.converged)
print(summary.metadata["expert_state_coverage"])
print(summary.metadata["expert_state_action_coverage"])
```

Fitted attributes follow the same convention as other estimators:

| Attribute | Meaning |
| --- | --- |
| `parameters` | Q parameters: feature coefficients for `q_type="linear"`, per-cell rewards for tabular. |
| `converged` | Whether L-BFGS-B reported convergence. |
| `log_likelihood` | Log-likelihood of expert actions under the Q-induced policy. |
| `policy` | Choice probabilities pi(a given s), shape (n_states, n_actions). |
| `value_function` | V(s) = sigma * logsumexp(Q(s, :) / sigma), shape (n_states,). |
| `metadata["expert_state_coverage"]` | Fraction of MDP states visited in the expert panel. |
| `metadata["expert_state_action_coverage"]` | Fraction of state-action pairs visited. |

## Q Parameterizations

```python
# Tabular: free Q(s, a) for each cell -- no feature structure
config = IQLearnConfig(q_type="tabular", divergence="chi2", alpha=1.0)

# Linear: Q(s, a) = phi(s, a)^T theta, matches the utility feature spec
config = IQLearnConfig(q_type="linear", divergence="chi2", alpha=1.0)

# Neural: small feedforward net mapping state features to Q(s, :)
config = IQLearnConfig(q_type="neural", divergence="chi2", alpha=1.0,
                       hidden_dim=64, num_layers=2)
```

The linear parameterization is the most useful for structural interpretation:
it constrains Q to live in the same feature space as the utility, so the
recovered theta is directly comparable to the data-generating parameters and
propagates to unvisited state-action pairs.

## Common Pitfall

Do not pair `q_type="tabular"` with `divergence="simple"`. The simple
objective has no upper bound on a free Q table and the optimizer drives Q to
numerical overflow (value RMSE can exceed 1e21). The chi-squared objective
with `alpha >= 1` keeps the problem bounded, so it is the documented setting
for tabular IQ-Learn.
