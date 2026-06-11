# IQ-Learn

## Overview

IQ-Learn parameterizes soft Q-values directly and recovers a Bellman-implied
reward as a byproduct. The current package position is a research preview for
imitation quality and Q-based diagnostics.

Use the structural estimators when the project requires reported
counterfactual policy evidence.

## Source Papers

This page draws on {ref}`Garg et al. (2021) <garg-2021>` for IQ-Learn and
inverse soft-Q learning.

## When to Use

Use IQ-Learn when:

- inverse soft-Q learning is the estimator of interest;
- you want to compare policy imitation with structural reward recovery;
- transitions are available for the inverse Bellman reward calculation.

Avoid IQ-Learn when you need structural counterfactual evidence in this
package.

## Basic Usage

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
```

Use `q_type="neural"` for a neural Q diagnostic, but keep evidence scope
separate from imitation accuracy.

## Evidence

IQ-Learn is available as a research preview. The current evidence is strongest
for imitation and Q diagnostics rather than full structural reward recovery.

The low- and high-dimensional action-dependent DGPs are synthetic dynamic
choice benchmarks with action-specific rewards; the high-dimensional version
uses encoded states. The state-only diagnostic removes action-varying rewards.
The machine-readable results file records expert state and state-action coverage as
hard gates. Treat recovered rewards and counterfactuals as diagnostics unless
`summary.metadata["expert_state_coverage"] == 1.0` and
`summary.metadata["expert_state_action_coverage"] >= 0.95`, and all structural
recovery gates pass. Sparse support is a warning, not evidence for structural
counterfactual validity.

## Further Reading

- Machine-readable results file: [iq_learn_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/iq_learn/iq_learn_results.json)
