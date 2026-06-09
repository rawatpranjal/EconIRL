# IQ-Learn

## Overview

IQ-Learn parameterizes soft Q-values directly and recovers a Bellman-implied
reward as a byproduct. The current package position is a research preview for
imitation quality and Q-based diagnostics.

Use the structural estimators when the project requires reported
counterfactual policy evidence.

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

The low- and high-dimensional action-dependent DGPs are known-truth dynamic
choice benchmarks with action-specific rewards; the high-dimensional version
uses encoded states. The state-only diagnostic removes action-varying rewards.

## Further Reading

- Machine-readable artifact: [iq_learn_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/iq_learn/iq_learn_results.json)
