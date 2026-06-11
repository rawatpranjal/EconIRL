# f-IRL

## Overview

f-IRL learns a reward by matching expert and model visitation distributions
under an f-divergence. The paper-side target is state-marginal matching with a
state-only reward.

Action-dependent DDC examples are comparison exercises in the current package.
The supported structural target is state-marginal reward recovery.

## Source Papers

This page draws on {ref}`Ni et al. (2020) <ni-2020>` for f-IRL and
state-marginal matching.

## When to Use

Use f-IRL when:

- the reward target is state-only;
- state marginal matching is the study question;
- transitions are available for policy evaluation;
- you want an f-divergence IRL baseline rather than a likelihood-based DDC
  estimator.

Avoid f-IRL for generic action-dependent structural DDC reward recovery.

## Basic Usage

```python
from econirl.estimation import FIRLEstimator

estimator = FIRLEstimator(
    f_divergence="fkl",
    marginal_space="state",
    reward_scope="state",
)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
```

Use `marginal_space="state_action"` only when the project is explicitly about a
state-action comparison exercise.

## Evidence

The current simulation study uses the paper-faithful state-marginal f-IRL DGP.
That is the structural scope reported by this page.

The state-marginal f-IRL DGP uses state-only rewards and state-marginal
matching.

## Further Reading

- Machine-readable results file: [f_irl_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/f_irl.json)
